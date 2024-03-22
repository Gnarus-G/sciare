#![allow(clippy::needless_return)]

use color_eyre::eyre::Context;
use primitives::*;
use splits::TextSplitter;

pub mod context {
    use crate::{llm, splits::TextSplitter};
    use sqlx::SqlitePool;

    pub struct Context<Splitter: TextSplitter> {
        pub conn_pool: SqlitePool,
        pub splitter: Splitter,
        pub llm: Box<dyn llm::Llm>,
    }
}

pub async fn search_documents<Splitter: TextSplitter>(
    ctx: &context::Context<Splitter>,
    phrase: String,
    limit: usize,
) -> color_eyre::Result<Vec<Chunk>> {
    eprintln!("[INFO] fetching all the chunks of documents we have");
    let chunks = sqlx::query_as!(
        CompleteChunk,
        r#"SELECT document_name, content, page_number, content_embedding FROM chunk"#
    )
    .fetch_all(&ctx.conn_pool)
    .await?;

    eprintln!("[INFO] generating embeddings for the search-phrase");
    let phrase_embedding = ctx.llm.create_embeddings(phrase).await?;

    const SIMILARITY_THRESHOLD: f64 = 0.18;
    eprintln!("[INFO] selecting only chunks above a score of {SIMILARITY_THRESHOLD} similarity with our search-phrase");
    let mut chunks = chunks
        .into_iter()
        .map(|chunk| {
            let sim = chunk.get_embedding().similarity_with(&phrase_embedding);
            (sim, Chunk::from(chunk))
        })
        .filter(|&(sim, _)| sim > SIMILARITY_THRESHOLD)
        .collect::<Vec<_>>();

    eprintln!("[INFO] sorting by similarity score");
    chunks.sort_by(|(a_sim, _), (b_sim, _)| b_sim.total_cmp(a_sim));

    let chunks = chunks
        .into_iter()
        .take(limit)
        .inspect(|(sim, _)| eprintln!("[DEBUG] found content with similarity score {}", sim))
        .map(|(_, chunk)| chunk)
        .collect();

    Ok(chunks)
}

pub async fn save_document<'s, Splitter: splits::TextSplitter>(
    ctx: &context::Context<Splitter>,
    document: &'s impl primitives::Document<'s>,
) -> color_eyre::Result<()> {
    let name = document.name();

    let maybe_record = sqlx::query!(r#"SELECT rowid from document WHERE name = ?"#, name)
        .fetch_optional(&ctx.conn_pool)
        .await
        .context("failed to query if document already exists")?;

    if maybe_record.is_some() {
        return Err(color_eyre::eyre::anyhow!(
            "There is already a document '{}' in the database.",
            name
        ));
    }

    eprintln!("[INFO] saving the document");

    let document_source = document.get_source_uri().as_str();

    sqlx::query!(
        r#"INSERT INTO document (name, source_uri) VALUES (?, ?)"#,
        name,
        document_source
    )
    .execute(&ctx.conn_pool)
    .await
    .context("failed to save a document")?;

    eprintln!("[INFO] chunking documents");
    let chunks: Vec<_> = document
        .get_text()?
        .into_iter()
        .enumerate()
        .flat_map(|(page_idx, page)| {
            ctx.splitter
                .split_text(page)
                .into_iter()
                .map(move |s| (page_idx, s))
        })
        .map(|(idx, c)| Chunk {
            document_name: document.name().to_string(),
            page_number: idx as i64 + 1,
            content: c,
        })
        .collect();

    eprintln!("[INFO] creating vector embeddings for each chunk");
    let chunks = ctx.llm.create_multiple_embeddings(chunks).await;

    eprintln!("[INFO] saving the document chunks");
    let query_string = format!(
        r#"INSERT INTO chunk (document_name, page_number, content, content_embedding) VALUES {}"#,
        chunks
            .iter()
            .map(|_| "(?, ?, ?, ?)".to_string())
            .collect::<Vec<String>>()
            .join(",")
    );

    let mut q = sqlx::query(&query_string);

    for chunk in chunks {
        q = q
            .bind(chunk.document_name)
            .bind(chunk.page_number)
            .bind(chunk.content)
            .bind(chunk.content_embedding.0);
    }

    q.execute(&ctx.conn_pool).await?;

    Ok(())
}

pub mod llm {
    use std::{mem, sync::Arc};

    use async_trait::async_trait;
    use color_eyre::eyre::eyre;
    use ollama_rs::generation::chat::request::ChatMessageRequest;
    use ollama_rs::generation::chat::{ChatMessage, ChatMessageResponseStream};
    use ollama_rs::Ollama;
    use tokio::task::JoinSet;
    use tokio_stream::StreamExt;

    use crate::{Chunk, CompleteChunk, VectorEmbedding};

    pub type StreamResponse = std::pin::Pin<
        Box<dyn tokio_stream::Stream<Item = Result<String, color_eyre::Report>> + Send>,
    >;

    #[async_trait]
    pub trait Llm {
        async fn create_embeddings(&self, text: String) -> color_eyre::Result<VectorEmbedding>;
        async fn create_multiple_embeddings(&self, chunks: Vec<Chunk>) -> Vec<CompleteChunk>;
        async fn answer_query_stream(
            &self,
            query: String,
            documunent_chunks: Vec<Chunk>,
        ) -> color_eyre::Result<StreamResponse>;
    }

    pub struct OllamaLlm {
        ollama: Arc<Ollama>,
        model: String,
    }

    impl OllamaLlm {
        pub fn new(model: &str, ollama: Ollama) -> Self {
            Self {
                ollama: Arc::new(ollama),
                model: model.to_string(),
            }
        }
    }

    #[async_trait]
    impl Llm for OllamaLlm {
        async fn create_embeddings(&self, text: String) -> color_eyre::Result<VectorEmbedding> {
            let response = self
                .ollama
                .generate_embeddings(self.model.clone(), text, None)
                .await?;

            Ok(VectorEmbedding(response.embeddings))
        }

        async fn create_multiple_embeddings(&self, chunks: Vec<Chunk>) -> Vec<CompleteChunk> {
            let mut future_set = JoinSet::new();

            for chunk in chunks {
                let ollama = Arc::clone(&self.ollama);
                let model = self.model.clone();

                let task_future = async move {
                    let result = ollama
                        .generate_embeddings(model, chunk.content.clone(), None)
                        .await;

                    result.map(|em| chunk.complete_with(VectorEmbedding(em.embeddings).into()))
                };

                future_set.spawn(task_future);
            }

            let mut created_embeddings = vec![];

            while let Some(res) = future_set.join_next().await {
                match res {
                    Ok(result) => match result {
                        Ok(value) => {
                            created_embeddings.push(value);
                        }
                        Err(err) => {
                            eprintln!("[ERROR] failed to created embeddings for a prompt: {err}");
                        }
                    },
                    Err(err) => {
                        eprintln!("[ERROR] failed to created embeddings for a prompt: {err}");
                    }
                }
            }

            created_embeddings
        }

        async fn answer_query_stream(
            &self,
            q: String,
            documunent_chunks: Vec<Chunk>,
        ) -> color_eyre::Result<StreamResponse> {
            let mut messages: Vec<_> = documunent_chunks
                .into_iter()
                .map(|doc_chunk| {
                    ChatMessage::new(
                        ollama_rs::generation::chat::MessageRole::System,
                        format!(
                            r#"Consider this excerpt from the document {}, on page {}:
                            {}"#,
                            doc_chunk.document_name, doc_chunk.page_number, doc_chunk.content
                        ),
                    )
                })
                .collect();

            messages.push(ChatMessage::new(
                ollama_rs::generation::chat::MessageRole::System,
                "You are a knowledgeable expert on many subjects. As you consider all excerpts from the documents. Respond to the query from this person.".to_string(),
            ));

            messages.push(ChatMessage::new(
                ollama_rs::generation::chat::MessageRole::User,
                q,
            ));

            let stream: ChatMessageResponseStream = self
                .ollama
                .send_chat_messages_stream(ChatMessageRequest::new(self.model.clone(), messages))
                .await?;

            let stream = Box::new(stream.map(|res| {
                res.map_err(|_| eyre!("failed to get ollama chat response"))
                    .and_then(|ans| {
                        ans.message
                            .ok_or(eyre!("didn't get a message back from ollama"))
                    })
                    .map(|ans| ans.content)
            }));

            Ok(std::pin::Pin::from(stream))
        }
    }

    use llm_chain::options;
    use llm_chain::options::*;
    use llm_chain::traits::Embeddings;

    pub struct LlamaLlmChain {
        // exec: llm_chain_llama::Executor,
        embeddings: llm_chain_llama::embeddings::Embeddings,
    }

    impl LlamaLlmChain {
        pub fn new(model: &str) -> color_eyre::Result<Self> {
            let opts = options!(
                Model: ModelRef::from_path(model), // Notice that we reference the model binary path
                ModelType: "llama",
                NThreads: 4_usize,
                MaxTokens: 2048_usize
            );
            let embeddings = llm_chain_llama::embeddings::Embeddings::new_with_options(opts)?;

            Ok(Self { embeddings })
        }
    }

    #[async_trait]
    impl Llm for LlamaLlmChain {
        async fn create_embeddings(&self, text: String) -> color_eyre::Result<VectorEmbedding> {
            let mut embedded_vecs = self
                .embeddings
                .embed_texts(vec![text.to_string()])
                .await
                .expect("failed to embed text");

            let embedding = mem::take(&mut embedded_vecs[0]);
            Ok(embedding.into())
        }

        async fn create_multiple_embeddings(&self, chunks: Vec<Chunk>) -> Vec<CompleteChunk> {
            let embedded_vecs = self
                .embeddings
                .embed_texts(chunks.iter().map(|c| c.content.clone()).collect())
                .await
                .expect("failed to embed texts");

            embedded_vecs
                .into_iter()
                .zip(chunks.into_iter())
                .map(|(embeddings, chunk)| {
                    chunk.complete_with(VectorEmbedding::from(embeddings).into())
                })
                .collect::<Vec<_>>()
        }

        async fn answer_query_stream(
            &self,
            _q: String,
            _documunent_chunks: Vec<Chunk>,
        ) -> color_eyre::Result<StreamResponse> {
            todo!()
        }
    }
}

pub mod splits {
    pub trait TextSplitter {
        fn split_text(&self, text: String) -> Vec<String>;
    }

    pub struct WordSplitter;

    impl TextSplitter for WordSplitter {
        fn split_text(&self, text: String) -> Vec<String> {
            let words: Vec<_> = text.split(' ').map(|s| s.to_string()).collect();
            words.chunks(125).map(|chunk| chunk.join(" ")).collect()
        }
    }
}

pub mod document_kind {

    use color_eyre::eyre::ContextCompat;
    use poppler::PopplerDocument;

    use crate::primitives;

    pub struct PdfDocument {
        name: String,
        source: primitives::Source,
        document: PopplerDocument,
    }

    impl PdfDocument {
        pub fn new_from_path(path: std::path::PathBuf, document: PopplerDocument) -> Self {
            let name = path
                .file_name()
                .expect("should be a file path, not a directory")
                .to_string_lossy()
                .to_string();

            Self {
                name,
                source: primitives::Source::Path(path),
                document,
            }
        }

        pub fn new_from_url(
            url: reqwest::Url,
            document: PopplerDocument,
        ) -> color_eyre::Result<Self> {
            let name = url
                .path_segments()
                .map(|split| split.collect::<Vec<_>>().join("_"))
                .unwrap_or_default();

            let name = format!("{}_{}", url.domain().unwrap_or_default(), name);

            if name.is_empty() {
                return Err(color_eyre::eyre::anyhow!(
                    "could not derive a name for this pdf document from the url"
                ));
            }

            Ok(Self {
                name,
                source: primitives::Source::Url(url),
                document,
            })
        }
    }

    impl<'s> primitives::Document<'s> for PdfDocument {
        fn name(&'s self) -> &'s str {
            &self.name
        }

        fn get_number_of_pages(&self) -> usize {
            self.document.get_n_pages()
        }

        fn get_source_uri(&self) -> &primitives::Source {
            &self.source
        }

        fn get_page(&self, page_num: usize) -> color_eyre::Result<String> {
            let page = self.document.get_page(page_num).context("no such page");

            let maybe_text = page.and_then(|p| {
                p.get_text()
                    .map(|t| t.to_string())
                    .context("failed to get page text content")
            });

            maybe_text
        }
    }

    pub struct TextDocument {
        name: String,
        source: primitives::Source,
        content: String,
    }

    impl TextDocument {
        pub fn new(path: std::path::PathBuf, content: String) -> Self {
            let name = path
                .file_name()
                .expect("should be a file path, not a directory")
                .to_string_lossy()
                .to_string();

            Self {
                name,
                source: primitives::Source::Path(path),
                content,
            }
        }
    }

    impl<'s> primitives::Document<'s> for TextDocument {
        fn name(&'s self) -> &'s str {
            &self.name
        }

        fn get_number_of_pages(&self) -> usize {
            1
        }

        fn get_source_uri(&self) -> &primitives::Source {
            &self.source
        }

        fn get_page(&self, _page_num: usize) -> color_eyre::Result<String> {
            Ok(self.content.clone())
        }
    }
}

pub mod primitives {

    pub enum Source {
        Path(std::path::PathBuf),
        Url(reqwest::Url),
    }

    impl Source {
        pub fn as_str(&self) -> &str {
            let s = match self {
                Source::Url(url) => url.as_str(),
                Source::Path(path) => path.to_str().expect("file path should be valid utf-8"),
            };

            s
        }
    }

    pub trait Document<'s> {
        fn name(&'s self) -> &'s str;
        fn get_number_of_pages(&self) -> usize;
        fn get_source_uri(&self) -> &Source;
        fn get_page(&self, page_num: usize) -> color_eyre::Result<String>;

        fn get_text(&self) -> color_eyre::Result<Vec<String>> {
            let mut texts = vec![];

            let page_numbers = self.get_number_of_pages();

            for page_num in 0..page_numbers {
                match self.get_page(page_num) {
                    Ok(text) => {
                        texts.push(text);
                    }
                    Err(err) => {
                        eprintln!("[ERROR] failed to get page: {}", page_num);
                        eprintln!("{err:#}");
                    }
                }
            }

            Ok(texts)
        }
    }

    #[derive(Debug)]
    pub struct Chunk {
        pub document_name: String,
        pub page_number: i64,
        pub content: String,
    }

    impl Chunk {
        pub fn complete_with(self, embedding_blob: VectorEmbeddingBlob) -> CompleteChunk {
            CompleteChunk {
                content_embedding: embedding_blob,
                document_name: self.document_name,
                page_number: self.page_number,
                content: self.content,
            }
        }
    }

    pub struct CompleteChunk {
        pub document_name: String,
        pub page_number: i64,
        pub content: String,
        pub content_embedding: VectorEmbeddingBlob,
    }

    impl From<CompleteChunk> for Chunk {
        fn from(value: CompleteChunk) -> Self {
            Self {
                document_name: value.document_name,
                page_number: value.page_number,
                content: value.content,
            }
        }
    }

    impl CompleteChunk {
        pub fn get_embedding(&self) -> VectorEmbedding {
            VectorEmbedding::from(&self.content_embedding)
        }
    }

    pub struct VectorEmbedding(pub Vec<f64>);

    impl From<Vec<f32>> for VectorEmbedding {
        fn from(value: Vec<f32>) -> Self {
            let value = value.into_iter().map(|v| v as f64).collect();
            Self(value)
        }
    }

    impl VectorEmbedding {
        pub fn similarity_with(&self, other_embedding: &VectorEmbedding) -> f64 {
            let n = self.0.len();

            let mut dot_product = 0f64;

            for i in 0..n {
                let a = self.0[i];
                let b = other_embedding.0[i];

                dot_product += a * b
            }

            let mut a_squares = 0f64;
            let mut b_squares = 0f64;

            for i in 0..n {
                let a = self.0[i];
                let b = other_embedding.0[i];

                a_squares += a.powi(2);
                b_squares += b.powi(2);
            }

            dot_product / (a_squares.sqrt() * b_squares.sqrt())
        }
    }

    /// The blob representing vector embeddings that we can save in sqlite.
    #[repr(transparent)]
    pub struct VectorEmbeddingBlob(pub Vec<u8>);

    impl From<Vec<u8>> for VectorEmbeddingBlob {
        fn from(value: Vec<u8>) -> Self {
            Self(value)
        }
    }

    impl From<&VectorEmbeddingBlob> for VectorEmbedding {
        fn from(VectorEmbeddingBlob(data): &VectorEmbeddingBlob) -> Self {
            let floats: Vec<f64> = data
                .chunks(8)
                .map(|bytes| {
                    f64::from_be_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ])
                })
                .collect();

            Self(floats)
        }
    }

    impl From<VectorEmbedding> for VectorEmbeddingBlob {
        fn from(value: VectorEmbedding) -> Self {
            let bytes = value
                .0
                .into_iter()
                .flat_map(|ha| ha.to_be_bytes())
                .collect::<Vec<_>>();

            VectorEmbeddingBlob(bytes)
        }
    }
}
