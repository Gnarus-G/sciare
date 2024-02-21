use color_eyre::eyre::Context;
use ollama_rs::Ollama;
use primitives::*;
use splits::TextSplitter;

pub mod context {
    use crate::splits::TextSplitter;
    use sqlx::SqlitePool;

    pub struct Context<Splitter: TextSplitter> {
        pub conn_pool: SqlitePool,
        pub splitter: Splitter,
    }
}

pub async fn search_documents<Splitter: TextSplitter>(
    ctx: &context::Context<Splitter>,
    phrase: String,
    limit: usize,
) -> color_eyre::Result<Vec<Chunk>> {
    let ollama = Ollama::new("http://localhost".to_string(), 11434);

    eprintln!("[INFO] fetching all the chunks of documents we have");
    let chunks = sqlx::query_as!(
        CompleteChunk,
        r#"SELECT document_name, content, page_number, content_embedding FROM chunk"#
    )
    .fetch_all(&ctx.conn_pool)
    .await?;

    eprintln!("[INFO] generating embeddings for the search-phrase");
    let phrase_embedding = ollama
        .generate_embeddings("llama2:latest".to_string(), phrase, None)
        .await
        .map(|result| VectorEmbedding(result.embeddings))?;

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
    sqlx::query!(r#"INSERT INTO document (name) VALUES (?)"#, name)
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

    let ollama = Ollama::new("http://localhost".to_string(), 11434);

    eprintln!("[INFO] creating vector embeddings for each chunk");
    let chunks = llm::create_embeddings(ollama, chunks).await;

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

mod llm {
    use std::sync::Arc;

    use ollama_rs::Ollama;
    use tokio::task::JoinSet;

    use crate::{Chunk, CompleteChunk, VectorEmbedding};

    pub async fn create_embeddings(ollama: Ollama, chunks: Vec<Chunk>) -> Vec<CompleteChunk> {
        let mut set = JoinSet::new();

        let ollama_ref = Arc::new(ollama);

        for chunk in chunks {
            let ollama = Arc::clone(&ollama_ref);

            let task_future = async move {
                let result = ollama
                    .generate_embeddings("llama2:latest".to_string(), chunk.content.clone(), None)
                    .await;

                result.map(|em| chunk.complete_with(VectorEmbedding(em.embeddings).into()))
            };

            set.spawn(task_future);
        }

        let mut created_embeddings = vec![];

        while let Some(res) = set.join_next().await {
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
        document: PopplerDocument,
    }

    impl PdfDocument {
        pub fn new(name: &str, document: PopplerDocument) -> Self {
            Self {
                name: name.to_string(),
                document,
            }
        }
    }

    impl<'s> primitives::Document<'s> for PdfDocument {
        fn name(&'s self) -> &'s str {
            &self.name
        }

        fn get_number_of_pages(&self) -> usize {
            self.document.get_n_pages()
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
        content: String,
    }

    impl TextDocument {
        pub fn new(name: &str, content: String) -> Self {
            Self {
                name: name.to_string(),
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

        fn get_page(&self, _page_num: usize) -> color_eyre::Result<String> {
            Ok(self.content.clone())
        }
    }
}

mod primitives {

    pub trait Document<'s> {
        fn name(&'s self) -> &'s str;
        fn get_number_of_pages(&self) -> usize;
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
