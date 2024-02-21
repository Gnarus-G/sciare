use color_eyre::eyre::{Context, ContextCompat};
use ollama_rs::Ollama;
use poppler::PopplerDocument;
use sqlx::SqlitePool;

#[derive(Debug)]
pub struct Chunk {
    pub document_name: String,
    pub page_number: i64,
    pub content: String,
}

impl Chunk {
    fn complete_with(self, embedding_blob: VectorEmbeddingBlob) -> CompleteChunk {
        CompleteChunk {
            content_embedding: embedding_blob,
            document_name: self.document_name,
            page_number: self.page_number,
            content: self.content,
        }
    }
}

struct VectorEmbedding(Vec<f64>);

impl VectorEmbedding {
    fn similarity_with(&self, other_embedding: &VectorEmbedding) -> f64 {
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
struct VectorEmbeddingBlob(Vec<u8>);

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
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
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

struct CompleteChunk {
    document_name: String,
    page_number: i64,
    content: String,
    content_embedding: VectorEmbeddingBlob,
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
    fn get_embedding(&self) -> VectorEmbedding {
        VectorEmbedding::from(&self.content_embedding)
    }
}

pub async fn search_documents(
    conn_pool: &SqlitePool,
    phrase: String,
) -> color_eyre::Result<Vec<Chunk>> {
    let ollama = Ollama::new("http://localhost".to_string(), 11434);

    let chunks = sqlx::query_as!(
        CompleteChunk,
        r#"select document_name, content, page_number, content_embedding from chunk"#
    )
    .fetch_all(conn_pool)
    .await?;

    let phrase_embedding = ollama
        .generate_embeddings("llama2:latest".to_string(), phrase, None)
        .await
        .map(|result| VectorEmbedding(result.embeddings))?;

    let chunks = chunks
        .into_iter()
        .map(|chunk| {
            let sim = chunk.get_embedding().similarity_with(&phrase_embedding);
            (sim, chunk.into())
        })
        .filter(|&(sim, _)| sim > 0.18)
        .inspect(|(sim, _)| eprintln!("[DEBUG] found content with similarity score {}", sim))
        .map(|(_, chunk)| chunk)
        .collect();

    Ok(chunks)
}

pub async fn save_document(
    conn_pool: &SqlitePool,
    name: &str,
    document: PopplerDocument,
) -> color_eyre::Result<()> {
    eprintln!("[INFO] saving the document");
    sqlx::query!(r#"INSERT INTO document (name) VALUES (?)"#, name,)
        .execute(conn_pool)
        .await
        .context("faled to save a document")?;

    let content = get_text_from_pdf(document)?;

    eprintln!("[INFO] chunking documents");
    let chunks: Vec<_> = content
        .into_iter()
        .enumerate()
        .map(|(idx, c)| Chunk {
            document_name: name.to_string(),
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

    q.execute(conn_pool).await?;

    Ok(())
}

fn get_text_from_pdf(document: PopplerDocument) -> color_eyre::Result<Vec<String>> {
    let mut texts = vec![];

    let page_numbers = document.get_n_pages();

    for page_num in 0..page_numbers {
        let page = document.get_page(page_num).context("no such page");

        let maybe_text = page.and_then(|p| {
            p.get_text()
                .map(|t| t.to_string())
                .context("failed to get page text content")
        });

        match maybe_text {
            Ok(text) => {
                texts.push(text);
            }
            Err(err) => {
                eprintln!("failed to get page: {}", page_num);
                eprintln!("{err:#}");
            }
        }
    }

    Ok(texts)
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
