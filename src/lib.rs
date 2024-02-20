use std::path::Path;

use color_eyre::eyre::{Context, ContextCompat};
use ollama_rs::Ollama;
use poppler::PopplerDocument;
use sqlx::SqlitePool;

type Timestamp = chrono::NaiveDateTime;

struct Chunk {
    document_name: String,
    page_number: i64,
    content: String,
}

impl Chunk {
    pub fn complete_with(self, embedding: Vec<f64>) -> CompleteChunk {
        CompleteChunk {
            embedding: format!("{:?}", embedding),
            document_name: self.document_name,
            page_number: self.page_number,
            content: self.content,
        }
    }
}

pub struct CompleteChunk {
    document_name: String,
    page_number: i64,
    content: String,
    embedding: String,
}

impl CompleteChunk {
    fn get_embedding(&self) -> Vec<f64> {
        let e = self.embedding[1..self.embedding.len() - 1]
            .split(',')
            .map(|n| {
                n.parse::<f64>()
                    .expect("should only be number in the array for embeddings")
            })
            .collect();

        return e;
    }
}

pub struct Document {
    name: String,
    saved_at: Timestamp,
}

impl Document {
    fn new(name: String, saved_at: Timestamp) -> Document {
        Document { name, saved_at }
    }
}

pub async fn save_document(conn_pool: &SqlitePool, file_path: &Path) -> color_eyre::Result<()> {
    let name = file_path
        .file_name()
        .expect("should be a file path, not a directory")
        .to_str()
        .expect("file path should be in valid utf8");

    sqlx::query!(r#"INSERT INTO document (name) VALUES (?)"#, name,)
        .execute(conn_pool)
        .await
        .context("faled to save a document")?;

    let content = get_text_from_pdf_file(file_path)?;

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

    let chunks = llm::create_embeddings(ollama, chunks).await;

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
            .bind(chunk.embedding);
    }

    q.execute(conn_pool).await?;

    Ok(())
}

fn get_text_from_pdf_file(path: &Path) -> color_eyre::Result<Vec<String>> {
    let mut texts = vec![];
    let document = PopplerDocument::new_from_file(path, None)?;

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

    use crate::{Chunk, CompleteChunk};

    pub async fn create_embeddings(ollama: Ollama, chunks: Vec<Chunk>) -> Vec<CompleteChunk> {
        let mut set = JoinSet::new();

        let ollama_ref = Arc::new(ollama);

        for chunk in chunks {
            let ollama = Arc::clone(&ollama_ref);

            let task_future = async move {
                let result = ollama
                    .generate_embeddings("llama2:latest".to_string(), chunk.content.clone(), None)
                    .await;

                result.map(|em| chunk.complete_with(em.embeddings))
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
