use std::path::Path;

use color_eyre::eyre::{Context, ContextCompat};
use poppler::PopplerDocument;
use sqlx::SqlitePool;

type Timestamp = chrono::NaiveDateTime;

pub struct Document {
    name: String,
    saved_at: Timestamp,
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

    let chunks = content;

    let query_string = format!(
        r#"INSERT INTO chunk (documentName, content) VALUES {}"#,
        chunks
            .iter()
            .map(|_| "(?, ?)".to_string())
            .collect::<Vec<String>>()
            .join(",")
    );

    let mut q = sqlx::query(&query_string);

    for c in chunks {
        q = q.bind(name).bind(c);
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
