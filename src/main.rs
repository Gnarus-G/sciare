use clap::{Parser, Subcommand};
use poppler::PopplerDocument;
use reqwest::Url;
use sciare::{save_document, search_documents};
use sqlx::{sqlite::SqliteConnectOptions, SqlitePool};
use std::{path::PathBuf, str::FromStr};

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Upload a file to index.
    Upload {
        /// path to a file (pdf only)
        file: PathBuf,
    },

    /// Download a file from the internet.
    Download {
        /// Url to file
        url: Url,
    },

    Search {
        phrase: String,
    },
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let cli = Cli::parse();

    let options = SqliteConnectOptions::from_str("sqlite:./data.db")?;
    let db_connection = SqlitePool::connect_with(options).await?;

    match cli.command {
        CliCommand::Upload { file } => {
            let name = file
                .file_name()
                .expect("should be a file path, not a directory")
                .to_str()
                .expect("file path should be in valid utf8");

            save_document(
                &db_connection,
                name,
                PopplerDocument::new_from_file(&file, None)?,
            )
            .await?;
        }
        CliCommand::Download { url } => {
            eprintln!("[INFO] downloading file: {url}");
            let response = reqwest::get(url).await?;

            if response.headers()["content-type"] != "application/pdf" {
                return Err(color_eyre::eyre::anyhow!(
                    "only links to pdfs are supported"
                ));
            }

            let name = response
                .url()
                .path_segments()
                .map(|split| split.collect::<Vec<_>>().join("_"))
                .unwrap_or_default();

            let name = format!("{}_{}", response.url().domain().unwrap_or_default(), name);

            if name.is_empty() {
                return Err(color_eyre::eyre::anyhow!(
                    "could not derive a name for this pdf document from the url"
                ));
            }

            eprintln!("[INFO] downloaded pdf as {name}");

            let mut data = response.bytes().await?.to_vec();

            save_document(
                &db_connection,
                &name,
                PopplerDocument::new_from_data(&mut data, None)?,
            )
            .await?;
        }
        CliCommand::Search { phrase } => {
            let chunks = search_documents(&db_connection, phrase).await?;

            for chunk in chunks {
                println!("-----------------");
                println!(
                    "in document: {}; page: {}",
                    chunk.document_name, chunk.page_number
                );
                println!("----");
                println!("{}...", &chunk.content);
                println!();
            }
        }
    };

    return Ok(());
}
