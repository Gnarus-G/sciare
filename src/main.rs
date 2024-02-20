use clap::{Parser, Subcommand};
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

    Search {
        phrase: String,
    },
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let cli = Cli::parse();

    match cli.command {
        CliCommand::Upload { file } => {
            let options = SqliteConnectOptions::from_str("sqlite:./data.db")?;
            let db_connection = SqlitePool::connect_with(options).await?;

            save_document(&db_connection, &file).await?;
        }
        CliCommand::Search { phrase } => {
            let options = SqliteConnectOptions::from_str("sqlite:./data.db")?;
            let db_connection = SqlitePool::connect_with(options).await?;

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
