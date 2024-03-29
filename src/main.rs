mod config;

use clap::{crate_name, Args, Parser, Subcommand};
use ollama_rs::Ollama;
use poppler::PopplerDocument;
use reqwest::Url;
use sciare::{
    context, document_kind::PdfDocument, llm, primitives::Document, save_document,
    search_documents, splits,
};
use sqlx::{sqlite::SqliteConnectOptions, SqlitePool};
use std::{
    net::{IpAddr, Ipv4Addr},
    path::PathBuf,
    str::FromStr,
};
use tokio_stream::StreamExt;

/// CLI tool to manage documents, doing semantic searches through documents, and prompting usign the RAG (Retrieval Augmented Generation)
/// strategy to use relevant info from documents to extend the llm's strategy.
#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: CliCommand,

    /// Ip address serving Ollama api, assuming port 11434
    #[arg(global = true, long)]
    ollama_ip: Option<IpAddr>,
}

#[derive(Args)]
#[group(id = "inhousellm")]
struct UseInHouseLlama {
    /// Don't use ollama, use llama more directly [Experimental/Not Recommended]
    #[arg(
        global = true,
        long = "no-ollama",
        default_value = "false",
        group = "inhousellm",
        requires = "model"
    )]
    selected: bool,

    /// The llama model to use (a .gguf file).
    #[arg(global = true, short, long)]
    model: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// List all the documents in the database.
    List {
        /// Show more info about each document
        #[arg(short, long)]
        verbose: bool,
    },

    /// Delete a document from the database.
    Delete {
        /// Name of the document to delete
        name: String,
    },

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

    /// Search across all the content saved.
    Search {
        /// Search-phrase by which to match semantically.
        phrase: String,

        #[arg(short, long, default_value = "5")]
        /// Maximum number of document chunks to consider as matches.
        limit: usize,
    },

    /// Ask a question, and get an answer that considers information across all the content saved.
    Ask {
        /// Question/Query to ask the language model.
        q: String,

        #[arg(short, long, default_value = "20")]
        /// Maximum number of document chunks to consider as matches.
        limit: usize,
    },
    /// Manage configuration values.
    Config(config::ConfigArgs),

    /// Generate a completions file for a specified shell
    Completion {
        // The shell for which to generate completions
        shell: clap_complete::Shell,
    },
}

impl Cli {
    fn choose_llm(&self) -> Box<dyn llm::Llm> {
        // if self.inhousellm.selected {
        //            Box::new(
        //                llm::LlamaLlmChain::new(
        //                    self.inhousellm
        //                        .model
        //                        .clone()
        //                        .expect("no llama model provided: clap should have caught this")
        //                        .to_str()
        //                        .expect("path of the model given should be in valid utf-8"),
        //                )
        //                .expect("failed to instantiate llama"),
        //            )
        //        } else
        {
            let ip = self
                .ollama_ip
                .unwrap_or(Ipv4Addr::new(127, 0, 0, 1).into())
                .to_string();

            Box::new(llm::OllamaLlm::new(
                "llama2-uncensored:latest",
                Ollama::new(format!("http://{ip}"), 11434),
            ))
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let cli = Cli::parse();

    let cfg = config::MyConfig::load()?;
    let db_url = format!("sqlite:{}", cfg.db_path.display());

    let options = SqliteConnectOptions::from_str(&db_url)?.create_if_missing(true);
    let db_connection = SqlitePool::connect_with(options).await?;

    sqlx::migrate!().run(&db_connection).await?;

    let context = context::Context {
        conn_pool: db_connection,
        splitter: splits::WordSplitter,
        llm: cli.choose_llm(),
    };

    match cli.command {
        CliCommand::Upload { file } => {
            let document = PopplerDocument::new_from_file(&file, None)?;
            let document = PdfDocument::new_from_path(file, document);

            save_document(&context, &document).await?;

            eprintln!("[INFO] uploaded file as {}", document.name());
        }
        CliCommand::Download { url } => {
            eprintln!("[INFO] downloading file: {url}");
            let response = reqwest::get(url).await?;

            if response.headers()["content-type"] != "application/pdf" {
                return Err(color_eyre::eyre::anyhow!(
                    "only links to pdfs are supported"
                ));
            }
            let url = response.url();

            let url = url.clone();
            let mut data = response.bytes().await?.to_vec();
            let document =
                PdfDocument::new_from_url(url, PopplerDocument::new_from_data(&mut data, None)?)?;

            save_document(&context, &document).await?;

            eprintln!("[INFO] downloaded pdf as {}", document.name());
        }
        CliCommand::Search { phrase, limit } => {
            let chunks = search_documents(&context, phrase, limit).await?;

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
        CliCommand::Ask { q, limit } => {
            let chunks = search_documents(&context, q.clone(), limit).await?;
            let mut stream = context.llm.answer_query_stream(q, chunks).await?;

            while let Some(Ok(ans_token)) = stream.next().await {
                print!("{ans_token}");
            }
            println!();
        }
        CliCommand::List { verbose } => {
            let docs = sqlx::query!(r#"select * from document"#)
                .fetch_all(&context.conn_pool)
                .await?;

            if docs.is_empty() {
                eprintln!("[INFO] there are no documents in the database");
            } else {
                eprintln!("[INFO] fetched a list of documents in the database");
            }

            for doc in docs {
                println!("{}", doc.name);
                if verbose {
                    println!("  ↳ added on: {}", doc.saved_at);
                    println!("  ↳ source:   {}", doc.source_uri);
                }
            }
        }
        CliCommand::Delete { name } => {
            eprintln!(
                "[WARN] about to delete document '{}' and all its saved excerpts",
                name
            );
            sqlx::query!(r#"delete from chunk where document_name = ?"#, name)
                .execute(&context.conn_pool)
                .await?;

            sqlx::query!(r#"delete from document where name = ?"#, name)
                .execute(&context.conn_pool)
                .await?;

            eprintln!("[INFO] Successfully Deleted {name}");
        }
        CliCommand::Config(c) => c.handle()?,
        CliCommand::Completion { shell } => clap_complete::generate(
            shell,
            &mut <Cli as clap::CommandFactory>::command(),
            crate_name!(),
            &mut std::io::stdout(),
        ),
    };

    return Ok(());
}
