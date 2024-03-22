mod config;

use clap::{Args, Parser, Subcommand};
use ollama_rs::Ollama;
use poppler::PopplerDocument;
use reqwest::Url;
use sciare::{context, document_kind::PdfDocument, llm, save_document, search_documents, splits};
use sqlx::{sqlite::SqliteConnectOptions, SqlitePool};
use std::{
    net::{IpAddr, Ipv4Addr},
    path::PathBuf,
    str::FromStr,
};
use tokio_stream::StreamExt;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: CliCommand,

    #[command(flatten)]
    #[group(id = "inhousellm")]
    inhousellm: UseInHouseLlama,

    /// Ip address serving Ollama api, assuming port 11434
    #[arg(global = true, long, conflicts_with = "inhousellm")]
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

        #[arg(short, long, default_value = "20")]
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
    Config(config::ConfigArgs),

    /// Generate a completions file for a specified shell
    Completion {
        // The shell for which to generate completions
        shell: clap_complete::Shell,
    },
}

impl Cli {
    fn choose_llm(&self) -> Box<dyn llm::Llm> {
        if self.inhousellm.selected {
            Box::new(
                llm::LlamaLlmChain::new(
                    self.inhousellm
                        .model
                        .clone()
                        .expect("no llama model provided: clap should have caught this")
                        .to_str()
                        .expect("path of the model given should be in valid utf-8"),
                )
                .expect("failed to instantiate llama"),
            )
        } else {
            let ip = self
                .ollama_ip
                .unwrap_or(Ipv4Addr::new(127, 0, 0, 1).into())
                .to_string();

            Box::new(llm::OllamaLlm::new(
                "llama2:latest",
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
            let name = file
                .file_name()
                .expect("should be a file path, not a directory")
                .to_str()
                .expect("file path should be in valid utf8");

            save_document(
                &context,
                &PdfDocument::new(name, PopplerDocument::new_from_file(&file, None)?),
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
                &context,
                &PdfDocument::new(&name, PopplerDocument::new_from_data(&mut data, None)?),
            )
            .await?;
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
        CliCommand::Config(c) => c.handle()?,
        CliCommand::Completion { shell } => clap_complete::generate(
            shell,
            &mut <Cli as clap::CommandFactory>::command(),
            "sciare",
            &mut std::io::stdout(),
        ),
    };

    return Ok(());
}
