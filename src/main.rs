use clap::{Parser, Subcommand};
use color_eyre::eyre::ContextCompat;
use poppler::PopplerDocument;
use std::path::{Path, PathBuf};

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
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let cli = Cli::parse();

    match cli.command {
        CliCommand::Upload { file } => {
            let content = get_text_from_pdf_file(&file)?;

            for t in content {
                println!("{t}");
            }
        }
    };

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
