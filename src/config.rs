use std::path::PathBuf;

use clap::{command, Args, Subcommand};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Serialize, Deserialize)]
struct MyConfig {
    db_path: Option<std::path::PathBuf>,
}

#[derive(Debug, Args)]
pub struct ConfigArgs {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Set configuration values.
    Set {
        /// Set the path of the sqlite database file.
        db_path: PathBuf,
    },
    /// View configuration values.
    Show {
        /// Use the json format.
        #[arg(long)]
        json: bool,
    },

    /// Get the configuration file's path.
    Path,
}

impl ConfigArgs {
    const APP_NAME: &'static str = "sciare";

    pub fn handle(self) -> color_eyre::Result<()> {
        match self.command {
            Command::Set { db_path } => {
                let mut cfg: MyConfig = confy::load(Self::APP_NAME, None)?;
                cfg.db_path = Some(db_path);
                confy::store(Self::APP_NAME, None, cfg)?
            }
            Command::Show { json } => {
                let cfg: MyConfig = confy::load(Self::APP_NAME, None)?;
                let display = if json {
                    serde_json::to_string(&cfg)?
                } else {
                    ron::to_string(&cfg)?
                };

                println!("{display}");
            }
            Command::Path => {
                let p = confy::get_configuration_file_path(Self::APP_NAME, None)?;
                println!("{}", p.display())
            }
        };

        Ok(())
    }
}
