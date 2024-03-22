use std::path::PathBuf;

use clap::{command, Args, Subcommand};
use color_eyre::eyre::Context;
use serde::{Deserialize, Serialize};

const APP_NAME: &str = "sciare";

#[derive(Debug, Serialize, Deserialize)]
pub struct MyConfig {
    pub db_path: std::path::PathBuf,
}

impl Default for MyConfig {
    fn default() -> Self {
        let home = get_home_dir().unwrap_or_else(|e| panic!("{e}"));
        Self {
            db_path: home.join("sciare.db"),
        }
    }
}

impl MyConfig {
    pub fn load() -> color_eyre::Result<Self> {
        let cfg: MyConfig = confy::load(APP_NAME, None)?;
        Ok(cfg)
    }
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
    pub fn handle(self) -> color_eyre::Result<()> {
        match self.command {
            Command::Set { db_path } => {
                let mut cfg: MyConfig = confy::load(APP_NAME, None)?;
                cfg.db_path = db_path;
                confy::store(APP_NAME, None, cfg)?
            }
            Command::Show { json } => {
                let cfg: MyConfig = confy::load(APP_NAME, None)?;
                let display = if json {
                    serde_json::to_string(&cfg)?
                } else {
                    ron::to_string(&cfg)?
                };

                println!("{display}");
            }
            Command::Path => {
                let p = confy::get_configuration_file_path(APP_NAME, None)?;
                println!("{}", p.display())
            }
        };

        Ok(())
    }
}

fn get_home_dir() -> color_eyre::Result<PathBuf> {
    #[cfg(unix)]
    let home_dir_key = "HOME";

    let home = std::env::var(home_dir_key).with_context(|| {
        format!(
            "failed to read the user's home directory, using the {} environment variable",
            home_dir_key
        )
    })?;

    Ok(home.into())
}
