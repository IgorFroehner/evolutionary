use crate::population::GeneCod;

use std::fs;
use serde_derive::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub runs: i32,
    pub gene_cod: GeneCod,
    pub range: Option<Vec<f64>>,
    pub population_size: u32,
    pub dimension: u32,
}


#[derive(Deserialize, Debug)]
pub struct Data {
    config: Config,
}

pub fn read_config(file_name: &str) -> Option<Config> {
    let data: Option<Data> = match fs::read_to_string(file_name) {
        Ok(contents) => {
            match toml::from_str(&contents) {
                Ok(config) => config,
                Err(error) => {
                    println!("Failed to parse config file: {}", error);
                    None
                },
            }
        },
        Err(error) => {
            println!("Failed to read config file: {} <{}>", error, file_name);
            None
        }
    };

    match data {
        Some(data) => Some(data.config),
        None => None,
    }
}
