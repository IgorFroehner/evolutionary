use std::fs;
use serde_derive::Deserialize;

use crate::{population::{GeneCod, Bin, IntPerm}, evolution::EvolutionConfig};

#[derive(Debug, Deserialize, Clone)]
pub struct RawConfig {
    pub runs: i32,
    pub gene_cod: GeneCod,
    pub range: Option<Vec<f64>>,
    pub population_size: u32,
    pub dimension: u32,
}

impl Into<EvolutionConfig<Bin>> for RawConfig {
    fn into(self) -> EvolutionConfig<Bin> {
        EvolutionConfig {
            dimension: self.dimension,
            population_size: self.population_size,
            range: (),
            gene_cod: GeneCod::Bin,
        }
    }
}

impl Into<EvolutionConfig<IntPerm>> for RawConfig {
    fn into(self) -> EvolutionConfig<IntPerm> {
        EvolutionConfig {
            dimension: self.dimension,
            population_size: self.population_size,
            range: (),
            gene_cod: GeneCod::Bin,
        }
    }
}

#[derive(Deserialize, Debug)]
pub struct Data {
    config: RawConfig,
}

pub fn read_config(file_name: &str) -> Option<RawConfig> {
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
