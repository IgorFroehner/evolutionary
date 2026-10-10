use serde_derive::Deserialize;
use std::fs;

use crate::{
    evolution::EvolutionConfig,
    population::{Bin, GeneCod, IntPerm},
};

#[derive(Debug, Deserialize, Clone)]
pub struct RawConfig {
    pub runs: i32,
    pub gene_cod: GeneCod,
    pub range: Option<Vec<f64>>,
    pub population_size: u32,
    pub dimension: u32,
}

impl From<RawConfig> for EvolutionConfig<Bin> {
    fn from(val: RawConfig) -> Self {
        EvolutionConfig {
            dimension: val.dimension,
            population_size: val.population_size,
            range: (),
            gene_cod: GeneCod::Bin,
        }
    }
}

impl From<RawConfig> for EvolutionConfig<IntPerm> {
    fn from(val: RawConfig) -> Self {
        EvolutionConfig {
            dimension: val.dimension,
            population_size: val.population_size,
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
        Ok(contents) => match toml::from_str(&contents) {
            Ok(config) => config,
            Err(error) => {
                println!("Failed to parse config file: {}", error);
                None
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
