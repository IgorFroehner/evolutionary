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
            gene_cod: GeneCod::IntPerm,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes `contents` to a temporary file, reads it back with `read_config` and removes it.
    fn read(name: &str, contents: &str) -> Option<RawConfig> {
        let path =
            std::env::temp_dir().join(format!("evolutionary-{}-{name}.toml", std::process::id()));
        fs::write(&path, contents).unwrap();

        let config = read_config(path.to_str().unwrap());

        fs::remove_file(&path).unwrap();
        config
    }

    #[test]
    fn reads_the_config_section() {
        let config = read(
            "valid",
            "[config]\n\
             runs = 3\n\
             gene_cod = 'Real'\n\
             range = [-1.0, 2.5]\n\
             population_size = 15\n\
             dimension = 10\n",
        )
        .unwrap();

        assert_eq!(config.runs, 3);
        assert!(matches!(config.gene_cod, GeneCod::Real));
        assert_eq!(config.range, Some(vec![-1.0, 2.5]));
        assert_eq!(config.population_size, 15);
        assert_eq!(config.dimension, 10);
    }

    #[test]
    fn the_range_is_optional() {
        let config = read(
            "no-range",
            "[config]\nruns = 1\ngene_cod = 'Bin'\npopulation_size = 4\ndimension = 2\n",
        )
        .unwrap();

        assert_eq!(config.range, None);
    }

    #[test]
    fn a_missing_field_gives_none() {
        assert!(read("missing-field", "[config]\nruns = 1\ngene_cod = 'Bin'\n").is_none());
    }

    #[test]
    fn an_unknown_gene_cod_gives_none() {
        let contents =
            "[config]\nruns = 1\ngene_cod = 'Tree'\npopulation_size = 4\ndimension = 2\n";

        assert!(read("unknown-gene-cod", contents).is_none());
    }

    #[test]
    fn a_missing_file_gives_none() {
        assert!(read_config("there/is/no/such/config.toml").is_none());
    }

    #[test]
    fn converts_into_a_bin_evolution_config() {
        let raw = RawConfig {
            runs: 1,
            gene_cod: GeneCod::Bin,
            range: None,
            population_size: 15,
            dimension: 10,
        };

        let config: EvolutionConfig<Bin> = raw.into();

        assert_eq!(config.population_size, 15);
        assert_eq!(config.dimension, 10);
        assert!(matches!(config.gene_cod, GeneCod::Bin));
    }

    #[test]
    fn converts_into_an_int_perm_evolution_config() {
        let raw = RawConfig {
            runs: 1,
            gene_cod: GeneCod::IntPerm,
            range: None,
            population_size: 15,
            dimension: 10,
        };

        let config: EvolutionConfig<IntPerm> = raw.into();

        assert_eq!(config.population_size, 15);
        assert_eq!(config.dimension, 10);
        assert!(matches!(config.gene_cod, GeneCod::IntPerm));
    }
}
