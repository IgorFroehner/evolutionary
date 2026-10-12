use serde_derive::Deserialize;
use std::{
    error::Error,
    fmt, fs, io,
    path::{Path, PathBuf},
};

use crate::{
    evolution::EvolutionConfig,
    population::{Bin, GeneCod, IntPerm},
};

/// The `[config]` section of a config file, as [`read_config`] reads it.
#[derive(Debug, Deserialize, Clone)]
pub struct RawConfig {
    pub runs: u32,
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
        }
    }
}

impl From<RawConfig> for EvolutionConfig<IntPerm> {
    fn from(val: RawConfig) -> Self {
        EvolutionConfig {
            dimension: val.dimension,
            population_size: val.population_size,
            range: (),
        }
    }
}

#[derive(Deserialize, Debug)]
pub struct Data {
    config: RawConfig,
}

/// Why [`read_config`] could not read a config file.
#[derive(Debug)]
#[non_exhaustive]
pub enum ConfigError {
    /// The file could not be read.
    Io { path: PathBuf, source: io::Error },
    /// The file is not valid TOML, or its `[config]` section is missing, misses a field or has a
    /// field of the wrong type.
    Parse {
        path: PathBuf,
        source: Box<dyn Error + Send + Sync>,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    f,
                    "failed to read the config file {}: {source}",
                    path.display()
                )
            }
            Self::Parse { path, source } => {
                write!(
                    f,
                    "failed to parse the config file {}: {source}",
                    path.display()
                )
            }
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source.as_ref()),
        }
    }
}

/// Reads the `[config]` section of the TOML file at `path`.
///
/// # Example
///
/// ```no_run
/// use evolutionary::prelude::*;
///
/// let config = read_config("Config.toml").expect("failed to read the config");
/// println!("{} individuals of {} genes", config.population_size, config.dimension);
/// ```
pub fn read_config(path: impl AsRef<Path>) -> Result<RawConfig, ConfigError> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path).map_err(|source| ConfigError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let data: Data = toml::from_str(&contents).map_err(|source| ConfigError::Parse {
        path: path.to_path_buf(),
        source: Box::new(source),
    })?;

    Ok(data.config)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes `contents` to a temporary file, reads it back with `read_config` and removes it.
    fn read(name: &str, contents: &str) -> Result<RawConfig, ConfigError> {
        let path =
            std::env::temp_dir().join(format!("evolutionary-{}-{name}.toml", std::process::id()));
        fs::write(&path, contents).unwrap();

        let config = read_config(&path);

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
    fn a_missing_field_is_a_parse_error() {
        let error = read("missing-field", "[config]\nruns = 1\ngene_cod = 'Bin'\n").unwrap_err();

        assert!(matches!(error, ConfigError::Parse { .. }), "{error:?}");
        assert!(error.to_string().contains("population_size"), "{error}");
        assert!(error.source().is_some());
    }

    #[test]
    fn an_unknown_gene_cod_is_a_parse_error() {
        let contents =
            "[config]\nruns = 1\ngene_cod = 'Tree'\npopulation_size = 4\ndimension = 2\n";

        let error = read("unknown-gene-cod", contents).unwrap_err();

        assert!(matches!(error, ConfigError::Parse { .. }), "{error:?}");
    }

    #[test]
    fn negative_runs_are_a_parse_error() {
        let contents =
            "[config]\nruns = -1\ngene_cod = 'Bin'\npopulation_size = 4\ndimension = 2\n";

        let error = read("negative-runs", contents).unwrap_err();

        assert!(matches!(error, ConfigError::Parse { .. }), "{error:?}");
    }

    #[test]
    fn a_missing_file_is_an_io_error_naming_the_file() {
        let error = read_config("there/is/no/such/config.toml").unwrap_err();

        match &error {
            ConfigError::Io { path, source } => {
                assert_eq!(path, Path::new("there/is/no/such/config.toml"));
                assert_eq!(source.kind(), io::ErrorKind::NotFound);
            }
            other => panic!("expected an io error, got {other:?}"),
        }
        assert!(
            error.to_string().contains("there/is/no/such/config.toml"),
            "{error}"
        );
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
    }
}
