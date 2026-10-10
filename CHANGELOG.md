# Changelog

## 0.2.0 (2026-10-10)

### Breaking

- `EvolutionBuilder::build` returns `Result<Evolution<T>, BuildError>` instead of
  `Result<Evolution<T>, String>`. `BuildError` names the missing part or the inconsistent
  parameter, and implements `std::error::Error`.
- `build` rejects a population size of 0 (`BuildError::EmptyPopulation`).
- `build` checks elitism against the population size instead of the chromosome
  dimension, so some configurations that were accepted are now rejected and others that
  were rejected are now accepted.

### Fixed

- The population was twice the configured size: `Evolution::start` generated
  `population_size` individuals and then `population_size` more.
- `build` panicked when no stop condition was set. It now returns
  `BuildError::MissingStopCondition`.
- A `NaN` fitness panicked comparisons. `NaN` now ranks below every number: it never wins
  a tournament, never becomes an elitist or the best, and has no weight in roulette and
  SUS selection. `current_fitness_average` ignores it.
- `TournamentSelection` indexed out of bounds with a tournament of one.
- `GaussianMutation` and `ScrambleMutation` were not exported.

### Added

- `TournamentSelection::new(k, kp)`, with `k()` and `kp()` getters.
- `Real::new` and `Int::new`, to build an individual from a known chromosome.

### Changed

- `TournamentSelection` clones only the winner of each tournament.
- Elitism uses a partial sort instead of heaps, and the `ordered-float` dependency was
  removed.
