# Changelog

## 0.3.0 (unreleased)

### Breaking

- Moved to `rand` 0.9 and `rand_distr` 0.5. The `rand` in use is re-exported as
  `evolutionary::rand`, and its `Rng` trait is in the prelude.
- `Selection::get_mating_pool`, `Crossover::crossover` and `Mutation::mutate` take a new
  `rngs: &Rngs` argument, and custom operators must draw their random numbers from it, asking
  for one generator per unit of work: `rngs.rng(index)`.
- `Individual::generate_member` takes the generator to draw the genes from:
  `generate_member<R: Rng + ?Sized>(dimension, range, rng)`.

### Added

- `EvolutionBuilder::with_seed(u64)`: two runs with the same seed give the same populations and
  the same fitness history, whatever the number of threads. Without a seed every run differs,
  as before.
- `Evolution::seed()`, the seed of the current run, also when it was drawn at random, to
  replay it.
- `Rngs` and `TaskRng`, the generators handed to operators. `Rngs::new(seed)` builds them to
  test an operator on its own.
- A seeded `ExperimentRunner` is reproducible, and each of its runs gets its own seed.

### Changed

- Operators no longer use the thread-local generator. Each individual, pair of parents or
  mating pool slot gets a ChaCha8 generator derived from the seed, the generation, the step and
  its index, so the result doesn't depend on how rayon schedules the work.
- `current_fitness_average` sums in order instead of in parallel, so the average of a seeded
  run is the same to the last bit.

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
