# Changelog

## Unreleased

### Breaking

- `SimulatedBinaryCrossover` computes its spread factor `beta` with the `libm` crate's `pow`
  instead of `f64::powf`, which calls the platform's math library. IEEE 754 does not require
  `pow` to be correctly rounded, and platforms disagree in the last bit on some draws (Apple's
  libm rounds a few of them differently from glibc), so a seeded run with SBX could replay on
  one platform and not on another. `libm` is the same Rust code on every target, so SBX now
  gives the same children everywhere. Its `pow` is accurate to under one unit in the last
  place but not always correctly rounded, so it differs from the platform's result on a few
  percent of draws, by one unit in the last place: seeded runs that use SBX replay differently
  from 0.5.0, in general on every platform. Nothing to rewrite, and the distribution of the
  children is unchanged. `libm` (pure Rust, no dependencies) becomes a direct dependency; it
  was already built, through `rand_distr`. The golden run with SBX
  (`real_roulette_simulated_binary_gaussian`) is recorded again; every other golden run is
  unchanged.

### Fixed

- The README said a seeded run replays on every platform. It now says which built-in operator
  can't promise that: `GaussianMutation`, whose `rand_distr::Normal` calls the platform's `ln`
  and `exp`, can diverge across platforms on rare draws. Every other built-in operator gives
  the same result everywhere.
- `within_range` computes `2^l` with `libm::pow` instead of `f64::powf`, so it is the same on
  every platform for any `l`. For a whole number of bits both give exactly `2^l`, so its
  results are unchanged.

### Added

- `.claude/skills/bench-report/` says how to produce a PR's before-and-after benchmark report,
  and its `compare.py` renders the table from criterion's output.

### Changed

- The benchmark suite is split by concern, `generation`, `selection`, `crossover`, `mutation`
  and `checkpoint` (the last needs the `serde` feature). It covers a generation of every
  representation at 50x50, 300x300 and 2000x500, and at 300x300 and 2000x500 every built-in
  crossover and mutation on each representation it works for, every selection on `Real`, and
  the checkpoint round trip. Each benchmark is named `<group>/<case>/<population>x<genes>`,
  for comparing a branch with a saved baseline of main; `benches/README.md` says how.
  `benches/real_at_scale.rs` and its names are gone. `cargo bench` no longer runs the lib's
  empty libtest harness, which rejected criterion's flags such as `--save-baseline`. Nothing
  in the library changes.

## 0.5.0 (2026-10-13)

### Breaking

- `ExperimentRunner::run` returns `Result<(), BuildError>`: write `experiment.run()?` (or
  `experiment.run().unwrap()`) instead of `experiment.run()`. The builder is checked once before
  any run starts, even with zero runs, so an invalid builder returns its error instead of
  panicking inside a rayon thread.
- With the `plot` feature, `ExperimentRunner::run` no longer saves a chart to
  `results/<name>/<name>.png`: call `experiment.save_chart("results/<name>/<name>.png")` after
  `run` to keep saving it. Nothing is written unless you ask.
- `ExperimentResult::best_found` is `Evolution::best_ever`, the first individual that reached
  the run's best fitness, instead of the best of the last generation, so seeded runs can give a
  different `best_found`. With or without elitism, on a tie it can be a different chromosome with
  the same fitness; without elitism it can be better, when the last generation had lost the
  best. For the best of the final population, replay the run with
  `EvolutionBuilder::with_seed(result.seed)` and read `current_best()` instead of
  `best_found`. Each run's seed and fitness history are unchanged.
- `EvolutionBuilder::new` no longer takes the encoding, which the individual type already
  fixes: write `EvolutionBuilder::new(30, 10, ())` instead of
  `EvolutionBuilder::new(30, 10, GeneCod::Bin, ())`. `EvolutionConfig` loses its `gene_cod`
  field, which nothing read. `GeneCod` stays, for `RawConfig::gene_cod` with the `config`
  feature.
- `Individual::get_chromosome` returns `&[Gene]` and `get_mut_chromosome` returns
  `&mut [Gene]`. In an `Individual` implementation, write
  `fn get_chromosome(&self) -> &[Self::Gene]` instead of `-> &Vec<Self::Gene>`, and the same
  for `get_mut_chromosome`; the bodies (`&self.chromosome`) stay. To copy the genes, write
  `individual.get_chromosome().to_vec()` instead of `individual.get_chromosome().clone()`.
  Comparing a chromosome with a `Vec` still works. The built-in individuals keep their
  `Vec` fields. Code that resized a chromosome through `get_mut_chromosome` (a `push`, a
  `remove`) builds a new individual instead for `Bin` and `Int`, whose fields are private:
  `Bin::new(genes)` or `Int::new(genes, individual.range)`. For `IntPerm` and `Real` it can
  resize the public `chromosome` field: `individual.chromosome.push(gene)`.
- An individual that was never evaluated has a fitness of `NaN` instead of `0.0`:
  `generate_member` and `Bin::new`, `Int::new`, `IntPerm::new` and `Real::new` start there,
  so a fitness that was never computed can't pass for a real zero. Write
  `individual.get_fitness().is_nan()` instead of `individual.get_fitness() == 0.0` to tell
  one apart. Every comparison already ranks `NaN` last, and an evolution evaluates every
  individual before reading its fitness, so seeded runs are unchanged. With the `serde`
  feature such an individual is saved to JSON with a `null` fitness, which now loads back as
  `NaN`; files saved before load as they did. `serde_json` also writes an infinite fitness
  as `null`, so one saved to JSON now loads back as `NaN`, silently, where the file failed to
  load before. Binary formats keep infinities.
- `Evolution::new` is private: write `EvolutionBuilder::new(..)` with its `with_*` calls and
  `.build()?` instead of `Evolution::new(title, config, fitness, ..)`. The builder was already
  the supported way, and checks the parameters `Evolution::new` took on trust.
- With the `config` feature, `read_config` returns `Result<RawConfig, ConfigError>` instead
  of printing what went wrong to stdout and returning `None`: write
  `read_config(path).expect("..")` (unchanged) or `read_config(path)?`, and
  `if let Ok(config) = read_config(path)` instead of `if let Some(config) = read_config(path)`.
  `ConfigError`, in the prelude, tells a file that could not be read (`Io`) from one that
  could not be parsed (`Parse`), names the file and implements `std::error::Error`.
  `read_config` takes any path (`impl AsRef<Path>`), so `&str` still works.
- `RawConfig::runs` is a `u32` instead of an `i32`, so a negative `runs` is a parse error:
  write `report.generation >= config.runs` instead of `report.generation >= config.runs as u32`.
- `Metrics::total_time` returns a `Duration` instead of nanoseconds as a `u128`: write
  `metrics.total_time().as_nanos()` instead of `metrics.total_time()` where you need the
  number. `Metrics::step_times` is private: write `metrics.step_time(Steps::Selection)` (an
  `Option<Duration>`) instead of reading `metrics.step_times[&Steps::Selection].2`.
  `time_digest` prints the same thing, and its documentation now lists the elitism line it
  already printed.
- With the `plot` feature, `Evolution::plot_chart` takes only the path, as any
  `impl AsRef<Path>` like `ExperimentRunner::save_chart` and `read_config`, and captions the
  chart with the title given to `EvolutionBuilder::with_title`, which nothing read before:
  write `.with_title("TSP")` on the builder and `evolution.plot_chart("tsp.png")` instead of
  `evolution.plot_chart("tsp.png", "TSP")`. For another caption, write
  `evolution.metrics.plot_chart("tsp.png", "Another caption")`. A path that is not valid
  UTF-8 is an error.
- `within_range` no longer floors its result, which threw away the precision of the bits it
  decodes: `examples/math_func` decoded 16 bits into the five integers of `(-2, 2)` and now
  searches the whole interval, in steps of `4 / 65535`. Write
  `within_range(range, l, d).floor()` instead of `within_range(range, l, d)` to keep the
  integer. It also interpolates between the two ends instead of adding a rounded width to
  the lower one, so `0` and `2^l - 1` give exactly `range.0` and `range.1`, and no result
  leaves the range: adding the width gave `7.099999999999999` for the top of `(-3.3, 7.1)`
  and `0.20000000000000004`, above the range, for the top of `(-0.1, 0.2)`.
- `GaussianMutation::sigma` is a `Sigma`, a constant or a schedule over the generations,
  instead of an `f64`: in a struct literal, write `sigma: 0.1.into()` instead of `sigma: 0.1`,
  and to change it, `mutation.sigma = 0.2.into()` instead of `mutation.sigma = 0.2`.
  `GaussianMutation::new(0.05, 0.1)` takes any `impl Into<Sigma>` and compiles unchanged. To
  read the value, write `mutation.sigma.at(generation)` instead of `mutation.sigma`. A
  constant `sigma` draws exactly the same numbers as before, so seeded runs are unchanged.
- `StochasticUniversalSamplingSelection`, and `scaled` with it, no longer picks an individual
  with a weight of zero (a fitness of zero or `NaN`) in the last slot. When its random start
  point was just below the pointer spacing, rounding could put the last pointer past the total
  weight, and the walk ran off the end of the wheel and picked the last individual whatever
  its weight. It now picks the last individual with a positive weight, the slice the pointer
  belongs to. A seeded run can differ only in a generation where the start point rounded the
  last pointer past the total, which only a few of the 2^52 possible start points do, more in
  larger populations (still under one generation in 10^12 at 10 000 individuals), and then
  only in the last slot of the mating pool, when the population ends in zero weights. The
  number and order of random draws are unchanged, and every other run replays exactly as
  before; the golden runs are unchanged. Nothing to rewrite.

### Added

- Checkpoints: `evolution.checkpoint()` saves where a run is, and
  `EvolutionBuilder::with_checkpoint(checkpoint)` resumes it exactly, as if it had never
  stopped: the next generation is the one after the checkpoint's, with the same random numbers,
  no fitness is computed again, and the best ever, the fitness history and
  `generations_without_improvement` go on from the checkpoint's. Resuming a run from 15
  generations gives the same population, history and best ever after 30 as running 30 in one
  go, on 1 thread or 8. `Checkpoint<T>` (in the prelude) is a `#[non_exhaustive]` struct of
  public fields: the run's `seed`, its `generation`, the `population` with its fitnesses,
  `best_ever`, `best_fitnesses`, `avg_fitnesses` and `generations_without_improvement`.
  Timings are not part of it. It has no public constructor, so a field can be added without
  breaking anything: a checkpoint comes from `Evolution::checkpoint`, or from deserializing
  one. With the `serde` feature it is `Serialize` and `Deserialize`, with a `null` fitness in
  the history read back as `NaN`. To resume exactly from JSON, add
  `serde_json = { version = "1", features = ["float_roundtrip"] }` to your own dependencies,
  without which the last bit of some numbers changes on the way back. JSON also turns an
  infinite fitness into `NaN`, and the resume is then not exact: save to a binary format such
  as CBOR if your fitness can be infinite.
- A resumed run keeps the checkpoint's seed. `build` returns the new `BuildError` variants
  `CheckpointSeed` when `with_seed` gives another one (the same one is fine),
  `CheckpointWithInitialPopulation` when `with_initial_population` gives individuals too,
  `CheckpointPopulationSize`, `CheckpointIndividualDimension` and `CheckpointHistory` when the
  checkpoint doesn't fit the builder. An `ExperimentRunner` whose builder has a checkpoint
  resumes every run from it with a seed of its own, drawn from the checkpoint's, so the runs
  branch off it. Seeded runs without a checkpoint are unchanged.
- `Fitness::calculate_population(&self, population: &mut [T])` evaluates a whole generation in
  one call, which the evolution now makes once per generation. The default does what the
  evolution did before, `calculate_fitness` on every individual in parallel, so existing
  fitnesses and seeded runs are unchanged (`tests/golden.rs` passes as it was). Override it to
  evaluate the population as a batch: a simulation that steps every individual together, a
  batched forward pass, a GPU kernel.
- `EvolutionBuilder::with_reevaluate_elitists(true)` evaluates the elitists again every
  generation, with the rest of the population and in the same `calculate_population` call,
  instead of letting them carry their stored fitness. With a noisy fitness, one lucky
  evaluation no longer stays at the top for as long as elitism keeps it. It costs `elitism`
  extra evaluations per generation. Off by default, which is the behavior of 0.4.0; it draws
  no random number, so with a deterministic fitness a seeded run is the same either way.
- The `Fitness` documentation says what a seeded run needs from the fitness: the same number
  for the same individual every time. The fitness gets no `Rngs`; a noisy fitness that must
  replay derives its noise from a fixed hash of the chromosome, and counts the generations in
  `calculate_population` to make the noise change from one generation to the next. That
  count lives in the fitness value, copied by `Clone`, never behind an `Arc` or in a `static`,
  and a run resumed from a checkpoint starts it at `checkpoint.generation + 1`. It has a
  runnable example. `calculate_population` gets the population in a fixed order: the new
  individuals in mating-pool order, then the re-evaluated elitists.
- Scheduled `sigma` for `GaussianMutation`: `Sigma::schedule(|generation| ..)` sets it from
  the generation, and `Sigma::exponential_decay(start, factor)` multiplies it by `factor` at
  every generation, to explore first and refine later without stopping the run. A schedule
  changes only the spread of the numbers drawn, not which ones are drawn. The decay computes
  its powers with multiplications only, so it gives the same values on every platform.
- `Rngs::generation()`, the generation an operator is called for (1 at the first
  `Evolution::next`, as in its `GenerationReport`), so a custom operator can schedule its own
  parameters. It does not change the generators.
- `Minimize(fitness)` turns a value to minimize into a fitness by negating it, so the
  evolution, the stop condition and the reports keep maximizing, and `-report.best_fitness` is
  exactly the value. It hands the population to the wrapped fitness's `calculate_population`,
  so a batched fitness stays batched. `examples/xor` uses it instead of negating its error by
  hand, and replays bit for bit.
- Fitness scaling for the fitness-proportional selections:
  `RouletteSelection::scaled(scaling)` and `StochasticUniversalSamplingSelection::scaled(scaling)`
  draw with weights given by a `FitnessScaling` instead of the raw fitness, so negative
  fitnesses (from `Minimize`, say) work. `Window { epsilon }` weighs by `fitness - worst +
  epsilon`, `Rank` by rank with ties sharing their average rank, and `Sigma` by
  `1 + (fitness - mean) / (2 * standard deviation)`, floored at `0.1`, with the mean and
  standard deviation over the finite fitnesses. A `NaN` fitness still gets no chance. Only
  `Rank` accepts infinite fitnesses: `Sigma` gives them no chance, and `Window` panics on them.
  Unscaled `RouletteSelection` and `StochasticUniversalSamplingSelection` draw the same numbers
  as before and panic in the same cases. The `ScaledSelection` docs say when each scaling
  panics.
- Golden replay runs pin a rank-scaled SUS with an exponentially decaying `sigma`, and the
  thread tests cover windowed roulette and sigma-scaled SUS. Their constants are recorded
  from 0.5.0 code, since these operators are new in it.
- `RouletteSelection` and `StochasticUniversalSamplingSelection` implement `Debug`.
- `evolutionary::rayon` re-exports the `rayon` the crate runs on, as `evolutionary::rand`
  does for `rand`: write `use evolutionary::rayon::prelude::*;` in a parallel custom operator
  instead of adding `rayon` to your own dependencies at a matching version. It stays out of
  the prelude, so its traits only come into scope where they are asked for.
- `Evolution::population()` borrows the current population as a slice, in its current order,
  where `current_population()` clones and sorts it at every call.
- `EvolutionBuilder::with_title` takes any `impl Into<String>`, so `.with_title("TSP")` works
  without `.to_string()`, and `Evolution::title()` returns it.
- `ExperimentRunner::save_chart(path)`, with the `plot` feature, saves the chart of the best and
  average fitness per generation averaged over the runs, creating the parent directories.
- `ExperimentRunner::average_time`, `average_best_fitness`, `average_generations` and
  `total_time` expose the numbers `experiment_digest` prints. The digest now prints the average
  best fitness too, and says "generations" instead of "max iterations".
- `ExperimentResult::time` and `ExperimentResult::seed`: how long a run took and its seed, which
  `EvolutionBuilder::with_seed` takes to replay that run on its own.
- Golden replay tests (`tests/golden.rs`): one short seeded run per encoding, which between
  them use every built-in selection, must give exactly the final population and fitness
  history of 0.4.0. A seeded run that stops replaying now fails the tests instead of going
  unnoticed until someone fingerprints runs by hand.
- `rust-version = "1.82"` in `Cargo.toml`, the oldest Rust that builds the library with every
  feature (`Option::is_none_or` needs 1.82). With the `config` feature on Rust older than 1.85,
  run `cargo update -p indexmap --precise 2.13.1`: newer `indexmap` releases, which `toml` pulls
  in, need 1.85. The tests and examples need 1.85 too, through their dev-dependencies.

### Fixed

- Calling `ExperimentRunner::run` again on a seeded experiment repeated the seeds of the first
  call, so the new runs duplicated the old ones and the averages counted them twice. The seed of
  a run is now drawn from its position in `experiment_results`: the first call draws exactly the
  same seeds as before, and later calls draw new ones (the same as one call with more runs).
- Calling `ExperimentRunner::run` twice counted the first batch's time twice, and
  `experiment_digest` divided the total by the runs of one call instead of the number of
  results. The times are now added up from the results when asked for. The chart also averages
  over every result instead of the runs of one call.
- `experiment_digest` no longer panics when there are no results yet, and saving the chart then
  returns an error instead of panicking.
- `examples/subway` panicked at start: it read `examples/subway/subway.in`, which does not exist,
  instead of the `metro.in` network it ships with.
- `StochasticUniversalSamplingSelection`, and `scaled` with it, panicked on an index underflow
  when its random start point was exactly `0.0` (about one generation in 2^52): the first
  pointer moved past no individual. A pointer at `0.0` now picks the first individual with a
  positive weight, never one with a weight of zero. Every start point above `0.0` picks what
  it did, so seeded runs are unchanged.

### Changed

- `convert_bin` adds `2f64.powi(i)` per set bit instead of `2f64.powf(i)`, with the same
  results, and documents that it is exact up to 53 bits.
- Every operator that can panic says when in a `# Panics` section: a rate outside `[0, 1]`,
  an empty population in `RouletteSelection` and `RankSelection`, a negative fitness in
  `RouletteSelection`, an empty or non-finite range in the resetting and delta mutations, and
  so on. So do `Evolution::next`, `current_best_fitness` and `population_digest` before the
  evolution starts, and `Metrics::step_start` and `step_end`. `CycleCrossover`,
  `PartiallyMappedCrossover`, `SwapMutation` and `DeltaMutation` have a description at last.
- `GaussianMutation` panics on a non-finite `sigma` with a message that gives the value and
  the generation, instead of unwrapping a `rand_distr` error. `RouletteSelection` panics on
  an infinite total fitness with its documented message instead of `rand`'s `NonFinite` error.
- CI checks formatting, fails on rustdoc warnings, checks the library on Rust 1.82, and
  uses `actions/checkout` v4. Dependabot proposes cargo and GitHub Actions updates monthly.
- `Cargo.lock` is updated to the newest dependency versions that build on Rust 1.82, and
  `.cargo/config.toml` makes `cargo update` in this repository keep to them.

## 0.4.0 (2026-10-11)

### Breaking

- `RandomResettingMutation` on `Int` now draws reset genes from the whole range, upper bound
  included, as `Int::generate_member` does. Before, a gene could never be reset to
  `range.1`. Seeded runs that use it on `Int` give different results; `Real` is unchanged.

### Fixed

- Converting a `RawConfig` into an `EvolutionConfig<IntPerm>` set `gene_cod` to `Bin`
  instead of `IntPerm`.
- `StochasticUniversalSamplingSelection` panics with a clear message when no fitness is
  positive or their total is not finite, instead of an "empty range" panic from `rand`, the
  same way `RouletteSelection` does.
- With the `plot` feature, rendering a chart on Linux aborted debug builds with an "unsafe
  precondition(s) violated" panic from `font-kit` 0.11 (a null pointer for an empty glyph
  bitmap, caught by Rust's debug checks). `plotters` 0.3.7 is now the minimum, which pulls
  `font-kit` 0.14 where this is fixed.

### Added

- Unit tests for every mutation, selection and crossover operator that only had
  reproducibility coverage, for `generate_member` of each encoding, for `read_config` and for
  `plot_chart`, and an integration test for `ExperimentRunner`.

### Performance

- `StochasticUniversalSamplingSelection` no longer goes through rayon, and seeded runs are
  unchanged. Its only parallel part was computing the pointer positions, one multiplication
  per slot, which cost more than the whole sequential walk over the wheel: at 300 individuals
  of 300 genes it goes from about 70 µs to about 20 µs. A fully parallel version, one binary
  search over the cumulative weights per slot, gives the same pool but measured slower up to
  10 000 individuals: the work per slot is a clone, and rayon's fork and join costs more than
  the walk.
- `benches/real_at_scale.rs` benchmarks SUS too.

## 0.3.1 (2026-10-10)

### Added

- `SimulatedBinaryCrossover` and `LinearCrossover` are implemented. Both were exported but
  panicked with `todo!()` when called. SBX follows Deb & Agrawal, with a configurable
  distribution index `eta`; the linear crossover builds Wright's three candidates and replaces
  the parents with two of them drawn at random.

### Performance

Seeded runs are unchanged: every optimization preserves the exact random streams and the
floating-point evaluation order.

- The best individual and the average fitness are computed once per generation and cached.
  The report, the stop condition and the `on_generation` callback no longer rescan the
  population (previously 4-6 full scans per generation).
- `UniformCrossover` and `NPointsCrossover` swap genes in place instead of cloning both
  parents, and `ArithmeticCrossover` computes both offspring in place instead of cloning the
  parents and allocating new chromosomes.
- `RouletteSelection` builds its cumulative weight table once per generation instead of once
  per drawn slot (`O(n^2)` to `O(n log n)`).
- `RankSelection` sorts indices instead of cloning the whole population to sort it.
- The per-gene mutations build their Bernoulli distribution once per `mutate` call instead of
  once per gene.

### Changed

- The three per-encoding copies of `UniformCrossover` and `NPointsCrossover` are replaced by
  one generic implementation each; behavior is identical.
- `Metrics::plot_chart` and `plotting::plot_chart` take `&str` instead of `&String` (callers
  passing `&String` still compile).

## 0.3.0 (2026-10-10)

### Breaking

- Moved to `rand` 0.9 and `rand_distr` 0.5. The `rand` in use is re-exported as
  `evolutionary::rand`, and its `Rng` trait is in the prelude.
- `Selection::get_mating_pool`, `Crossover::crossover` and `Mutation::mutate` take a new
  `rngs: &Rngs` argument, and custom operators must draw their random numbers from it, asking
  for one generator per unit of work: `rngs.rng(index)`.
- The operator traits take slices: `get_mating_pool(&self, population: &[T], rngs)`,
  `crossover(&self, population: &mut [T], rngs)` and `mutate(&self, population: &mut [T], rngs)`.
  Calling them with a `&Vec` or `&mut Vec` still works.
- The stop condition receives a `&GenerationReport` instead of `(best_fitness, iterations,
  generations_without_improvement)`: write `|report| report.generation >= 100` instead of
  `|_, iterations, _| iterations >= 100`. `report.generation` is 0 for the initial population,
  where `iterations` counted it as 1, so a ported condition runs one more generation.
- `generations_without_improvement` (in the report and in `Metrics`) counts the generations since
  the best fitness of the whole run improved. It used to count the generations whose best was
  equal to the previous one, so a best that dropped and came back counted as improving.
- No feature is on by default, so embedding the library compiles neither `plotters`, `toml` nor
  `serde`. Enable what you use:
  - `plot` for `Evolution::plot_chart`, `Metrics::plot_chart` and the `plotting` module.
    `ExperimentRunner` works without it, but only saves its chart with it.
  - `config` for the TOML config reader: `config_read`, and `read_config` and `RawConfig` in
    the prelude.
  - `serde` for `Serialize` and `Deserialize`. `GeneCod` implements `Deserialize` only with it,
    and `config` turns it on.
- `BuildError` has two new variants, `InitialPopulationTooLarge` and
  `InitialIndividualDimension`. It is `#[non_exhaustive]`, so only matches without a wildcard
  arm break.
- `GaussianMutation` has a new public field, `clamp_to_range`, so building it with a struct
  literal needs the field or `..Default::default()`. `GaussianMutation::new(mutation_rate,
  sigma)` builds an unclamped one.
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
- `GenerationReport`, with the generation, the best individual and its fitness, the average
  fitness, the best individual of the whole run and the generations without improvement.
- `EvolutionBuilder::with_on_generation(|report| ..)`, called after every generation, the
  initial population included.
- `Evolution::best_ever()`, the best individual seen since the run started, which without
  elitism can be better than `current_best()`.
- `Evolution::report()`, the report of the last generation.
- `EvolutionBuilder::with_initial_population(Vec<T>)`, to resume a run from a saved population
  or seed it with known good individuals. Random individuals fill the rest of the population,
  and every fitness is calculated again when the run starts.
- `IntPerm::new`, to build a permutation from a known chromosome. It panics if the chromosome is
  not a permutation of `1..=n`.
- The `serde` feature: `Serialize` and `Deserialize` for `Bin`, `Int`, `IntPerm`, `Real` and
  `GeneCod`, to save and load populations and champions.
- `GaussianMutation::clamp_to_range` keeps mutated genes within the individual's `range`. Off
  by default: the `range` still only bounds the initial population, which is what network
  weights need.
- `GaussianMutation` documents its defaults (`mutation_rate = 0.05`, `sigma = 1.0`) and how to
  choose them for a few hundred genes.
- A `cargo bench` benchmark of `Real` at 300 individuals of 300 genes, and tests at that scale
  of Gaussian mutation, elitism above 1 and tournaments above 2.
- `ExperimentResult::average_fitnesses` and `best_fitnesses` are public.
- `examples/xor`: evolves the weights of an only-brain network until it computes XOR. only-brain
  is a dev-dependency, so library users never pull it in.
- A seeded `ExperimentRunner` is reproducible, and each of its runs gets its own seed.

### Fixed

- The crate description said "paralyzed" instead of "parallelized", and `homepage` and
  `repository` pointed to `evolutionary-computing` instead of `evolutionary`.
- A manual `start()` / `next()` loop left the clock stopped at its creation, so `total_time` and
  `time_digest` reported 0. Every generation now updates it, as `run()` does.
- Every crossover that pairs parents panicked on a population of odd size, as soon as the
  unpaired last individual was picked for crossover. It now passes through unchanged.

### Changed

- Public helpers take slices instead of vectors: `convert_bin`, `NPointsCrossover::swap_sections`,
  `OrderedCrossover::apply_ox` and the `plotting` functions. Calls with a `&Vec` still compile.
- `RawConfig` converts to `EvolutionConfig` with `From` instead of `Into`, so both `.into()` and
  `EvolutionConfig::from` work.
- `Metrics` implements `Default`.
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
