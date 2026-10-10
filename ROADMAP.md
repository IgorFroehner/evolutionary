# Roadmap

Push evolutionary forward so it can evolve neural-network weights for
racelearn (race cars driven by
[only-brain](https://github.com/IgorFroehner/only-brain) networks), while staying a
generic genetic algorithm library. Nothing here should know about networks or racing:
every item below is useful for any problem encoded as `Real`, `Bin`, `Int` or
`IntPerm`.

Step 1 shipped in 0.2.0. Steps 2 to 7 make up 0.3.0.

## 1. Bug fixes (0.2.0)

- [x] **Population is twice the configured size.** `Evolution::start` built the
  population in parallel and then still ran the old sequential loop (left over from
  `7e7a33d`), so every run evolved `2 × population_size` individuals.
- [x] **Elitism validated against the wrong value.** `build` compared `elitism` with
  `dimension` instead of `population_size`.
- [x] **`build` panicked without a stop condition.** `build` now returns a typed
  `BuildError` naming what is missing or inconsistent, and also rejects an empty
  population.
- [x] **NaN fitness panicked.** Every fitness comparison goes through one ordering where
  `NaN` ranks below every number, and gets no weight in roulette and SUS selection. The
  average ignores `NaN`s, and the elitism heaps (and the `ordered-float` dependency) are
  gone.
- [x] **Tournament size was fixed.** `TournamentSelection::new(k, kp)`. It also no
  longer indexes out of bounds with `k = 1`, and clones only the winner.
- [x] **`Real` couldn't be built from a chromosome.** `Real::new` and `Int::new`.
- [x] **`GaussianMutation` and `ScrambleMutation` were unreachable.** They were never
  re-exported, so users couldn't construct them.
- [x] Tests for each of these.

Ships in 0.2.0: `BuildError` replacing the `String` error of `build` is a breaking
change. Breaking changes are recorded in `CHANGELOG.md` as they land.

## 2. Reproducible runs with `rand` 0.9 (0.3.0, breaking)

- [x] Move to `rand` 0.9 and `rand_distr` 0.5, matching only-brain.
- [x] `EvolutionBuilder::with_seed(u64)`. Without a seed, behave as today.
- [x] Stop calling `thread_rng()` / `random()` inside operators (27 files today).
  Inside rayon that can never be reproducible, since work-stealing decides which
  thread handles which individual. `Individual::generate_member` takes the generator too.
- [x] Give each parallel task its own RNG derived from `(seed, generation, index)`, so
  results don't depend on thread scheduling. The step (selection, crossover, mutation) is
  part of the key too, so the steps of a generation don't share streams.
- [x] Pass the RNG to operators: change the `Selection`, `Crossover` and `Mutation`
  trait signatures. This is the breaking change behind 0.3.0.
- [x] Test: two runs with the same seed produce the same best chromosome and the same
  fitness history. They run on 1 and on 8 threads and cover every built-in operator.

## 3. Observing a run

- [x] `EvolutionBuilder::with_on_generation(|report| ..)`: a callback called after
  every generation with the generation number, the best individual, and the best and
  average fitness. This is how a caller saves the champion of each generation.
- [x] **Best ever / hall of fame**: keep the best individual seen in the whole run,
  since without elitism the current best can be lost. `Evolution::best_ever()`, also in
  the report.
- [x] Consider passing the same report to the stop condition instead of
  `(f64, u32, u32)`. Done: both get a `GenerationReport`.
- [x] Make a manual `start()` / `next()` loop end the clock too, so it is a full
  alternative to `run()`.

## 4. Starting and saving populations

- [x] `EvolutionBuilder::with_initial_population(Vec<T>)`, to resume a run or seed it
  with known-good individuals (for example saved pilots, or weights drawn from a
  custom initialisation). Random individuals fill the rest, and `IntPerm::new` lets
  permutations be seeded too.
- [x] Optional `serde` feature: `Serialize` / `Deserialize` for the built-in
  individuals, so populations and champions can be saved and loaded. Making `serde`
  optional meant putting the TOML config reader behind a `config` feature (on by default)
  already.

## 5. `Real` at neuroevolution scale

- [x] Tests and a benchmark for a few hundred individuals with a few hundred genes:
  `GaussianMutation`, elitism > 1 and tournament selection with `k > 2`.
  `tests/real_at_scale.rs` and `benches/real_at_scale.rs` (about 0.4 ms a generation at
  300 × 300 on an M-series laptop).
- [x] Document `GaussianMutation`'s defaults (`mutation_rate = 0.05`, `sigma = 1.0`)
  and when to lower `sigma`.
- [x] Optional clamping of mutated genes to `range`. Keep unbounded as the default:
  today `range` only bounds initialisation, which is what network weights need.
  `GaussianMutation::clamp_to_range`.
- `Fitness` stays `&T -> f64`. Lexicographic goals (laps first, lap time second) can be
  packed into one number by the caller, so no change is needed.

## 6. Lean and tidy

- [x] Put `plotters` and the `toml` config reader behind cargo features, so projects
  embedding the GA don't compile a charting stack. `plot`, `config` and `serde`, all off by
  default.
- [x] Fix crate metadata: `homepage` and `repository` point to
  `evolutionary-computing`, and the description says "paralyzed" instead of
  "parallelized".
- [x] Remove the unused-import warnings in `crossover` and `mutation`.
- [ ] CI: also run `cargo clippy` and `cargo test --all-features`. The `--all-features`
  and `--no-default-features` test runs are in (step 4); clippy is left.
- [ ] Release 0.3.0: bump the version and date the `CHANGELOG.md` entry.

## 7. Neuroevolution "hello world": XOR

Depends on only-brain 0.3 (`parameter_count()` and `from_parameters`).

- [ ] `examples/xor`: evolve an only-brain network to solve XOR, with `Real`
  individuals of `dimension = parameter_count()` and a fitness that builds the network
  with `from_parameters` and scores its four outputs.
- [ ] only-brain is a **dev-dependency** only, so library users never pull it in.
