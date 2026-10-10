# AGENTS.md

This file provides guidance to coding agents (Claude Code, Codex, and others) when working with
code in this repository.

## What this is

`evolutionary` is a Rust library crate (no binary) for parallelized genetic algorithms, published
on crates.io. It is also the GA engine behind the author's neuroevolution work (only-brain
networks), but nothing in the library may know about networks or racing: every feature must be
useful for any problem encoded as `Bin`, `Int`, `IntPerm` or `Real`. See `ROADMAP.md`.

## Commands

```bash
cargo test                                              # unit + integration + doc tests, no features
cargo test --all-features                               # also serde, config and plot code (CI runs both)
cargo test --test reproducibility                       # one integration test file
cargo test --test reproducibility bin                   # tests in that file whose name contains "bin"
cargo test --doc                                        # doc tests only
cargo clippy --all-targets --all-features -- -D warnings  # CI fails on any warning
cargo bench                                             # criterion: a generation of Real at 300x300
cargo run --example xor --release                       # examples: see required-features in Cargo.toml
```

`plot` pulls in `plotters`, which needs `fontconfig` on Linux (`libfontconfig1-dev`). Examples
that read `Config.toml` need `--features config`; those that save charts need `--features plot`.
Cargo refuses to build an example without its `required-features`, so check `Cargo.toml` first.
`maze`, `subway` and `xor` are auto-discovered and need no feature.

## Architecture

The whole algorithm lives in `src/evolution.rs`. One generation (`Evolution::next`) is:

1. `find_elitists` copies the `elitism` best individuals (default 1).
2. `Selection::get_mating_pool(&population, &rngs)` returns a new population.
3. `Crossover::crossover(&mut pool, &rngs)` and `Mutation::mutate(&mut pool, &rngs)` modify it in
   place.
4. `process_fitness` evaluates every individual in parallel (rayon).
5. `replace_worsts_with_elitists` puts the elitists back, then `finish_generation` computes the
   generation stats once (best index, average), updates `best_ever`, `metrics` and calls the
   `on_generation` callback.

`start()` builds generation 0 from `initial_population` plus random individuals; `run()` is
`start()` then `next()` until the stop condition holds. Both the stop condition and the callback get
a `GenerationReport` (`src/report.rs`), which borrows from the current population.

Operators and fitness are trait objects (`Box<dyn Selection<T>>` etc.) that must be `Clone`,
hence `DynClone`. `EvolutionBuilder` (`src/evolution_builder.rs`) is the only intended way to
construct an `Evolution`; its `build` validates everything and returns a `#[non_exhaustive]`
`BuildError`. `ExperimentRunner` runs one builder many times in parallel.

Encodings are in `src/population/`; each is a struct with a `Vec<Gene>` and a cached `fitness`,
implementing `Individual`. Operators are in `src/selection/`, `src/crossover/` and `src/mutation/`,
split by encoding where they are not generic. Everything public is re-exported through
`evolutionary::prelude`.

## Randomness and reproducibility (the hard constraint)

A seeded run must replay identically across versions, thread counts and platforms. This drives
most design decisions:

- No operator may use `rand::random()`, `thread_rng()` or any shared generator. Every operator
  receives an `Rngs` (`src/rng.rs`) and asks for one ChaCha8 generator per unit of work with
  `rngs.rng(index)`: one per mating-pool slot in selection, one per parent pair in crossover, one
  per individual in mutation. The generator is keyed by `(seed, generation, step, index)`, so
  rayon's scheduling cannot change the result.
- Changing the order or number of draws an operator makes, the key derivation in `Rngs::rng`, or
  the floating-point summation order of the average fitness changes every seeded run. That is a
  breaking change: record it in `CHANGELOG.md` and bump accordingly. Optimizations that would
  change the draw pattern (for example geometric skip-sampling in per-gene mutations) have been
  deliberately rejected for this reason.
- `src/rng.rs` pins the derivation with `numbers_are_stable_across_versions`;
  `tests/reproducibility.rs` checks that 1 thread and 8 threads agree for every built-in operator.
  Neither checks cross-version stability, so before an optimization pass, fingerprint seeded runs
  (final population plus fitness-history sums) in a temporary test and verify byte-identical output
  after.
- The average fitness is summed sequentially on purpose; do not parallelize it.

## Conventions

- `NaN` fitness never panics: all comparisons go through `fitness_cmp` and `selection_weight` in
  `src/fitness.rs`, where `NaN` ranks below every number. Keep using them instead of `partial_cmp`.
- Built-in crossovers pair the population two by two; with an odd size the last individual passes
  through unchanged (`tests/odd_population.rs`).
- Public helpers and operator traits take slices (`&[T]`, `&mut [T]`), not `&Vec<T>`.
- Each release has a dated section in `CHANGELOG.md` with Breaking / Added / Fixed / Changed /
  Performance headings; breaking changes are written as "write X instead of Y" migration notes.
  `ROADMAP.md` tracks planned work with checkboxes.
- Optional dependencies stay behind features (`serde`, `config`, `plot`), all off by default. New
  code that needs one must be `#[cfg(feature = "...")]`-gated and must compile under both
  `cargo test` and `cargo test --all-features`.
- Operator doc comments carry a runnable example; `cargo test --doc` must stay green.
