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
cargo test --test golden                                # seeded runs replay exactly as recorded
cargo test --doc                                        # doc tests only
cargo clippy --all-targets --all-features -- -D warnings  # CI fails on any warning
cargo fmt --all -- --check                              # CI checks formatting
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features  # CI fails on rustdoc warnings
cargo bench --all-features                              # criterion suite, see benches/README.md
cargo run --example xor --release                       # examples: see required-features in Cargo.toml
```

`plot` pulls in `plotters`, which needs `fontconfig` on Linux (`libfontconfig1-dev`). Examples
that read `Config.toml` need `--features config`; those that save charts need `--features plot`.
Cargo refuses to build an example without its `required-features`, so check `Cargo.toml` first.
`maze`, `subway` and `xor` are auto-discovered and need no feature.

## Benchmarks

A PR that can change performance (`src/evolution.rs`, an operator, a representation, fitness
evaluation, the RNG, a bump of `rand`, `rand_distr`, `rayon` or `libm`) includes a
before-and-after report in its description, produced by following
`.claude/skills/bench-report/SKILL.md`. The script that renders the table from criterion's
output, `compare.py`, lives next to it.

## Architecture

The whole algorithm lives in `src/evolution.rs`. One generation (`Evolution::next`) is:

1. `find_elitists` copies the `elitism` best individuals (default 1).
2. `Selection::get_mating_pool(&population, &rngs)` returns a new population.
3. `Crossover::crossover(&mut pool, &rngs)` and `Mutation::mutate(&mut pool, &rngs)` modify it in
   place.
4. `process_fitness` hands the population to `Fitness::calculate_population`, once per
   generation, whose default evaluates every individual in parallel (rayon). With
   `with_reevaluate_elitists(true)` the elitists are appended to the population for that call
   and split off again, so they get a fresh fitness in the same call.
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

A seeded run must replay identically across versions, thread counts and platforms (with the
one documented exception below). This drives most design decisions:

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
  `tests/reproducibility.rs` checks that 1 thread and 8 threads agree for every built-in operator;
  `tests/golden.rs` checks that one seeded run per encoding gives exactly the recorded final
  population and fitness history, and between them its runs use every built-in selection, every
  per-gene mutation, a scaled selection (`FitnessScaling::Rank`) and a scheduled `sigma` too.
  Its other scalings
  (`Window`, `Sigma`) are only covered by the thread test. A
  failing golden test is a breaking change: fix it, or record it in `CHANGELOG.md` and regenerate
  the constants in the same change (the file's header says how). An operator no golden run uses
  is only covered by the thread test, so before optimizing one, add it to a golden run first.
  Record that run's constants on code where the operator is unchanged since the last release
  (ideally check out the release tag), and commit them before the optimization. Constants
  recorded after the change pin the optimized behavior and catch nothing.
- Across platforms, only `+`, `-`, `*`, `/`, `sqrt` (and comparisons, `abs`, `min`/`max`,
  `floor`/`round`, casts) give the same bits everywhere: IEEE 754 specifies them exactly, and
  Rust never fuses `a * b + c` on its own. `powf`, `exp`, `ln`, `log*`, `cbrt`, `hypot`, the
  trigonometric functions and the like go to the platform's math library, which may round them
  differently in the last bit (Apple's `pow` and glibc's disagree on some SBX draws, for
  example). `powi` can differ between debug and release builds (LLVM expands a constant
  exponent inline into multiplications, otherwise it calls compiler-builtins' `__powidf2`), and
  `mul_add` falls back to the platform's software `fma` on a target without a hardware FMA,
  which isn't always conforming. Any such function whose result feeds a seeded stream, a gene
  or a selection decision must come from the `libm` crate (`libm::pow`, `libm::exp`,
  `libm::fma` for a fused multiply-add, ...), which is the same Rust code everywhere, or be
  built from exact operations (`Sigma::exponential_decay`'s repeated squaring). Never write
  `powi` for a square: write `x * x`. Code that only feeds metrics, charts or tests may use
  `std`. `libm`'s default `arch` feature sends only exactly-specified functions (`sqrt`, `fma`,
  `rint`, ...) to hardware instructions, never `pow`.
- The `libm` requirement is a caret (`0.2.16`) on purpose, like `rand`'s and `rand_distr`'s: a
  user's `cargo update` can pick up a later 0.2.x, so a `libm` release that rewrites `pow`
  (0.2.12 did that to `cbrt`) changes their SBX streams without a release of this crate. The
  golden tests and `simulated_binary_crossover.rs`'s `pow_is_pinned_bit_for_bit` only see it
  when our lockfile moves, which is Dependabot's monthly cargo PR. If such a bump fails them,
  either cap the requirement (`">=0.2.16, <0.2.N"`), or re-record the constants, list it under
  Breaking and raise the floor to `0.2.N`, so that every user of the release resolves the
  `pow` its golden run was recorded with.
- Known exception: `GaussianMutation` draws from `rand_distr::Normal`, whose ziggurat calls
  the platform's `ln` (the tail values) and `exp` (accepting a draw in a wedge), so it can
  diverge across platforms on rare draws. README "Reproducible runs" lists it; add any other
  exception there.
- The average fitness is summed sequentially on purpose; do not parallelize it.
- The fitness gets no `Rngs`: it must be deterministic for a seeded run to replay, and a noisy
  one derives its noise from a fixed hash of the chromosome (see the `Fitness` docs). Fitness
  evaluation and `with_reevaluate_elitists` draw no random numbers.

## Conventions

- `NaN` fitness never panics: all comparisons go through `fitness_cmp` and `selection_weight` in
  `src/fitness.rs`, where `NaN` ranks below every number. Keep using them instead of `partial_cmp`.
- Built-in crossovers pair the population two by two; with an odd size the last individual passes
  through unchanged (`tests/odd_population.rs`).
- Public helpers and operator traits take slices (`&[T]`, `&mut [T]`), not `&Vec<T>`, and
  `Individual` hands out its chromosome as one, so operators never change its length.
- Each release has a dated section in `CHANGELOG.md` with Breaking / Added / Fixed / Changed /
  Performance headings; breaking changes are written as "write X instead of Y" migration notes.
  `ROADMAP.md` tracks planned work with checkboxes.
- The minimum supported Rust version is `rust-version` in `Cargo.toml` (1.82), and CI checks
  the library, with every feature, on that toolchain. Do not use a newer standard-library API
  without raising it; clippy's `incompatible_msrv` flags most. The dev-dependencies need Rust
  1.85 (edition 2024), so the MSRV job does not build the tests.
- `Cargo.lock` must keep every library dependency buildable on that version: the MSRV job checks
  with `--locked`. `.cargo/config.toml` makes `cargo update` (cargo 1.84 and later) prefer
  versions whose `rust-version` fits, so update with a recent cargo. A plain fresh resolve picks
  `indexmap` 2.14 (edition 2024) through `toml`.
- Optional dependencies stay behind features (`serde`, `config`, `plot`), all off by default. New
  code that needs one must be `#[cfg(feature = "...")]`-gated and must compile under both
  `cargo test` and `cargo test --all-features`.
- Operator doc comments carry a runnable example; `cargo test --doc` must stay green.
- Parallelism is per individual (mutation, fitness), per pair (crossover) or per slot
  (selection). Before adding a `par_iter`, measure: at the sizes the crate targets (a few hundred
  individuals of a few hundred genes) one rayon fork and join costs tens of microseconds, so an
  operator whose work per item is a clone or a multiplication is faster sequential. SUS is kept
  sequential for that reason, with the measurement in `CHANGELOG.md`. `RAYON_NUM_THREADS=1 cargo
  bench` shows what each benchmark gains from the pool.
