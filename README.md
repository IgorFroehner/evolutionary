# Evolutionary

[![Crates.io](https://img.shields.io/crates/v/evolutionary.svg)](https://crates.io/crates/evolutionary)
[![Documentation](https://docs.rs/evolutionary/badge.svg)](https://docs.rs/evolutionary)
[![CI](https://github.com/IgorFroehner/evolutionary/actions/workflows/rust.yml/badge.svg)](https://github.com/IgorFroehner/evolutionary/actions/workflows/rust.yml)

A fully extensible Rust framework for solving problems with parallelized genetic algorithms.

- **Four built-in encodings**: `Bin`, `Int`, `IntPerm` (permutations) and `Real`, plus your own by
  implementing `Individual`.
- **Built-in operators** for selection, crossover and mutation (see the table below), plus your own
  by implementing `Selection`, `Crossover` or `Mutation`.
- **Parallel by default**: fitness and the operators run on [rayon](https://github.com/rayon-rs/rayon).
- **Reproducible**: a seeded run gives the same result however many threads it uses.
- **Observable**: a report after every generation, a best-ever individual, per-step timings and
  optional fitness charts.
- **Lean**: no optional dependency is compiled unless you ask for it.

## Getting started

```toml
[dependencies]
evolutionary = "0.4"
```

It needs Rust 1.82 or newer. With the `config` feature on Rust older than 1.85, also run
`cargo update -p indexmap --precise 2.13.1`: newer `indexmap` releases, which `toml` pulls in,
need Rust 1.85. Developing the crate itself (its tests, examples and benchmark) needs Rust
1.85, because of the dev-dependencies.

First, code your fitness function by implementing `Fitness` for the encoding you use. Here the
individual is a bit string and the fitness counts its ones:

```rust
use evolutionary::prelude::*;

#[derive(Clone)]
pub struct MaxFitness;

impl Fitness<Bin> for MaxFitness {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        individual.get_chromosome().iter().filter(|&&gene| gene).count() as f64
    }
}
```

Then build an `Evolution` with the `EvolutionBuilder`, picking the operators and the stop condition,
and run it:

```rust
fn main() {
    let mut evolution = EvolutionBuilder::new(30, 10, ())
        .with_fitness(MaxFitness)
        .with_selection(TournamentSelection::default())
        .with_crossover(NPointsCrossover::default())
        .with_mutation(BitFlipMutation::default())
        .with_stop_condition(|report| report.generation >= 1000)
        .build()
        .unwrap();

    evolution.run();

    println!("Best individual: {:?}", evolution.current_best());
    println!("Best fitness: {}", evolution.current_best_fitness());
}
```

`EvolutionBuilder::new` takes the population size, the chromosome length and the range of the
genes. The individual type (here `Bin`, from the fitness) sets the encoding and the type of the
range: `()` for `Bin` and `IntPerm`, `(i64, i64)` for `Int` and `(f64, f64)` for `Real`. `build`
returns a `BuildError` naming any missing or inconsistent parameter.

The stop condition receives a `GenerationReport`, so it can look at the generation, the best and
average fitness, or the generations without improvement:

```rust
.with_stop_condition(|report| {
    report.best_fitness == 10.0 || report.generations_without_improvement >= 50
})
```

There is a longer walkthrough in [docs/getting_started.md](./docs/getting_started.md), and the full
API is on [docs.rs](https://docs.rs/evolutionary).

## Built-in operators

Every operator has a `Default`, and most have fields or a `new` to tune them. The generic ones work
with any `Individual`, including your own.

| Operator | Works with |
|---|---|
| `TournamentSelection`, `RouletteSelection`, `RankSelection`, `StochasticUniversalSamplingSelection` | any |
| `NPointsCrossover`, `UniformCrossover` | `Bin`, `Int`, `Real` |
| `ArithmeticCrossover`, `BlendCrossover`, `LinearCrossover`, `SimulatedBinaryCrossover` | `Real` |
| `OrderedCrossover`, `PartiallyMappedCrossover`, `CycleCrossover` | `IntPerm` |
| `BitFlipMutation` | `Bin` |
| `RandomResettingMutation` | `Int`, `Real` |
| `GaussianMutation`, `DeltaMutation` | `Real` |
| `InsertionMutation` | `IntPerm` |
| `SwapMutation`, `ScrambleMutation` | any |

`DoNothingSelection`, `DoNothingCrossover` and `DoNothingMutation` skip a step entirely.

To write your own, implement the trait and pass it to the builder. Operators get their random
numbers from an `Rngs`, asking for one generator per unit of work with `rngs.rng(index)`, which is
what keeps a seeded run reproducible when the operator runs in parallel:

```rust
use evolutionary::prelude::*;
use rayon::prelude::*;

#[derive(Clone)]
struct FlipFirstGene;

impl Mutation<Bin> for FlipFirstGene {
    fn mutate(&self, population: &mut [Bin], rngs: &Rngs) {
        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, individual)| {
                let mut rng = rngs.rng(index);
                if rng.random_bool(0.5) {
                    individual.set_gene(0, !individual.get_gene(0));
                }
            });
    }
}
```

## Reproducible runs

Seed a run with `.with_seed(42)`: two runs with the same seed give the same populations and the
same fitness history, whatever the number of threads or the platform. Without a seed every run
differs, and `evolution.seed()` tells you the seed a run drew so you can replay it.

## Following a run

- `.with_on_generation(|report| ..)` is called after every generation, the initial one included,
  with the same `GenerationReport` the stop condition gets.
- `evolution.best_ever()` is the best individual seen in the whole run, which without elitism can
  be better than `current_best()`.
- `.with_elitism(n)` carries the `n` best individuals into the next generation unchanged. The
  default is 1.
- `evolution.start()` and `evolution.next()` drive the evolution one generation at a time, with
  `evolution.report()` in between, as an alternative to `run()`.
- `evolution.time_digest()` prints how long each step took.

## Starting from a known population

`.with_initial_population(individuals)` starts a run from given individuals, to resume one or to
seed it with known good solutions. Random individuals fill the rest of the population. `Bin::new`,
`Int::new`, `IntPerm::new` and `Real::new` build individuals from a chromosome, and the `serde`
feature lets you save and load them.

## Experiments

`ExperimentRunner` runs the same builder several times in parallel and collects the best and
average fitness of every generation of every run, along with the best individual and the time of
each run. `average_time`, `average_best_fitness` and `average_generations` aggregate them, and
`experiment_digest` prints them. With the `plot` feature, `save_chart(path)` saves a chart of the
averages over the runs; nothing is written unless you call it. A seeded experiment is reproducible,
and each of its runs gets its own seed, kept in its result so the run can be replayed on its own.
Calling `run` again adds runs with new seeds.

## Features

No feature is on by default, so embedding the library compiles only the genetic algorithm.

| Feature | Adds |
|---|---|
| `serde` | `Serialize` and `Deserialize` for `Bin`, `Int`, `IntPerm`, `Real` and `GeneCod`. |
| `config` | `read_config`, which reads the evolution parameters from a TOML file. Implies `serde`. |
| `plot` | `Evolution::plot_chart`, `Metrics::plot_chart` and `ExperimentRunner::save_chart`, with [plotters](https://github.com/plotters-rs/plotters). |

```toml
evolutionary = { version = "0.4", features = ["serde", "plot"] }
```

## Examples

The [examples folder](./examples) covers every encoding. Some need a feature; `cargo run` tells you
which if you forget it.

| Example | Encoding | Run with |
|---|---|---|
| [Party](./examples/party) (the getting-started problem) | `Bin` | `cargo run --example party --features plot` |
| [Math function](./examples/math_func) | `Bin` | `cargo run --example math_func --features config` |
| [Radios factory profit](./examples/radios) | `Bin` | `cargo run --example radios --features config` |
| [Maze](./examples/maze) | `Real` | `cargo run --example maze` |
| [Subway](./examples/subway) | `Real` | `cargo run --example subway` |
| [N-Queens](./examples/nqueens) | `IntPerm` | `cargo run --example nqueens --features config,plot` |
| [Valued N-Queens](./examples/nqueens_valued) | `IntPerm` | `cargo run --example nqueens_valued --features config` |
| [Travelling salesman](./examples/tsp) | `IntPerm` | `cargo run --example tsp --features plot` |
| [XOR](./examples/xor) (neuroevolution of an [only-brain](https://github.com/IgorFroehner/only-brain) network) | `Real` | `cargo run --release --example xor` |

[evolutionary-examples](https://github.com/IgorFroehner/evolutionary-examples) has visual examples
built with the [bevy game engine](https://bevyengine.org/).

## Development

```bash
cargo test                                          # unit, integration and doc tests
cargo test --all-features                           # also the serde, config and plot code
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo bench                                         # a generation of Real at 300 individuals x 300 genes
```

Charts need `fontconfig` on Linux (`libfontconfig1-dev` on Debian and Ubuntu).

A seeded run must replay identically across versions, so changes to how an operator draws its
random numbers are breaking changes and are recorded in the [changelog](./CHANGELOG.md).
`tests/golden.rs` fails when a seeded run no longer replays as it did. The
[roadmap](./ROADMAP.md) lists what is planned.

## License

[MIT](./LICENSE)
