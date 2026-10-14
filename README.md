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
evolutionary = "0.5"
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

The evolution maximizes the fitness. To minimize something instead (an error, a cost), wrap the
fitness that computes it in `Minimize`, as in `.with_fitness(Minimize(Error))`: it negates the
value, so the reports carry `-error` and `-report.best_fitness` is the error itself.

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
| `RouletteSelection::scaled(scaling)`, `StochasticUniversalSamplingSelection::scaled(scaling)`, with a `FitnessScaling` (`Window`, `Rank` or `Sigma`) that accepts negative fitnesses | any |
| `NPointsCrossover`, `UniformCrossover` | `Bin`, `Int`, `Real` |
| `ArithmeticCrossover`, `BlendCrossover`, `LinearCrossover`, `SimulatedBinaryCrossover` | `Real` |
| `OrderedCrossover`, `PartiallyMappedCrossover`, `CycleCrossover` | `IntPerm` |
| `BitFlipMutation` | `Bin` |
| `RandomResettingMutation` | `Int`, `Real` |
| `GaussianMutation`, `DeltaMutation` | `Real` |
| `InsertionMutation` | `IntPerm` |
| `SwapMutation`, `ScrambleMutation` | any |

`DoNothingSelection`, `DoNothingCrossover` and `DoNothingMutation` skip a step entirely.

`RouletteSelection` and `StochasticUniversalSamplingSelection` weigh individuals by their raw
fitness, which must not be negative; their `scaled` versions weigh them by a `FitnessScaling`
of it instead. `GaussianMutation`'s `sigma` is a number or a `Sigma`, which can change with the
generation, for example `Sigma::exponential_decay(0.5, 0.99)` to explore first and refine later.

To write your own, implement the trait and pass it to the builder. Operators get their random
numbers from an `Rngs`, asking for one generator per unit of work with `rngs.rng(index)`, which is
what keeps a seeded run reproducible when the operator runs in parallel. The crate re-exports the
`rayon` it runs on as `evolutionary::rayon`, so a parallel operator needs no `rayon` dependency of
its own:

```rust
use evolutionary::prelude::*;
use evolutionary::rayon::prelude::*;

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

## Expensive or batched fitness

The evolution evaluates each generation with one call to `Fitness::calculate_population`, which
by default runs `calculate_fitness` on every individual in parallel. When the population is
cheaper to evaluate together, for example a simulation that steps every individual at once, a
batched forward pass of many networks or a GPU kernel, override `calculate_population` and set
each individual's fitness with `set_fitness`. With a noisy fitness (random starting conditions,
a random sample of test cases), add `.with_reevaluate_elitists(true)`: the elitists are then
evaluated again every generation instead of keeping their fitness, so one lucky evaluation
can't stay at the top for the whole run. It costs `elitism` extra evaluations per generation,
passed in the same `calculate_population` call.

## Reproducible runs

Seed a run with `.with_seed(42)`: two runs with the same seed give the same populations and the
same fitness history, whatever the number of threads. Without a seed every run differs, and
`evolution.seed()` tells you the seed a run drew so you can replay it. A version that changes
how an operator draws its random numbers says so under "Breaking" in the
[changelog](./CHANGELOG.md): a run recorded with an earlier version replays with that version.

A seeded run is also the same on every platform, with one exception among the built-in
operators: `GaussianMutation`. It draws from `rand_distr`'s `Normal`, which calls the
platform's `ln` and `exp`. IEEE 754 does not require those to be correctly rounded, so on rare
draws two platforms' math libraries can differ in the last bit, and the run then diverges. The
other operators use only `+`, `-`, `*`, `/` and `sqrt`, which IEEE 754 specifies exactly, or
the `libm` crate's pure-Rust `pow` (`SimulatedBinaryCrossover`), which is the same code
everywhere. Your own fitness, operators and `Sigma` schedules replay across platforms under the
same condition. "Every platform" means every target whose `f64` arithmetic follows IEEE 754,
which leaves out 32-bit x86 without SSE2 (the `i586` targets).

The fitness must be deterministic for a run to replay: it gets no random generator, so a noisy
fitness should derive its noise from a hash of the chromosome. The `Fitness` documentation shows
how to make that noise change every generation and still replay.

## Following a run

- `.with_on_generation(|report| ..)` is called after every generation, the initial one included,
  with the same `GenerationReport` the stop condition gets.
- `evolution.best_ever()` is the best individual seen in the whole run, which without elitism can
  be better than `current_best()`.
- `.with_elitism(n)` carries the `n` best individuals into the next generation unchanged. The
  default is 1. With `.with_reevaluate_elitists(true)` they are evaluated again instead of
  keeping their fitness.
- `evolution.start()` and `evolution.next()` drive the evolution one generation at a time, with
  `evolution.report()` in between, as an alternative to `run()`.
- `evolution.population()` borrows the current population; `current_population()` returns a
  copy sorted by fitness.
- `evolution.time_digest()` prints how long each step took.

## Starting from a known population

`.with_initial_population(individuals)` starts a new run from given individuals, to seed it with
known good solutions. Random individuals fill the rest of the population, and the run starts at
generation 0. `Bin::new`, `Int::new`, `IntPerm::new` and `Real::new` build individuals from a
chromosome, and the `serde` feature lets you save and load them. An individual that was never
evaluated has a `NaN` fitness, which ranks below every number.

## Checkpoints

`evolution.checkpoint()` saves where a run is: its seed, its generation, the population with its
fitnesses, the best ever and the fitness history. `.with_checkpoint(checkpoint)` on the same
builder resumes from it exactly, as if the run had never stopped: the next generation draws the
same random numbers, no fitness is computed again, and the result is the one the uninterrupted
run gives, however many threads either uses. The resumed run keeps the checkpoint's seed, so
`.with_seed` is not needed (`build` fails if it gives another), and it can't be combined with
`.with_initial_population`. The stop condition can differ, for example to run longer.

```rust
evolution.start();
while !evolution.reached_stop_condition() {
    evolution.next();
    if evolution.report().generation % 100 == 0 {
        save(evolution.checkpoint()); // your own storage
    }
}

// Later, maybe in another process:
let mut resumed = builder().with_checkpoint(load()).build()?;
resumed.run();
```

With the `serde` feature a `Checkpoint` is `Serialize` and `Deserialize`, so you pick the format.
To resume exactly, the format must give back every `f64` bit for bit:

- A binary format such as CBOR (with `ciborium`) does, `NaN` and infinities included.
- For JSON, add `float_roundtrip` to `serde_json` in your own dependencies (this crate's
  dev-dependency does not turn it on for you). Without it the last bit of some numbers changes
  on the way back, and the resumed run drifts away from the saved one:

  ```toml
  serde_json = { version = "1", features = ["float_roundtrip"] }
  ```

- JSON has no `NaN` nor infinities, and `serde_json` writes both as `null`, which loads back as
  `NaN`. A `NaN` fitness survives, but an infinite one comes back as `NaN`, and the resume is
  then **not exact**: that individual ranks below every number, so selection, `best_ever` and
  `generations_without_improvement` diverge from the saved run. If your fitness can be infinite,
  use a binary format.

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
| `serde` | `Serialize` and `Deserialize` for `Bin`, `Int`, `IntPerm`, `Real`, `GeneCod` and `Checkpoint`. |
| `config` | `read_config`, which reads the evolution parameters from a TOML file. Implies `serde`. |
| `plot` | `Evolution::plot_chart`, `Metrics::plot_chart` and `ExperimentRunner::save_chart`, with [plotters](https://github.com/plotters-rs/plotters). |

```toml
evolutionary = { version = "0.5", features = ["serde", "plot"] }
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
cargo bench --all-features                          # generations, operators and checkpoints (serde)
```

The benchmarks cover a generation of every representation, every operator and checkpoints at
up to 2000 individuals of 500 genes, in about 7 minutes; the checkpoint ones need the `serde`
feature. [benches/README.md](./benches/README.md) has the names and how to compare a branch
with main.

Charts need `fontconfig` on Linux (`libfontconfig1-dev` on Debian and Ubuntu).

A seeded run must replay identically across versions, so changes to how an operator draws its
random numbers are breaking changes and are recorded in the [changelog](./CHANGELOG.md).
`tests/golden.rs` fails when a seeded run no longer replays as it did. The
[roadmap](./ROADMAP.md) lists what is planned.

## License

[MIT](./LICENSE)
