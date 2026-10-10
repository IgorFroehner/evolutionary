# Evolutionary

[![Crates.io](https://img.shields.io/crates/v/evolutionary.svg)](https://crates.io/crates/evolutionary)
[![Documentation](https://docs.rs/evolutionary/badge.svg)](https://docs.rs/evolutionary)

A fully extensible Rust framework for using parallelized genetic algorithms to solve problems.

Currently, it supports coding in `Binary`, `Real`, `Permuted Integers`, `Integers` and any other coding you may want to 
implement. Check out the built-in implementation for the genetic operators:

- [Selection](./src/selection)
- [Crossover](./src/crossover)
- [Mutation](./src/mutation)

You can also code your own `selection`, `crossover` or `mutation` implementing the traits and passing them to the 
`EvolutionBuilder`.

## Getting Started:

First you'll need to code your Fitness function:

```rust
use evolutionary::prelude::*;

#[derive(Clone)]
pub struct MaxFitness;

impl Fitness<Bin> for MaxFitness {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        let mut sum = 0.;
  
        for i in 0..individual.get_chromosome().len() {
            if individual.get_gene(i) {
                sum += 1.;
            }
        }
  
        sum
    }
}
```

Then you will be able to build an evolution object using the `EvolutionBuilder` and setting all the required parameters:

```rust
fn main() {
    let mut evolution = EvolutionBuilder::new(30, 10, GeneCod::Bin, ())
        .with_fitness(MaxFitness)
        .with_selection(TournamentSelection::default())
        .with_crossover(NPointsCrossover::default())
        .with_mutation(BitFlipMutation::default())
        .with_stop_condition(|report| report.generation >= 1000)
        .build().unwrap();

    evolution.run();

    println!("Best individual: {:?}", evolution.current_best());
    println!("Best fitness: {}", evolution.current_best_fitness());
}
```

To make a run reproducible, seed it with `.with_seed(42)`: two runs with the same seed give the
same result, however many threads they use. Without a seed, `evolution.seed()` tells you the one
a run drew, so you can replay it.

`.with_on_generation(|report| ..)` follows a run generation by generation, and
`.with_initial_population(saved)` starts one from known individuals, for example a population
saved with the `serde` feature, which implements `Serialize` and `Deserialize` for the built-in
individuals.

There is an extended getting started [here](./docs/getting_started.md).

## Features:

None is on by default, so embedding the library only compiles the genetic algorithm:

- `serde`: `Serialize` and `Deserialize` for the built-in individuals, to save and load populations.
- `config`: reads the evolution parameters from a TOML file, with `read_config`.
- `plot`: saves fitness charts with `evolution.plot_chart(..)` and from the `ExperimentRunner`.

```toml
evolutionary = { version = "0.3", features = ["serde", "plot"] }
```

Some examples need them: `cargo run --example party --features plot`.

## Examples and Projects:

* [evolutionary-examples](https://github.com/IgorFroehner/evolutionary-examples) - A project with some visual 
  examples of the `evolutionary` library use, implemented with the [bevy game engine](https://bevyengine.org/).

There are some examples in the [examples folder](./examples):
* [Max](./examples/party)
* [Math Function](./examples/math_func)
* [NQueens](./examples/nqueens)
* [Valued NQueens](./examples/nqueens_valued)
* [Maximize Radios Factory Profit](./examples/radios)
* [Maze](./examples/maze)
* [Subway](./examples/subway)
* [XOR](./examples/xor) - neuroevolution: the weights of an [only-brain](https://github.com/IgorFroehner/only-brain) network

## TODO:

- Individuals:
  - [ ] Tree-based chromosomes
- Selection:
    - [ ] Parallelize the SUS Selection
- Crossover:
  - Real:
    - [ ] Linear Crossover (LX)
    - [ ] Simulated Binary Crossover (SBX)
- Usability and Performance:
  - [ ] Logs System
  - [ ] Create macros to simplify the implementation of the traits where possible;
  - [ ] Allow fitness to be a function and not a struct that must be implemented
- Examples and Benchmark
  - [ ] Implement and Optimize the Salesman problem
