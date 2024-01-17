# Evolutionary

[![Crates.io](https://img.shields.io/crates/v/evolutionary.svg)](https://crates.io/crates/evolutionary)
[![Documentation](https://docs.rs/evolutionary/badge.svg)](https://docs.rs/evolutionary)

A fully extensible Rust framework for using paralyzed genetic algorithms to solve problems. 

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

Then you will be able to build an evolution object using the `EvolutionBuiler` and setting all the required parameters:

```rust
fn main() {
    let mut evolution = EvolutionBuilder::new(30, 10, GeneCod::Bin, ())
        .with_fitness(MaxFitness)
        .with_selection(TournamentSelection::default())
        .with_crossover(NPointsCrossover::default())
        .with_mutation(BitSwapMutation::default())
        .with_stop_condition(move |_, iterations, _| iterations >= 1000)
        .build().unwrap();

    evolution.run();

    println!("Best individual: {:?}", evolution.current_best());
    println!("Best fitness: {}", evolution.current_best_fitness());
}
```

There is an extended getting started [here](./docs/getting_started.md).

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
