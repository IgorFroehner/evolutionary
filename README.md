# Evolutionary

[![Crates.io](https://img.shields.io/crates/v/evolutionary.svg)](https://crates.io/crates/evolutionary)
[![Documentation](https://docs.rs/evolutionary/badge.svg)](https://docs.rs/evolutionary)

A fully extensible Rust framework for using paralyzed genetic algorithms to solve problems. 

Currently, it supports coding in `Binary`, `Real`, `Permuted Integers`, `Integers` and any other coding you may want to implement. 
It also has built in implementation of the following genetic operators:

- Selection:
  - [Roulette Wheel](./src/selection/roulette_selection.rs)
  - [Tournament](./src/selection/tournament.rs)
- Crossover:
  - [Multiple Point Crossover (MPX) (Binary)](./src/crossover/bin_crossover.rs) 
  - [Partially Mapped Crossover (PMX) (Permuted)](./src/crossover/pmx_crossover.rs)
  - [Cycle Crossover (CX) (Permuted)](./src/crossover/cx_crossover.rs)
- Mutation:
  - [Bit Flip (Binary)](./src/mutation/bin_mutation.rs)
  - [Swap (Permuted)](./src/mutation/perm_mutation.rs)

As well you can code your own selection, crossover or mutation and use it on the `EvolutionBuilder`.

## Example:

First you'll need to code your Fitness function:

```rust
use evolutionary::prelude::*; 

fn f(individual: &Bin) -> f64 {
    let mut sum = 0.;

    for i in 0..individual.0.len() {
        if individual.0[i] {
            sum += 1.;
        }
    }

    sum
}

#[derive(Clone)]
pub struct MaxFitness;

impl Fitness<Bin> for MaxFitness {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        f(individual)
    }
}
```

Then you'll need to create a coding for you individual so you can get the response at the end:

```rust
#[derive(Clone)]
struct MaxCoding;

impl Coding<Bin> for MaxCoding {
    type Output = f64;

    fn decode(&self, individual: &Bin) -> Self::Output {
        f(individual)
    }
}
```

Then you will be able to build a evolution object using the `EvolutionBuiler` and setting all the required parameters:

```rust
fn main() {
    let mut evolution = EvolutionBuilder::new(30, 10, GeneCod::Bin, ())
        .with_fitness(MaxFitness)
        .with_selection(TournamentSelection::default())
        .with_crossover(BinCrossover::default())
        .with_mutation(BinMutation::default())
        .with_title("Max".to_string())
        .with_stop_condition(move |_, iterations, _| iterations >= 1000)
        .with_coding(MaxCoding)
        .build().unwrap();

    evolution.run();

    println!("Best individual: {:?}", evolution.current_best());
    println!("Best fitness: {}", evolution.current_best_fitness());
}
```

Find this and other examples in the [examples folder](./examples).
