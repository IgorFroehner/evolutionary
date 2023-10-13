# Evolutionary

A fully extensible Rust framework for using paralyzed genetic algorithms to solve problems. 

Currently, it supports coding in `Binary`, `Real`, `Permuted Integers`, `Integers` and any other coding you may want to implement. 
It also has built in implementation of the following genetic operators:

- Selection:
  - Roulette Wheel
  - Tournament
- Crossover:
  - Multiple Point Crossover (MPX) (Binary) 
  - Cycle Crossover (CX) (Permuted)
  - Partially Mapped Crossover (PMX) (Permuted)
- Mutation:
  - Bit Flip (Binary)
  - Swap (Permuted)

As well you can code your own selection, crossover or mutation and use it on the `EvolutionBuilder`.

## Example:

First you'll need to code your Fitness function:

```rust
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
