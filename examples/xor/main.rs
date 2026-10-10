//! Neuroevolution "hello world": evolve the weights of an only-brain network until it computes
//! XOR. Each individual is a `Real` holding every weight and bias of a 2 -> 3 -> 1 network.
//!
//! Run with `cargo run --release --example xor`.

use evolutionary::prelude::*;
use only_brain::NeuralNetwork;

type Network = NeuralNetwork<2, 1>;

const HIDDEN: [usize; 1] = [3];

const CASES: [([f64; 2], f64); 4] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];

/// Below this squared error over the four cases, every output is within about 0.05 of its
/// target.
const SOLVED: f64 = 0.01;

fn network(individual: &Real) -> Network {
    Network::from_parameters(&HIDDEN, individual.get_chromosome())
}

fn squared_error(network: &Network) -> f64 {
    CASES
        .iter()
        .map(|(inputs, target)| (network.feed_forward(inputs)[0] - target).powi(2))
        .sum()
}

/// The fitness is maximised, so it is the error with its sign flipped.
#[derive(Clone)]
struct XorFitness;

impl Fitness<Real> for XorFitness {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        -squared_error(&network(individual))
    }
}

fn main() {
    let dimension = Network::parameter_count_for(&HIDDEN) as u32;

    let mut evolution = EvolutionBuilder::new(100, dimension, GeneCod::Real, (-1.0, 1.0))
        .with_fitness(XorFitness)
        .with_selection(TournamentSelection::new(3, 0.9))
        .with_crossover(UniformCrossover::default())
        // Only 13 genes, so mutate about one in five. The steps are big next to the initial range:
        // genes are unbounded, and XOR needs weights well outside (-1, 1).
        .with_mutation(GaussianMutation::new(0.2, 0.5))
        .with_elitism(2)
        .with_seed(1)
        .with_on_generation(|report| {
            if report.generation % 100 == 0 {
                println!(
                    "generation {:>4}: squared error {:.5}",
                    report.generation, -report.best_fitness
                );
            }
        })
        .with_stop_condition(|report| -report.best_fitness < SOLVED || report.generation >= 3000)
        .build()
        .unwrap();

    evolution.run();

    let report = evolution.report();
    let champion = network(report.best_ever);
    println!(
        "\n{} parameters, generation {}, squared error {:.5}",
        dimension,
        report.generation,
        squared_error(&champion)
    );
    for (inputs, target) in CASES {
        let [output] = champion.feed_forward(&inputs);
        println!("{inputs:?} -> {output:.3} (expected {target})");
    }

    assert!(squared_error(&champion) < SOLVED, "XOR was not solved");
}
