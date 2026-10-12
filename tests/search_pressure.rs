//! Scheduled `sigma`, `Minimize` and fitness scaling, in whole runs.

use evolutionary::prelude::*;
use std::sync::{Arc, Mutex};

const POPULATION: u32 = 32;
const DIMENSION: u32 = 8;
const GENERATIONS: u32 = 15;

/// The distance of the genes from the origin, squared: a value to bring down to 0.
#[derive(Clone)]
struct SquaredNorm;

impl Fitness<Real> for SquaredNorm {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        individual.chromosome.iter().map(|gene| gene * gene).sum()
    }
}

fn builder(selection: impl Selection<Real>, sigma: Sigma, seed: u64) -> EvolutionBuilder<Real> {
    EvolutionBuilder::new(POPULATION, DIMENSION, (-5.0, 5.0))
        .with_fitness(Minimize(SquaredNorm))
        .with_selection(selection)
        .with_crossover(UniformCrossover::default())
        .with_mutation(GaussianMutation::new(0.2, sigma))
        .with_seed(seed)
        .with_stop_condition(|report| report.generation >= GENERATIONS)
}

#[test]
fn a_schedule_is_asked_for_each_generation_the_report_carries() {
    let asked = Arc::new(Mutex::new(Vec::new()));
    let reported = Arc::new(Mutex::new(Vec::new()));
    let sigma = {
        let asked = Arc::clone(&asked);
        Sigma::schedule(move |generation| {
            asked.lock().unwrap().push(generation);
            0.1
        })
    };

    let mut evolution = builder(TournamentSelection::default(), sigma, 3)
        .with_on_generation({
            let reported = Arc::clone(&reported);
            move |report| reported.lock().unwrap().push(report.generation)
        })
        .build()
        .unwrap();
    evolution.run();

    // Generation 0 is the random initial population, which is not mutated.
    assert_eq!(
        *asked.lock().unwrap(),
        (1..=GENERATIONS).collect::<Vec<_>>()
    );
    assert_eq!(
        *reported.lock().unwrap(),
        (0..=GENERATIONS).collect::<Vec<_>>()
    );
}

#[test]
fn a_constant_schedule_replays_a_constant_sigma() {
    let history = |sigma: Sigma| {
        let mut evolution = builder(TournamentSelection::default(), sigma, 9)
            .build()
            .unwrap();
        evolution.run();
        evolution.metrics.best_fitnesses.clone()
    };

    assert_eq!(history(0.3.into()), history(Sigma::schedule(|_| 0.3)));
    assert_ne!(history(0.3.into()), history(Sigma::schedule(|_| 0.2)));
}

#[test]
fn minimize_brings_the_value_down() {
    let mut evolution = builder(
        TournamentSelection::default(),
        Sigma::exponential_decay(0.5, 0.95),
        1,
    )
    .build()
    .unwrap();
    evolution.run();

    let history = &evolution.metrics.best_fitnesses;
    let (first, last) = (-history[0], -history[history.len() - 1]);
    assert!(last < first, "{first} -> {last}");
    assert_eq!(
        last,
        SquaredNorm.calculate_fitness(evolution.current_best())
    );
}

/// Negative fitnesses, which unscaled roulette and SUS refuse, through every scaling, on one
/// and on eight threads.
#[test]
fn scaled_selections_run_on_negative_fitnesses_and_replay_on_any_thread_count() {
    let scalings = [
        FitnessScaling::Window { epsilon: 1e-3 },
        FitnessScaling::Rank,
        FitnessScaling::Sigma,
    ];
    let builders = scalings.iter().flat_map(|&scaling| {
        [
            builder(RouletteSelection::scaled(scaling), 0.3.into(), 7),
            builder(
                StochasticUniversalSamplingSelection::scaled(scaling),
                0.3.into(),
                7,
            ),
        ]
    });

    for builder in builders {
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut evolution = builder.build().unwrap();
                    evolution.run();
                    (
                        evolution.metrics.best_fitnesses.clone(),
                        evolution.metrics.avg_fitnesses.clone(),
                    )
                })
        };

        let (best, average) = run(1);
        assert_eq!((best.clone(), average), run(8));
        assert!(best.iter().all(|&fitness| fitness < 0.0));
        assert!(best[best.len() - 1] > best[0], "{best:?}");
    }
}
