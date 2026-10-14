//! Every selection operator, on a `Real` population with varied positive fitnesses.
//!
//! Run with `cargo bench --bench selection`.

mod common;

use common::*;
use criterion::{criterion_group, criterion_main, Criterion};
use evolutionary::prelude::*;
use std::hint::black_box;

/// The cases. Both tournaments keep the default `kp = 1.0`.
fn selections() -> Vec<(&'static str, Box<dyn Selection<Real>>)> {
    let window = FitnessScaling::Window { epsilon: 0.1 };
    vec![
        ("tournament_k2", Box::new(TournamentSelection::default())),
        ("tournament_k5", Box::new(TournamentSelection::new(5, 1.0))),
        ("roulette", Box::new(RouletteSelection)),
        ("rank", Box::new(RankSelection)),
        ("sus", Box::new(StochasticUniversalSamplingSelection)),
        (
            "roulette_window",
            Box::new(RouletteSelection::scaled(window)),
        ),
        (
            "roulette_rank",
            Box::new(RouletteSelection::scaled(FitnessScaling::Rank)),
        ),
        (
            "roulette_sigma",
            Box::new(RouletteSelection::scaled(FitnessScaling::Sigma)),
        ),
        (
            "sus_window",
            Box::new(StochasticUniversalSamplingSelection::scaled(window)),
        ),
        (
            "sus_rank",
            Box::new(StochasticUniversalSamplingSelection::scaled(
                FitnessScaling::Rank,
            )),
        ),
        (
            "sus_sigma",
            Box::new(StochasticUniversalSamplingSelection::scaled(
                FitnessScaling::Sigma,
            )),
        ),
    ]
}

fn selection(c: &mut Criterion) {
    // Not SEED, which would replay the random numbers that generated the population.
    let rngs = Rngs::new(OPERATOR_SEED);
    let populations = SIZES.map(|size| (size, with_fitnesses(real(size))));

    let mut group = c.benchmark_group("selection");
    for (case, selection) in selections() {
        for (size, population) in &populations {
            configure(&mut group, *size);
            group.bench_function(size.id(case), |b| {
                b.iter(|| black_box(selection.get_mating_pool(population, &rngs)))
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = criterion();
    targets = selection
}
criterion_main!(benches);
