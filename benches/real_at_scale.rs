//! A generation of `Real` at the scale of neuroevolution: 300 individuals of 300 genes, with a
//! cheap fitness so that the time is the algorithm's own.
//!
//! Run with `cargo bench`.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use evolutionary::prelude::*;
use std::hint::black_box;

const POPULATION: u32 = 300;
const GENES: u32 = 300;

#[derive(Clone)]
struct NearOrigin;

impl Fitness<Real> for NearOrigin {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        -individual.get_chromosome().iter().map(|x| x * x).sum::<f64>()
    }
}

fn evolution() -> Evolution<Real> {
    let mut evolution = EvolutionBuilder::new(POPULATION, GENES, GeneCod::Real, (-1.0, 1.0))
        .with_fitness(NearOrigin)
        .with_selection(TournamentSelection::new(3, 0.9))
        .with_crossover(UniformCrossover::default())
        .with_mutation(GaussianMutation::new(1.0 / GENES as f64, 0.1))
        .with_elitism(5)
        .with_stop_condition(|_| false)
        .with_seed(1)
        .build()
        .unwrap();
    evolution.start();
    evolution
}

fn generation(c: &mut Criterion) {
    let mut evolution = evolution();

    c.bench_function("generation 300x300", |b| b.iter(|| evolution.next()));
}

fn operators(c: &mut Criterion) {
    let population = evolution().current_population();
    let rngs = Rngs::new(2);

    let gaussian = GaussianMutation::new(0.05, 0.1);
    c.bench_function("gaussian mutation 300x300", |b| {
        b.iter_batched_ref(
            || population.clone(),
            |population| gaussian.mutate(population, &rngs),
            BatchSize::SmallInput,
        )
    });

    let tournament = TournamentSelection::new(5, 0.9);
    c.bench_function("tournament k=5 300x300", |b| {
        b.iter(|| black_box(tournament.get_mating_pool(&population, &rngs)))
    });

    let uniform = UniformCrossover::default();
    c.bench_function("uniform crossover 300x300", |b| {
        b.iter_batched_ref(
            || population.clone(),
            |population| uniform.crossover(population, &rngs),
            BatchSize::SmallInput,
        )
    });

    // Fitness-proportional selections need positive fitnesses.
    let mut weighted_population = population.clone();
    for (index, individual) in weighted_population.iter_mut().enumerate() {
        individual.set_fitness((index % 97 + 1) as f64);
    }

    c.bench_function("roulette 300x300", |b| {
        b.iter(|| black_box(RouletteSelection.get_mating_pool(&weighted_population, &rngs)))
    });

    c.bench_function("rank 300x300", |b| {
        b.iter(|| black_box(RankSelection.get_mating_pool(&weighted_population, &rngs)))
    });
}

criterion_group!(benches, generation, operators);
criterion_main!(benches);
