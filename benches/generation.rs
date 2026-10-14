//! One generation, `Evolution::next`, of each representation, with a cheap fitness so that the
//! time is the algorithm's own, and the `Real` evolution with the options that change its cost.
//!
//! Every iteration measures generation 1 of the same seeded run: the cost of a generation
//! changes as the population evolves, so going on with one run would measure a different part
//! of it depending on how fast the code is.
//!
//! Run with `cargo bench --bench generation`.

mod common;

use common::*;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use evolutionary::prelude::*;

/// Builds and starts `builder` outside of the measured time, then measures its first `next`.
fn bench<T: Individual>(
    c: &mut Criterion,
    case: &str,
    sizes: &[Size],
    builder: fn(Size) -> EvolutionBuilder<T>,
) {
    let mut group = c.benchmark_group("generation");
    for &size in sizes {
        configure(&mut group, size);
        group.bench_function(size.id(case), |b| {
            b.iter_batched_ref(
                || {
                    let mut evolution = builder(size).build().unwrap();
                    evolution.start();
                    evolution
                },
                |evolution| evolution.next(),
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

fn bin_evolution(size: Size) -> EvolutionBuilder<Bin> {
    EvolutionBuilder::new(size.population, size.genes, ())
        .with_fitness(CountOnes)
        .with_selection(TournamentSelection::default())
        .with_crossover(NPointsCrossover::default())
        .with_mutation(BitFlipMutation {
            mutation_rate: size.rate(),
        })
        .with_elitism(5)
        .with_stop_condition(|_| false)
        .with_seed(SEED)
}

fn int_evolution(size: Size) -> EvolutionBuilder<Int> {
    EvolutionBuilder::new(size.population, size.genes, INT_RANGE)
        .with_fitness(Sum)
        .with_selection(TournamentSelection::default())
        .with_crossover(UniformCrossover::default())
        .with_mutation(RandomResettingMutation {
            mutation_rate: size.rate(),
        })
        .with_elitism(5)
        .with_stop_condition(|_| false)
        .with_seed(SEED)
}

fn perm_evolution(size: Size) -> EvolutionBuilder<IntPerm> {
    EvolutionBuilder::new(size.population, size.genes, ())
        .with_fitness(TourLength)
        .with_selection(TournamentSelection::default())
        .with_crossover(PartiallyMappedCrossover::default())
        .with_mutation(SwapMutation::new(size.rate()))
        .with_elitism(5)
        .with_stop_condition(|_| false)
        .with_seed(SEED)
}

fn representations(c: &mut Criterion) {
    bench(c, "real", &GENERATION_SIZES, real_evolution);
    bench(c, "bin", &GENERATION_SIZES, bin_evolution);
    bench(c, "int", &GENERATION_SIZES, int_evolution);
    bench(c, "perm", &GENERATION_SIZES, perm_evolution);
}

fn options(c: &mut Criterion) {
    bench(c, "real_no_elitism", &GENERATION_SIZES, |size| {
        real_evolution(size).with_elitism(0)
    });
    bench(c, "real_reevaluate_elitists", &GENERATION_SIZES, |size| {
        real_evolution(size).with_reevaluate_elitists(true)
    });
}

fn fitnesses(c: &mut Criterion) {
    bench(c, "real_heavy_fitness", &SIZES, |size| {
        real_evolution(size).with_fitness(Heavy)
    });
    bench(c, "real_batch_fitness", &[MEDIUM], |size| {
        real_evolution(size).with_fitness(BatchNearOrigin)
    });
}

criterion_group! {
    name = benches;
    config = criterion();
    targets = representations, options, fitnesses
}
criterion_main!(benches);
