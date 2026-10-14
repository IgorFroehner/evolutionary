//! Every crossover operator, on a fresh copy of a seeded population of its representation.
//!
//! Run with `cargo bench --bench crossover`.

mod common;

use common::*;
use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkGroup, Criterion};
use evolutionary::prelude::*;

/// Crosses a clone of each population, so every iteration starts from the same parents.
fn bench<T: Individual, C: Crossover<T>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    case: &str,
    crossover: C,
    populations: &[(Size, Vec<T>)],
) {
    let rngs = Rngs::new(2);
    for (size, population) in populations {
        configure(group, *size);
        group.bench_function(size.id(case), |b| {
            b.iter_batched_ref(
                || population.clone(),
                |population| crossover.crossover(population, &rngs),
                BatchSize::SmallInput,
            )
        });
    }
}

fn n_points(n_points: usize) -> NPointsCrossover {
    NPointsCrossover {
        n_points,
        ..Default::default()
    }
}

fn crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover");

    let real = SIZES.map(|size| (size, real(size)));
    bench(
        &mut group,
        "real_uniform",
        UniformCrossover::default(),
        &real,
    );
    bench(&mut group, "real_n_points_2", n_points(2), &real);
    bench(&mut group, "real_n_points_5", n_points(5), &real);
    bench(
        &mut group,
        "arithmetic",
        ArithmeticCrossover::default(),
        &real,
    );
    bench(&mut group, "blend", BlendCrossover::default(), &real);
    bench(&mut group, "linear", LinearCrossover::default(), &real);
    bench(
        &mut group,
        "simulated_binary",
        SimulatedBinaryCrossover::default(),
        &real,
    );

    let bin = SIZES.map(|size| (size, bin(size)));
    bench(&mut group, "bin_uniform", UniformCrossover::default(), &bin);
    bench(&mut group, "bin_n_points_2", n_points(2), &bin);

    let int = SIZES.map(|size| (size, int(size)));
    bench(&mut group, "int_uniform", UniformCrossover::default(), &int);
    bench(&mut group, "int_n_points_2", n_points(2), &int);

    let perm = SIZES.map(|size| (size, perm(size)));
    bench(
        &mut group,
        "pmx",
        PartiallyMappedCrossover::default(),
        &perm,
    );
    bench(&mut group, "cycle", CycleCrossover::default(), &perm);
    bench(&mut group, "ordered", OrderedCrossover::default(), &perm);

    group.finish();
}

criterion_group! {
    name = benches;
    config = criterion();
    targets = crossover
}
criterion_main!(benches);
