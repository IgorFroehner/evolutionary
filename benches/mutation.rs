//! Every mutation operator, on a fresh copy of a seeded population of each representation it
//! works for.
//!
//! The per-gene operators mutate one gene per individual on average (`1.0 / genes`).
//! Scramble and insertion take a rate per individual, and keep their default.
//!
//! Run with `cargo bench --bench mutation`.

mod common;

use common::*;
use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkGroup, Criterion};
use evolutionary::prelude::*;

/// Mutates a clone of each population, with the mutation `mutation` builds for its size, so
/// every iteration starts from the same individuals.
fn bench<T: Individual, M: Mutation<T>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    case: &str,
    mutation: impl Fn(Size) -> M,
    populations: &[(Size, Vec<T>)],
) {
    let rngs = Rngs::new(SEED);
    for (size, population) in populations {
        let mutation = mutation(*size);
        configure(group, *size);
        group.bench_function(size.id(case), |b| {
            b.iter_batched_ref(
                || population.clone(),
                |population| mutation.mutate(population, &rngs),
                BatchSize::PerIteration,
            )
        });
    }
}

/// Swap and scramble, which work for every representation: `<representation>_swap` and
/// `<representation>_scramble`.
fn generic<T: Individual>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    representation: &str,
    populations: &[(Size, Vec<T>)],
) {
    bench(
        group,
        &format!("{representation}_swap"),
        |size| SwapMutation::new(size.rate()),
        populations,
    );
    bench(
        group,
        &format!("{representation}_scramble"),
        |_| ScrambleMutation::default(),
        populations,
    );
}

fn mutation(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutation");

    let real = SIZES.map(|size| (size, real(size)));
    bench(
        &mut group,
        "gaussian",
        |size| GaussianMutation {
            mutation_rate: size.rate(),
            ..Default::default()
        },
        &real,
    );
    bench(
        &mut group,
        "delta",
        |size| DeltaMutation {
            mutation_rate: size.rate(),
            ..Default::default()
        },
        &real,
    );
    bench(
        &mut group,
        "real_random_resetting",
        |size| RandomResettingMutation {
            mutation_rate: size.rate(),
        },
        &real,
    );
    generic(&mut group, "real", &real);

    let bin = SIZES.map(|size| (size, bin(size)));
    bench(
        &mut group,
        "bit_flip",
        |size| BitFlipMutation {
            mutation_rate: size.rate(),
        },
        &bin,
    );
    generic(&mut group, "bin", &bin);

    let int = SIZES.map(|size| (size, int(size)));
    bench(
        &mut group,
        "int_random_resetting",
        |size| RandomResettingMutation {
            mutation_rate: size.rate(),
        },
        &int,
    );
    generic(&mut group, "int", &int);

    let perm = SIZES.map(|size| (size, perm(size)));
    generic(&mut group, "perm", &perm);
    bench(
        &mut group,
        "insertion",
        |_| InsertionMutation::default(),
        &perm,
    );

    group.finish();
}

criterion_group! {
    name = benches;
    config = criterion();
    targets = mutation
}
criterion_main!(benches);
