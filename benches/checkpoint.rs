//! A checkpoint round trip of the `Real` evolution after 10 generations: taking it and saving it
//! to JSON, reading it back, and resuming an evolution from it.
//!
//! Needs the `serde` feature: run with `cargo bench --features serde --bench checkpoint`.

mod common;

use common::*;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use evolutionary::prelude::*;
use std::hint::black_box;

const GENERATIONS: u32 = 10;

fn checkpoint(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint");
    let evolutions = SIZES.map(|size| {
        let mut evolution = real_evolution(size).build().unwrap();
        evolution.start();
        for _ in 0..GENERATIONS {
            evolution.next();
        }
        let json = serde_json::to_string(&evolution.checkpoint()).unwrap();
        (size, evolution, json)
    });

    for (size, evolution, _) in &evolutions {
        configure(&mut group, *size);
        group.bench_function(size.id("to_json"), |b| {
            b.iter(|| serde_json::to_string(&evolution.checkpoint()).unwrap())
        });
    }

    for (size, _, json) in &evolutions {
        configure(&mut group, *size);
        group.bench_function(size.id("from_json"), |b| {
            b.iter(|| serde_json::from_str::<Checkpoint<Real>>(black_box(json)).unwrap())
        });
    }

    for (size, evolution, _) in &evolutions {
        let checkpoint = evolution.checkpoint();
        configure(&mut group, *size);
        group.bench_function(size.id("resume"), |b| {
            b.iter_batched(
                || checkpoint.clone(),
                |checkpoint| {
                    let mut resumed = real_evolution(*size)
                        .with_checkpoint(checkpoint)
                        .build()
                        .unwrap();
                    resumed.start();
                    resumed
                },
                BatchSize::PerIteration,
            )
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = criterion();
    targets = checkpoint
}
criterion_main!(benches);
