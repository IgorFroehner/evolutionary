//! Shared by every benchmark: the sizes, the criterion settings, cheap fitnesses and seeded
//! populations of each representation.
//!
//! Each bench uses only part of it.
#![allow(dead_code)]

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use evolutionary::prelude::*;
use std::hint::black_box;
use std::time::Duration;

/// A population size and a number of genes per individual.
#[derive(Clone, Copy)]
pub struct Size {
    pub population: u32,
    pub genes: u32,
}

pub const SMALL: Size = Size {
    population: 50,
    genes: 50,
};
pub const MEDIUM: Size = Size {
    population: 300,
    genes: 300,
};
pub const LARGE: Size = Size {
    population: 2000,
    genes: 500,
};

/// The sizes of a generation.
pub const GENERATION_SIZES: [Size; 3] = [SMALL, MEDIUM, LARGE];
/// The sizes of an operator or a checkpoint.
pub const SIZES: [Size; 2] = [MEDIUM, LARGE];

/// The seed of every population and evolution, so each run measures the same work.
pub const SEED: u64 = 1;

impl Size {
    /// The per-gene mutation rate: one gene per individual on average.
    pub fn rate(self) -> f64 {
        1.0 / self.genes as f64
    }

    /// `<case>/<pop>x<genes>`: the benchmark id within its group.
    pub fn id(self, case: &str) -> BenchmarkId {
        BenchmarkId::new(case, format!("{}x{}", self.population, self.genes))
    }
}

/// A short warm-up and measurement, so the whole suite runs in a few minutes, and a 3% noise
/// threshold, as runs of the same code move by a few percent. `--warm-up-time`,
/// `--measurement-time` and `--noise-threshold` on the command line override them.
pub fn criterion() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .noise_threshold(0.03)
}

/// Sets the throughput to the population, so criterion also reports the time per individual,
/// and takes 20 samples instead of 100 at the large size, whose iterations are long.
///
/// It applies to the benchmarks added to `group` after it, so call it before each one.
pub fn configure(group: &mut BenchmarkGroup<'_, WallTime>, size: Size) {
    group.throughput(Throughput::Elements(size.population as u64));
    if size.population >= LARGE.population {
        group.sample_size(20);
    } else {
        group.sample_size(100);
    }
}

/// A seeded population of `size`, without fitnesses.
pub fn population<T: Individual>(size: Size, range: &T::RangeType) -> Vec<T> {
    let rngs = Rngs::new(SEED);
    (0..size.population as usize)
        .map(|index| T::generate_member(size.genes, range, &mut rngs.rng(index)))
        .collect()
}

pub const REAL_RANGE: (f64, f64) = (-1.0, 1.0);
pub const INT_RANGE: (i64, i64) = (0, 100);

pub fn real(size: Size) -> Vec<Real> {
    population(size, &REAL_RANGE)
}

pub fn bin(size: Size) -> Vec<Bin> {
    population(size, &())
}

pub fn int(size: Size) -> Vec<Int> {
    population(size, &INT_RANGE)
}

pub fn perm(size: Size) -> Vec<IntPerm> {
    population(size, &())
}

/// Gives the individuals varied positive fitnesses, in no order, as fitness-proportional
/// selections need.
pub fn with_fitnesses<T: Individual>(mut population: Vec<T>) -> Vec<T> {
    for (index, individual) in population.iter_mut().enumerate() {
        individual.set_fitness((index * 7919 % 1009 + 1) as f64);
    }
    population
}

/// `Real`: the closer to the origin, the better.
#[derive(Clone)]
pub struct NearOrigin;

impl Fitness<Real> for NearOrigin {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        -individual
            .get_chromosome()
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
    }
}

/// `Bin`: the number of ones.
#[derive(Clone)]
pub struct CountOnes;

impl Fitness<Bin> for CountOnes {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        individual
            .get_chromosome()
            .iter()
            .filter(|&&gene| gene)
            .count() as f64
    }
}

/// `Int`: the sum of the genes.
#[derive(Clone)]
pub struct Sum;

impl Fitness<Int> for Sum {
    fn calculate_fitness(&self, individual: &Int) -> f64 {
        individual.get_chromosome().iter().sum::<i64>() as f64
    }
}

/// `IntPerm`: the length of a tour where the distance between two cities is the difference of
/// their numbers, to make shorter.
#[derive(Clone)]
pub struct TourLength;

impl Fitness<IntPerm> for TourLength {
    fn calculate_fitness(&self, individual: &IntPerm) -> f64 {
        let tour = individual.get_chromosome();
        let closing = (tour[tour.len() - 1] - tour[0]).abs();
        let length: i64 = tour.windows(2).map(|pair| (pair[1] - pair[0]).abs()).sum();
        -((length + closing) as f64)
    }
}

/// The steps of [`Heavy`]: about 50 µs an individual on an M-series laptop.
pub const HEAVY_STEPS: u32 = 18_000;

/// `Real`: [`NearOrigin`] followed by a fixed chain of floating point operations, so that an
/// individual costs about 50 µs, as a small simulation would.
#[derive(Clone)]
pub struct Heavy;

impl Fitness<Real> for Heavy {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        let mut x = NearOrigin.calculate_fitness(individual);
        for _ in 0..HEAVY_STEPS {
            x = black_box(x * 0.999_999 + 1e-9);
        }
        x
    }
}

/// `Real`: [`NearOrigin`] evaluated as a batch, in one sequential pass over the population.
#[derive(Clone)]
pub struct BatchNearOrigin;

impl Fitness<Real> for BatchNearOrigin {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        NearOrigin.calculate_fitness(individual)
    }

    fn calculate_population(&self, population: &mut [Real]) {
        for individual in population.iter_mut() {
            let fitness = NearOrigin.calculate_fitness(individual);
            individual.set_fitness(fitness);
        }
    }
}

/// The `Real` evolution of the generation and checkpoint benchmarks: tournament, uniform
/// crossover and Gaussian mutation, with 5 elitists.
pub fn real_evolution(size: Size) -> EvolutionBuilder<Real> {
    EvolutionBuilder::new(size.population, size.genes, REAL_RANGE)
        .with_fitness(NearOrigin)
        .with_selection(TournamentSelection::new(3, 0.9))
        .with_crossover(UniformCrossover::default())
        .with_mutation(GaussianMutation::new(size.rate(), 0.1))
        .with_elitism(5)
        .with_stop_condition(|_| false)
        .with_seed(SEED)
}
