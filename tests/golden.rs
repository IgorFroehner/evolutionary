//! Golden replay tests: a seeded run must give exactly the same result in every version.
//!
//! Each test runs one short seeded evolution per encoding and compares a fingerprint of the
//! final population and of the best and average fitness history with recorded constants. They
//! were first recorded from 0.4.0; the history of this file shows when each was last
//! regenerated. Between them the four runs use every built-in selection and a different crossover and
//! mutation each.
//!
//! A fifth run, `real_minimize_rank_scaled_sus_decaying_sigma`, pins operators that are new in
//! 0.5.0: a scaled selection (`FitnessScaling::Rank`) and a scheduled `sigma`
//! (`Sigma::exponential_decay`). There is no older release to pin them to, so its constants
//! were recorded from the code that introduced them, before 0.5.0 was released.
//!
//! **A failing test here is a breaking change.** It means a seeded run no longer replays as it
//! did: an operator draws its random numbers in another order or number, `Rngs::rng` derives
//! its keys differently, or the floating-point evaluation order changed. If that is a bug, fix
//! it. If it is deliberate, record it under "Breaking" in `CHANGELOG.md`, bump the version
//! accordingly, and regenerate the constants in the same change:
//!
//! ```text
//! cargo test --test golden
//! ```
//!
//! Every failing test prints the fingerprint the run actually gave, as a `Golden { .. }` literal
//! to paste over the old one. Only regenerate the tests whose operators you meant to change.
//!
//! The fingerprint is a 64-bit FNV-1a hash written out by hand, rather than `DefaultHasher`,
//! whose output the standard library does not promise to keep across Rust versions. Genes and
//! fitnesses are hashed bit for bit (`f64::to_bits`), so a change in the last digit counts.

use evolutionary::prelude::*;
use std::fmt;

const POPULATION: u32 = 16;
const DIMENSION: u32 = 12;
const GENERATIONS: u32 = 20;
const SEED: u64 = 0x0E70_1000;

/// 64-bit FNV-1a.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }
}

/// Feeds a gene to the hash with a fixed byte layout.
trait HashGene {
    fn hash_into(&self, hasher: &mut Fnv);
}

impl HashGene for bool {
    fn hash_into(&self, hasher: &mut Fnv) {
        hasher.write(&[u8::from(*self)]);
    }
}

impl HashGene for i64 {
    fn hash_into(&self, hasher: &mut Fnv) {
        hasher.write_u64(*self as u64);
    }
}

impl HashGene for f64 {
    fn hash_into(&self, hasher: &mut Fnv) {
        hasher.write_u64(self.to_bits());
    }
}

fn hash_floats(values: &[f64]) -> u64 {
    let mut hasher = Fnv::new();
    hasher.write_u64(values.len() as u64);
    for value in values {
        value.hash_into(&mut hasher);
    }
    hasher.0
}

/// What a seeded run must reproduce.
#[derive(PartialEq, Eq)]
struct Golden {
    /// Generations recorded in the metrics, generation 0 included.
    generations: u32,
    population: u64,
    best_fitnesses: u64,
    avg_fitnesses: u64,
}

/// Prints as a literal to paste into the test when regenerating on purpose.
impl fmt::Debug for Golden {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Golden {{ generations: {}, population: 0x{:016x}, best_fitnesses: 0x{:016x}, \
             avg_fitnesses: 0x{:016x} }}",
            self.generations, self.population, self.best_fitnesses, self.avg_fitnesses
        )
    }
}

fn fingerprint<T: Individual>(builder: EvolutionBuilder<T>) -> Golden
where
    T::Gene: HashGene,
{
    let mut evolution = builder
        .with_stop_condition(|report| report.generation >= GENERATIONS)
        .with_seed(SEED)
        .build()
        .unwrap();
    evolution.run();

    let population = evolution.current_population();
    let mut hasher = Fnv::new();
    hasher.write_u64(population.len() as u64);
    for individual in &population {
        let chromosome = individual.get_chromosome();
        hasher.write_u64(chromosome.len() as u64);
        for gene in chromosome {
            gene.hash_into(&mut hasher);
        }
    }

    Golden {
        generations: evolution.metrics.iterations,
        population: hasher.0,
        best_fitnesses: hash_floats(&evolution.metrics.best_fitnesses),
        avg_fitnesses: hash_floats(&evolution.metrics.avg_fitnesses),
    }
}

fn assert_golden<T: Individual>(builder: EvolutionBuilder<T>, expected: Golden)
where
    T::Gene: HashGene,
{
    let actual = fingerprint(builder);
    assert_eq!(
        actual, expected,
        "a seeded run no longer replays as when its constants were recorded, which is a breaking \
         change: see the top of tests/golden.rs. If deliberate, the new constants are {actual:?}"
    );
}

#[derive(Clone)]
struct CountOnes;

impl Fitness<Bin> for CountOnes {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        individual
            .get_chromosome()
            .iter()
            .filter(|&&gene| gene)
            .count() as f64
    }
}

#[derive(Clone)]
struct SumOfGenes;

impl Fitness<Int> for SumOfGenes {
    fn calculate_fitness(&self, individual: &Int) -> f64 {
        individual.get_chromosome().iter().sum::<i64>() as f64
    }
}

/// One plus the number of genes already in their sorted place: positive, for SUS.
#[derive(Clone)]
struct InPlace;

impl Fitness<IntPerm> for InPlace {
    fn calculate_fitness(&self, individual: &IntPerm) -> f64 {
        let in_place = individual
            .get_chromosome()
            .iter()
            .enumerate()
            .filter(|&(index, &gene)| gene == index as i64 + 1)
            .count();
        1.0 + in_place as f64
    }
}

/// Positive, for roulette selection.
#[derive(Clone)]
struct Sphere;

impl Fitness<Real> for Sphere {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        let distance: f64 = individual.get_chromosome().iter().map(|x| x * x).sum();
        1.0 / (1.0 + distance)
    }
}

#[test]
fn bin_tournament_n_points_bit_flip() {
    assert_golden(
        EvolutionBuilder::new(POPULATION, DIMENSION, ())
            .with_fitness(CountOnes)
            .with_selection(TournamentSelection::new(3, 0.8))
            .with_crossover(NPointsCrossover {
                crossover_rate: 0.9,
                n_points: 2,
            })
            .with_mutation(BitFlipMutation::default())
            .with_elitism(2),
        Golden {
            generations: 21,
            population: 0x9cd77a31c999625f,
            best_fitnesses: 0x8ea40bdeaa2fa034,
            avg_fitnesses: 0x223583f5bb6b3871,
        },
    );
}

#[test]
fn int_rank_uniform_random_resetting() {
    assert_golden(
        EvolutionBuilder::new(POPULATION, DIMENSION, (-10, 10))
            .with_fitness(SumOfGenes)
            .with_selection(RankSelection)
            .with_crossover(UniformCrossover::default())
            .with_mutation(RandomResettingMutation::default()),
        Golden {
            generations: 21,
            population: 0x9e8bdd081c7f45d1,
            best_fitnesses: 0x88e17461a4762bf3,
            avg_fitnesses: 0xe6bf9abf92cb71a6,
        },
    );
}

#[test]
fn perm_sus_partially_mapped_insertion() {
    assert_golden(
        EvolutionBuilder::new(POPULATION, DIMENSION, ())
            .with_fitness(InPlace)
            .with_selection(StochasticUniversalSamplingSelection)
            .with_crossover(PartiallyMappedCrossover::default())
            .with_mutation(InsertionMutation { mutation_rate: 0.3 }),
        Golden {
            generations: 21,
            population: 0x3c6160fbe5949435,
            best_fitnesses: 0x663be6cd9d14a2cc,
            avg_fitnesses: 0x413f9caf1932977c,
        },
    );
}

#[test]
fn real_roulette_simulated_binary_gaussian() {
    assert_golden(
        EvolutionBuilder::new(POPULATION, DIMENSION, (-1.0, 1.0))
            .with_fitness(Sphere)
            .with_selection(RouletteSelection)
            .with_crossover(SimulatedBinaryCrossover::default())
            .with_mutation(GaussianMutation::new(0.2, 0.1))
            .with_elitism(3),
        Golden {
            generations: 21,
            population: 0x9c5e7b42705e7bbc,
            best_fitnesses: 0x9600a73129554763,
            avg_fitnesses: 0x962d0e3645ae0154,
        },
    );
}

/// The squared distance from the origin, to bring down with `Minimize`: every fitness is
/// negative, which only a scaled fitness-proportional selection accepts.
#[derive(Clone)]
struct SquaredDistance;

impl Fitness<Real> for SquaredDistance {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        individual.get_chromosome().iter().map(|x| x * x).sum()
    }
}

/// Recorded from 0.5.0 code, which introduced these operators: see the top of this file.
#[test]
fn real_minimize_rank_scaled_sus_decaying_sigma() {
    assert_golden(
        EvolutionBuilder::new(POPULATION, DIMENSION, (-1.0, 1.0))
            .with_fitness(Minimize(SquaredDistance))
            .with_selection(StochasticUniversalSamplingSelection::scaled(
                FitnessScaling::Rank,
            ))
            .with_crossover(BlendCrossover::default())
            .with_mutation(GaussianMutation::new(
                0.2,
                Sigma::exponential_decay(0.3, 0.9),
            ))
            .with_elitism(2),
        Golden {
            generations: 21,
            population: 0x050a5f41ddf0e2d0,
            best_fitnesses: 0x96dc59e3d9d1a19c,
            avg_fitnesses: 0x64b57572c528c03b,
        },
    );
}
