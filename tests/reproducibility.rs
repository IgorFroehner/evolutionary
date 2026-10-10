//! A seeded run must give the same result every time, however many threads rayon uses. Each
//! test runs every built-in operator of an encoding at least once.

use evolutionary::prelude::*;
use std::fmt::Debug;

/// What two runs must agree on: the final population and the fitness history.
#[derive(Debug, PartialEq)]
struct Outcome<G> {
    population: Vec<Vec<G>>,
    best_fitnesses: Vec<f64>,
    avg_fitnesses: Vec<f64>,
}

fn run_on<T: Individual>(builder: &EvolutionBuilder<T>, threads: usize) -> Outcome<T::Gene> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    pool.install(|| {
        let mut evolution = builder.build().unwrap();
        evolution.run();

        Outcome {
            population: evolution
                .current_population()
                .iter()
                .map(|individual| individual.get_chromosome().clone())
                .collect(),
            best_fitnesses: evolution.metrics.best_fitnesses.clone(),
            avg_fitnesses: evolution.metrics.avg_fitnesses.clone(),
        }
    })
}

/// Runs `builder` with one and with several threads and checks both agree, then checks that
/// another seed gives another run.
fn assert_reproducible<T: Individual>(builder: impl Fn(u64) -> EvolutionBuilder<T>)
where
    T::Gene: PartialEq + Debug,
{
    let sequential = run_on(&builder(7), 1);
    let parallel = run_on(&builder(7), 8);
    assert_eq!(sequential, parallel);

    let other_seed = run_on(&builder(8), 8);
    assert_ne!(sequential.population, other_seed.population);
}

const POPULATION: u32 = 64;
const DIMENSION: u32 = 16;
const GENERATIONS: u32 = 25;

fn stop(_: f64, iterations: u32, _: u32) -> bool {
    iterations >= GENERATIONS
}

#[derive(Clone)]
struct CountOnes;

impl Fitness<Bin> for CountOnes {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        individual.get_chromosome().iter().filter(|&&gene| gene).count() as f64
    }
}

#[derive(Clone)]
struct SumOfGenes;

impl Fitness<Int> for SumOfGenes {
    fn calculate_fitness(&self, individual: &Int) -> f64 {
        individual.get_chromosome().iter().sum::<i64>() as f64
    }
}

/// Positive, so that it also works with roulette and SUS selection.
#[derive(Clone)]
struct Sphere;

impl Fitness<Real> for Sphere {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        let distance: f64 = individual.get_chromosome().iter().map(|x| x * x).sum();
        1.0 / (1.0 + distance)
    }
}

/// Number of genes already in their sorted place.
#[derive(Clone)]
struct InPlace;

impl Fitness<IntPerm> for InPlace {
    fn calculate_fitness(&self, individual: &IntPerm) -> f64 {
        individual
            .get_chromosome()
            .iter()
            .enumerate()
            .filter(|&(index, &gene)| gene == index as i64 + 1)
            .count() as f64
    }
}

#[test]
fn bin_tournament_n_points_bit_flip() {
    assert_reproducible(|seed| {
        EvolutionBuilder::new(POPULATION, DIMENSION, GeneCod::Bin, ())
            .with_fitness(CountOnes)
            .with_selection(TournamentSelection::new(3, 0.8))
            .with_crossover(NPointsCrossover {
                crossover_rate: 0.9,
                n_points: 2,
            })
            .with_mutation(BitFlipMutation::default())
            .with_elitism(2)
            .with_stop_condition(stop)
            .with_seed(seed)
    });
}

#[test]
fn bin_roulette_uniform_scramble() {
    assert_reproducible(|seed| {
        EvolutionBuilder::new(POPULATION, DIMENSION, GeneCod::Bin, ())
            .with_fitness(CountOnes)
            .with_selection(RouletteSelection)
            .with_crossover(UniformCrossover::default())
            .with_mutation(ScrambleMutation { mutation_rate: 0.3 })
            .with_stop_condition(stop)
            .with_seed(seed)
    });
}

#[test]
fn int_rank_n_points_random_resetting() {
    assert_reproducible(|seed| {
        EvolutionBuilder::new(POPULATION, DIMENSION, GeneCod::Int, (-10, 10))
            .with_fitness(SumOfGenes)
            .with_selection(RankSelection)
            .with_crossover(NPointsCrossover::default())
            .with_mutation(RandomResettingMutation::default())
            .with_stop_condition(stop)
            .with_seed(seed)
    });
}

#[test]
fn int_tournament_uniform_swap() {
    assert_reproducible(|seed| {
        EvolutionBuilder::new(POPULATION, DIMENSION, GeneCod::Int, (-10, 10))
            .with_fitness(SumOfGenes)
            .with_selection(TournamentSelection::default())
            .with_crossover(UniformCrossover::default())
            .with_mutation(SwapMutation::default())
            .with_stop_condition(stop)
            .with_seed(seed)
    });
}

#[test]
fn real_tournament_blend_gaussian() {
    assert_reproducible(|seed| {
        EvolutionBuilder::new(POPULATION, DIMENSION, GeneCod::Real, (-1.0, 1.0))
            .with_fitness(Sphere)
            .with_selection(TournamentSelection::new(4, 0.9))
            .with_crossover(BlendCrossover::default())
            .with_mutation(GaussianMutation {
                mutation_rate: 0.1,
                sigma: 0.1,
            })
            .with_elitism(3)
            .with_stop_condition(stop)
            .with_seed(seed)
    });
}

#[test]
fn real_sus_arithmetic_delta() {
    assert_reproducible(|seed| {
        EvolutionBuilder::new(POPULATION, DIMENSION, GeneCod::Real, (-1.0, 1.0))
            .with_fitness(Sphere)
            .with_selection(StochasticUniversalSamplingSelection)
            .with_crossover(ArithmeticCrossover::default())
            .with_mutation(DeltaMutation::default())
            .with_stop_condition(stop)
            .with_seed(seed)
    });
}

#[test]
fn real_roulette_n_points_and_uniform_random_resetting() {
    assert_reproducible(|seed| {
        EvolutionBuilder::new(POPULATION, DIMENSION, GeneCod::Real, (-1.0, 1.0))
            .with_fitness(Sphere)
            .with_selection(RouletteSelection)
            .with_crossover(NPointsCrossover::default())
            .with_mutation(RandomResettingMutation::default())
            .with_stop_condition(stop)
            .with_seed(seed)
    });
    assert_reproducible(|seed| {
        EvolutionBuilder::new(POPULATION, DIMENSION, GeneCod::Real, (-1.0, 1.0))
            .with_fitness(Sphere)
            .with_selection(RankSelection)
            .with_crossover(UniformCrossover::default())
            .with_mutation(GaussianMutation::default())
            .with_stop_condition(stop)
            .with_seed(seed)
    });
}

#[test]
fn perm_ordered_insertion() {
    assert_reproducible(|seed| {
        EvolutionBuilder::new(POPULATION, DIMENSION, GeneCod::IntPerm, ())
            .with_fitness(InPlace)
            .with_selection(TournamentSelection::default())
            .with_crossover(OrderedCrossover::default())
            .with_mutation(InsertionMutation { mutation_rate: 0.3 })
            .with_stop_condition(stop)
            .with_seed(seed)
    });
}

#[test]
fn perm_partially_mapped_swap() {
    assert_reproducible(|seed| {
        EvolutionBuilder::new(POPULATION, DIMENSION, GeneCod::IntPerm, ())
            .with_fitness(InPlace)
            .with_selection(RankSelection)
            .with_crossover(PartiallyMappedCrossover::default())
            .with_mutation(SwapMutation::default())
            .with_stop_condition(stop)
            .with_seed(seed)
    });
}

#[test]
fn perm_cycle_scramble() {
    assert_reproducible(|seed| {
        EvolutionBuilder::new(POPULATION, DIMENSION, GeneCod::IntPerm, ())
            .with_fitness(InPlace)
            .with_selection(TournamentSelection::new(3, 1.0))
            .with_crossover(CycleCrossover::default())
            .with_mutation(ScrambleMutation { mutation_rate: 0.3 })
            .with_stop_condition(stop)
            .with_seed(seed)
    });
}

fn unseeded() -> EvolutionBuilder<Real> {
    EvolutionBuilder::new(POPULATION, DIMENSION, GeneCod::Real, (-1.0, 1.0))
        .with_fitness(Sphere)
        .with_selection(TournamentSelection::default())
        .with_crossover(BlendCrossover::default())
        .with_mutation(GaussianMutation::default())
        .with_stop_condition(stop)
}

#[test]
fn without_a_seed_every_run_differs() {
    let mut evolution = unseeded().build().unwrap();

    evolution.run();
    let first = evolution.current_best().get_chromosome().clone();
    evolution.run();
    let second = evolution.current_best().get_chromosome().clone();

    assert_ne!(first, second);
}

#[test]
fn the_seed_of_an_unseeded_run_replays_it() {
    let mut evolution = unseeded().build().unwrap();
    assert_eq!(evolution.seed(), None);

    evolution.run();
    let seed = evolution.seed().unwrap();

    let mut replay = unseeded().with_seed(seed).build().unwrap();
    replay.run();

    assert_eq!(replay.seed(), Some(seed));
    assert_eq!(
        replay.current_best().get_chromosome(),
        evolution.current_best().get_chromosome()
    );
    assert_eq!(replay.metrics.best_fitnesses, evolution.metrics.best_fitnesses);
}
