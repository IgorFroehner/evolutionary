//! A run stopped at a checkpoint and resumed must be exactly the run that never stopped.

use evolutionary::prelude::*;
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const POPULATION: u32 = 24;
const DIMENSION: u32 = 10;
const GENERATIONS: u32 = 30;
const CHECKPOINT_AT: u32 = 15;
const SEED: u64 = 11;

/// Everything a resumed run must agree on with the uninterrupted one. Fitnesses are compared bit
/// for bit, so that `NaN`s compare equal and a change in the last digit counts.
#[derive(Debug, PartialEq)]
struct Outcome<G> {
    seed: Option<u64>,
    generation: u32,
    /// Every individual in the population's order, with its fitness.
    population: Vec<(Vec<G>, u64)>,
    best_fitnesses: Vec<u64>,
    avg_fitnesses: Vec<u64>,
    best_ever: (Vec<G>, u64),
    generations_without_improvement: u32,
    average_fitness: u64,
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn individual<T: Individual>(individual: &T) -> (Vec<T::Gene>, u64) {
    (
        individual.get_chromosome().to_vec(),
        individual.get_fitness().to_bits(),
    )
}

fn outcome<T: Individual>(evolution: &Evolution<T>) -> Outcome<T::Gene> {
    let report = evolution.report();

    Outcome {
        seed: evolution.seed(),
        generation: report.generation,
        population: evolution.population().iter().map(individual).collect(),
        best_fitnesses: bits(&evolution.metrics.best_fitnesses),
        avg_fitnesses: bits(&evolution.metrics.avg_fitnesses),
        best_ever: individual(evolution.best_ever()),
        generations_without_improvement: report.generations_without_improvement,
        average_fitness: report.average_fitness.to_bits(),
    }
}

fn in_pool<R: Send>(threads: usize, work: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(work)
}

/// Runs `CHECKPOINT_AT` generations and takes a checkpoint.
fn checkpoint_of<T: Individual>(builder: EvolutionBuilder<T>) -> Checkpoint<T> {
    let mut evolution = builder.build().unwrap();
    evolution.start();
    for _ in 0..CHECKPOINT_AT {
        evolution.next();
    }

    evolution.checkpoint()
}

/// Runs `GENERATIONS` generations in one go, and again in two halves with a checkpoint in
/// between, resumed into a fresh `Evolution`, and checks that both agree.
fn assert_resumes_exactly<T: Individual>(builder: fn() -> EvolutionBuilder<T>, threads: usize)
where
    T::Gene: PartialEq + Debug,
{
    in_pool(threads, || {
        let mut uninterrupted = builder().build().unwrap();
        uninterrupted.run();

        let checkpoint = checkpoint_of(builder());
        assert_eq!(checkpoint.generation, CHECKPOINT_AT);
        assert_eq!(checkpoint.seed, SEED);

        let mut resumed = builder().with_checkpoint(checkpoint).build().unwrap();
        resumed.run();

        assert_eq!(resumed.report().generation, GENERATIONS);
        assert_eq!(outcome(&resumed), outcome(&uninterrupted));
    });
}

fn stop<T: Individual>(report: &GenerationReport<T>) -> bool {
    report.generation >= GENERATIONS
}

#[derive(Clone)]
struct Sphere;

impl Fitness<Real> for Sphere {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        -individual
            .get_chromosome()
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
    }
}

fn real() -> EvolutionBuilder<Real> {
    EvolutionBuilder::new(POPULATION, DIMENSION, (-2.0, 2.0))
        .with_fitness(Sphere)
        .with_selection(TournamentSelection::new(3, 0.8))
        .with_crossover(BlendCrossover::default())
        .with_mutation(GaussianMutation::new(0.2, 0.3))
        .with_elitism(2)
        .with_stop_condition(stop)
        .with_seed(SEED)
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

/// Without elitism, so the best ever is not always in the population.
fn bin() -> EvolutionBuilder<Bin> {
    EvolutionBuilder::new(POPULATION, 40, ())
        .with_fitness(CountOnes)
        .with_selection(RouletteSelection)
        .with_crossover(NPointsCrossover::default())
        .with_mutation(BitFlipMutation { mutation_rate: 0.2 })
        .with_elitism(0)
        .with_stop_condition(stop)
        .with_seed(SEED)
}

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

fn perm() -> EvolutionBuilder<IntPerm> {
    EvolutionBuilder::new(POPULATION, DIMENSION, ())
        .with_fitness(InPlace)
        .with_selection(StochasticUniversalSamplingSelection)
        .with_crossover(PartiallyMappedCrossover::default())
        .with_mutation(SwapMutation::default())
        .with_stop_condition(stop)
        .with_seed(SEED)
}

#[test]
fn real_resumes_exactly_on_one_and_on_eight_threads() {
    assert_resumes_exactly(real, 1);
    assert_resumes_exactly(real, 8);
}

#[test]
fn bin_without_elitism_resumes_exactly_on_one_and_on_eight_threads() {
    assert_resumes_exactly(bin, 1);
    assert_resumes_exactly(bin, 8);
}

#[test]
fn perm_resumes_exactly_on_one_and_on_eight_threads() {
    assert_resumes_exactly(perm, 1);
    assert_resumes_exactly(perm, 8);
}

#[test]
fn the_bin_run_has_lost_its_best_when_the_checkpoint_is_taken() {
    // Otherwise it would not check that the best ever is restored.
    let checkpoint = checkpoint_of(bin());

    let best_in_population = checkpoint
        .population
        .iter()
        .map(|individual| individual.get_fitness())
        .fold(f64::MIN, f64::max);
    assert!(best_in_population < checkpoint.best_ever.get_fitness());
}

#[test]
fn a_resumed_run_starts_where_the_checkpoint_was_taken() {
    let mut saved = real().build().unwrap();
    saved.start();
    for _ in 0..CHECKPOINT_AT {
        saved.next();
    }

    let mut resumed = real().with_checkpoint(saved.checkpoint()).build().unwrap();
    resumed.start();

    assert_eq!(outcome(&resumed), outcome(&saved));
    assert_eq!(resumed.metrics.iterations, saved.metrics.iterations);
}

#[test]
fn a_checkpoint_of_the_initial_population_resumes_the_whole_run() {
    let mut first = real().build().unwrap();
    first.start();

    let mut resumed = real().with_checkpoint(first.checkpoint()).build().unwrap();
    resumed.run();
    let mut uninterrupted = real().build().unwrap();
    uninterrupted.run();

    assert_eq!(outcome(&resumed), outcome(&uninterrupted));
}

#[test]
fn a_resumed_run_can_be_checkpointed_and_resumed_again() {
    let mut uninterrupted = real().build().unwrap();
    uninterrupted.run();

    let mut middle = real()
        .with_checkpoint(checkpoint_of(real()))
        .build()
        .unwrap();
    middle.start();
    for _ in 0..7 {
        middle.next();
    }
    let mut last = real().with_checkpoint(middle.checkpoint()).build().unwrap();
    last.run();

    assert_eq!(outcome(&last), outcome(&uninterrupted));
}

/// Counts every fitness it computes.
#[derive(Clone)]
struct Counting(Arc<AtomicUsize>);

impl Fitness<Real> for Counting {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        self.0.fetch_add(1, Ordering::Relaxed);
        Sphere.calculate_fitness(individual)
    }
}

#[test]
fn resuming_computes_no_fitness() {
    let count = Arc::new(AtomicUsize::new(0));
    let checkpoint = checkpoint_of(real());

    let mut resumed = real()
        .with_fitness(Counting(Arc::clone(&count)))
        .with_checkpoint(checkpoint)
        .build()
        .unwrap();
    resumed.start();
    assert_eq!(count.load(Ordering::Relaxed), 0);

    resumed.next();
    assert_eq!(count.load(Ordering::Relaxed), POPULATION as usize);
}

#[test]
fn every_start_resumes_from_the_checkpoint() {
    let mut resumed = real()
        .with_checkpoint(checkpoint_of(real()))
        .build()
        .unwrap();
    resumed.run();
    let first = outcome(&resumed);

    resumed.run();

    assert_eq!(outcome(&resumed), first);
}

#[test]
fn the_callback_sees_the_generations_after_the_checkpoint() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&seen);

    let mut resumed = real()
        .with_checkpoint(checkpoint_of(real()))
        .with_on_generation(move |report| recorder.lock().unwrap().push(report.generation))
        .build()
        .unwrap();
    resumed.run();

    assert_eq!(
        *seen.lock().unwrap(),
        (CHECKPOINT_AT + 1..=GENERATIONS).collect::<Vec<u32>>()
    );
}

#[test]
fn the_stop_condition_can_run_a_resumed_run_further() {
    let mut longer = real()
        .with_stop_condition(|report| report.generation >= GENERATIONS + 5)
        .build()
        .unwrap();
    longer.run();

    let mut resumed = real().build().unwrap();
    resumed.run();
    let mut resumed = real()
        .with_checkpoint(resumed.checkpoint())
        .with_stop_condition(|report| report.generation >= GENERATIONS + 5)
        .build()
        .unwrap();
    resumed.run();

    assert_eq!(outcome(&resumed), outcome(&longer));
}

#[test]
fn the_checkpoint_carries_the_seed_of_an_unseeded_run() {
    let unseeded = || {
        EvolutionBuilder::new(POPULATION, DIMENSION, (-2.0, 2.0))
            .with_fitness(Sphere)
            .with_selection(TournamentSelection::default())
            .with_crossover(BlendCrossover::default())
            .with_mutation(GaussianMutation::new(0.2, 0.3))
            .with_stop_condition(stop)
    };
    let mut first = unseeded().build().unwrap();
    first.start();
    let checkpoint = first.checkpoint();
    let seed = first.seed().unwrap();
    assert_eq!(checkpoint.seed, seed);
    for _ in 0..GENERATIONS {
        first.next();
    }

    let mut resumed = unseeded().with_checkpoint(checkpoint).build().unwrap();
    resumed.run();

    assert_eq!(resumed.seed(), Some(seed));
    assert_eq!(outcome(&resumed), outcome(&first));
}

#[test]
fn the_same_seed_as_the_checkpoint_is_accepted() {
    let checkpoint = checkpoint_of(real());

    assert!(real()
        .with_seed(SEED)
        .with_checkpoint(checkpoint)
        .build()
        .is_ok());
}

#[test]
fn another_seed_than_the_checkpoint_is_an_error() {
    let checkpoint = checkpoint_of(real());

    let error = real()
        .with_seed(SEED + 1)
        .with_checkpoint(checkpoint)
        .build()
        .err()
        .unwrap();

    assert_eq!(
        error,
        BuildError::CheckpointSeed {
            seed: SEED + 1,
            checkpoint_seed: SEED
        }
    );
}

#[test]
fn a_checkpoint_and_an_initial_population_is_an_error() {
    let checkpoint = checkpoint_of(real());
    let champion = checkpoint.best_ever.clone();

    let error = real()
        .with_initial_population(vec![champion])
        .with_checkpoint(checkpoint.clone())
        .build()
        .err();
    assert_eq!(error, Some(BuildError::CheckpointWithInitialPopulation));

    // An empty initial population is the same as none.
    assert!(real()
        .with_initial_population(Vec::new())
        .with_checkpoint(checkpoint)
        .build()
        .is_ok());
}

#[test]
fn a_checkpoint_of_the_wrong_dimension_is_an_error() {
    let checkpoint = checkpoint_of(real());

    let error = EvolutionBuilder::new(POPULATION, DIMENSION + 1, (-2.0, 2.0))
        .with_fitness(Sphere)
        .with_selection(TournamentSelection::default())
        .with_crossover(BlendCrossover::default())
        .with_mutation(GaussianMutation::default())
        .with_stop_condition(stop)
        .with_checkpoint(checkpoint)
        .build()
        .err()
        .unwrap();

    assert_eq!(
        error,
        BuildError::CheckpointIndividualDimension {
            index: Some(0),
            genes: DIMENSION as usize,
            dimension: DIMENSION + 1
        }
    );
    assert!(error.to_string().contains("individual 0 has 10 genes"));
}

#[test]
fn a_best_ever_of_the_wrong_dimension_is_an_error() {
    let mut checkpoint = checkpoint_of(real());
    checkpoint.best_ever.chromosome.pop();

    let error = real().with_checkpoint(checkpoint).build().err().unwrap();

    assert_eq!(
        error,
        BuildError::CheckpointIndividualDimension {
            index: None,
            genes: DIMENSION as usize - 1,
            dimension: DIMENSION
        }
    );
    assert!(error.to_string().contains("best ever has 9 genes"));
}

#[test]
fn a_checkpoint_of_another_population_size_is_an_error() {
    let mut checkpoint = checkpoint_of(real());
    checkpoint.population.pop();

    let error = real().with_checkpoint(checkpoint).build().err();

    assert_eq!(
        error,
        Some(BuildError::CheckpointPopulationSize {
            given: POPULATION as usize - 1,
            population_size: POPULATION
        })
    );
}

#[test]
fn a_history_that_does_not_match_the_generation_is_an_error() {
    let mut checkpoint = checkpoint_of(real());
    checkpoint.avg_fitnesses.pop();

    let error = real().with_checkpoint(checkpoint).build().err().unwrap();

    assert_eq!(
        error,
        BuildError::CheckpointHistory {
            generation: CHECKPOINT_AT,
            best_fitnesses: CHECKPOINT_AT as usize + 1,
            avg_fitnesses: CHECKPOINT_AT as usize
        }
    );
    assert!(error.to_string().contains("needs 16"));
}

#[test]
fn a_checkpoint_can_be_built_from_its_parts() {
    let saved = checkpoint_of(bin());

    let rebuilt = Checkpoint::new(
        saved.seed,
        saved.generation,
        saved.population.clone(),
        saved.best_ever.clone(),
        saved.best_fitnesses.clone(),
        saved.avg_fitnesses.clone(),
        saved.generations_without_improvement,
    );
    let mut from_parts = bin().with_checkpoint(rebuilt).build().unwrap();
    from_parts.run();
    let mut uninterrupted = bin().build().unwrap();
    uninterrupted.run();

    assert_eq!(outcome(&from_parts), outcome(&uninterrupted));
}

#[test]
fn an_experiment_branches_off_a_checkpoint() {
    let checkpoint = checkpoint_of(real());

    let mut experiment = ExperimentRunner::new(
        "branches".to_string(),
        3,
        real().with_checkpoint(checkpoint.clone()),
    );
    experiment.run().unwrap();
    let results = &experiment.experiment_results;

    // Every run goes on from the checkpoint's history...
    for result in results {
        assert_eq!(
            result.best_fitnesses[..=CHECKPOINT_AT as usize],
            checkpoint.best_fitnesses[..]
        );
        assert_eq!(result.iterations, GENERATIONS + 1);
    }
    // ...in a direction of its own.
    assert_ne!(results[0].seed, results[1].seed);
    assert_ne!(results[0].average_fitnesses, results[1].average_fitnesses);

    // A run replays with its seed in the checkpoint.
    let mut branch = checkpoint.clone();
    branch.seed = results[1].seed;
    let mut replay = real()
        .with_seed(results[1].seed)
        .with_checkpoint(branch)
        .build()
        .unwrap();
    replay.run();
    assert_eq!(replay.metrics.avg_fitnesses, results[1].average_fitnesses);
}

#[cfg(feature = "serde")]
mod serde {
    use super::*;

    #[test]
    fn a_checkpoint_resumes_exactly_after_a_round_trip_through_json() {
        let json = serde_json::to_string(&checkpoint_of(real())).unwrap();
        let loaded: Checkpoint<Real> = serde_json::from_str(&json).unwrap();

        let mut resumed = real().with_checkpoint(loaded).build().unwrap();
        resumed.run();
        let mut uninterrupted = real().build().unwrap();
        uninterrupted.run();

        assert_eq!(outcome(&resumed), outcome(&uninterrupted));
    }

    #[test]
    fn nan_fitnesses_round_trip_through_json() {
        let never_evaluated = Bin::new(vec![true; DIMENSION as usize]);
        let checkpoint = Checkpoint::new(
            3,
            1,
            vec![never_evaluated.clone()],
            never_evaluated.clone(),
            vec![f64::NAN, 2.0],
            vec![f64::NAN, f64::NAN],
            1,
        );

        let json = serde_json::to_string(&checkpoint).unwrap();
        let loaded: Checkpoint<Bin> = serde_json::from_str(&json).unwrap();

        assert!(json.contains("null"), "{json}");
        assert!(loaded.population[0].get_fitness().is_nan());
        assert!(loaded.best_ever.get_fitness().is_nan());
        assert!(loaded.best_fitnesses[0].is_nan());
        assert_eq!(loaded.best_fitnesses[1], 2.0);
        assert!(loaded.avg_fitnesses.iter().all(|average| average.is_nan()));
        assert_eq!(loaded.seed, 3);
        assert_eq!(loaded.generation, 1);
        assert_eq!(loaded.generations_without_improvement, 1);
    }
}
