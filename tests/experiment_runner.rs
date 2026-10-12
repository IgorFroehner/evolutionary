//! `ExperimentRunner` runs one builder many times in parallel. A seeded experiment must replay
//! identically, while its runs still differ from each other.

use std::time::Duration;

use evolutionary::experiment_runner::ExperimentResult;
use evolutionary::prelude::*;

const RUNS: u32 = 4;
const GENERATIONS: u32 = 10;

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

fn builder(seed: u64) -> EvolutionBuilder<Bin> {
    EvolutionBuilder::new(20, 16, GeneCod::Bin, ())
        .with_fitness(CountOnes)
        .with_selection(TournamentSelection::default())
        .with_crossover(NPointsCrossover::default())
        .with_mutation(BitFlipMutation::default())
        .with_stop_condition(|report| report.generation >= GENERATIONS)
        .with_seed(seed)
}

/// Runs a seeded experiment called `name` once.
fn run(name: &str, seed: u64) -> ExperimentRunner<Bin> {
    let mut runner = ExperimentRunner::new(name.to_string(), RUNS, builder(seed));
    runner.run().unwrap();
    runner
}

/// What two runs must agree on.
fn outcome(result: &ExperimentResult<Bin>) -> (Vec<bool>, Vec<f64>, Vec<f64>) {
    (
        result.best_found.get_chromosome().clone(),
        result.best_fitnesses.clone(),
        result.average_fitnesses.clone(),
    )
}

#[test]
fn records_every_run() {
    let runner = run("experiment_runner_records_every_run", 1);

    assert_eq!(runner.experiment_results.len(), RUNS as usize);
    for result in &runner.experiment_results {
        assert_eq!(result.best_fitnesses.len(), result.iterations as usize);
        assert_eq!(result.average_fitnesses.len(), result.iterations as usize);
        // With elitism the best of the last generation is the best found.
        assert_eq!(
            result.best_found.get_fitness(),
            *result.best_fitnesses.last().unwrap()
        );
    }

    runner.experiment_digest();
}

#[test]
fn a_seeded_experiment_replays_and_its_runs_differ() {
    let first = run("experiment_runner_replays", 7);
    let second = run("experiment_runner_replays", 7);

    let first: Vec<_> = first.experiment_results.iter().map(outcome).collect();
    let second: Vec<_> = second.experiment_results.iter().map(outcome).collect();

    assert_eq!(first, second);
    assert!(first.iter().any(|run| run.2 != first[0].2));
}

#[test]
fn a_run_replays_on_its_own_from_its_seed() {
    let runner = run("experiment_runner_replays_a_run", 3);

    for result in &runner.experiment_results {
        let mut evolution = builder(0).with_seed(result.seed).build().unwrap();
        evolution.run();

        assert_eq!(evolution.metrics.best_fitnesses, result.best_fitnesses);
        assert_eq!(
            evolution.best_ever().get_chromosome(),
            result.best_found.get_chromosome()
        );
    }
}

#[test]
fn running_twice_counts_each_run_time_once() {
    let mut runner = run("experiment_runner_runs_twice", 2);
    let first_batch = runner.total_time();
    assert!(first_batch > Duration::ZERO);

    runner.run().unwrap();

    let results = &runner.experiment_results;
    assert_eq!(results.len(), 2 * RUNS as usize);
    let second_batch: Duration = results[RUNS as usize..].iter().map(|r| r.time).sum();
    assert_eq!(runner.total_time(), first_batch + second_batch);
    assert_eq!(
        runner.average_time().unwrap(),
        Duration::from_nanos((runner.total_time().as_nanos() / (2 * RUNS as u128)) as u64)
    );
}

#[test]
fn aggregates_average_over_every_run() {
    let empty = ExperimentRunner::new("empty".to_string(), RUNS, builder(5));
    assert_eq!(empty.average_time(), None);
    assert_eq!(empty.average_best_fitness(), None);
    assert_eq!(empty.average_generations(), None);
    empty.experiment_digest();

    let runner = run("experiment_runner_aggregates", 5);
    let results = &runner.experiment_results;
    let runs = results.len() as f64;

    let best = results
        .iter()
        .map(|r| r.best_found.get_fitness())
        .sum::<f64>()
        / runs;
    let generations = results.iter().map(|r| r.iterations as f64).sum::<f64>() / runs;
    assert_eq!(runner.average_best_fitness(), Some(best));
    assert_eq!(runner.average_generations(), Some(generations));
    assert_eq!(runner.average_generations(), Some(GENERATIONS as f64 + 1.0));
}

/// Best at the origin, so a strong mutation easily throws the best individual away.
#[derive(Clone)]
struct NearOrigin;

impl Fitness<Real> for NearOrigin {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        -individual
            .get_chromosome()
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
    }
}

#[test]
fn best_found_is_the_best_ever_without_elitism() {
    let builder = EvolutionBuilder::new(10, 3, GeneCod::Real, (-1.0, 1.0))
        .with_fitness(NearOrigin)
        .with_selection(TournamentSelection::default())
        .with_crossover(DoNothingCrossover)
        .with_mutation(GaussianMutation::new(1.0, 1.0))
        .with_elitism(0)
        .with_seed(1)
        .with_stop_condition(|report| report.generation >= 50);
    let mut runner = ExperimentRunner::new("no_elitism".to_string(), RUNS, builder);
    runner.run().unwrap();

    let mut lost = false;
    for result in &runner.experiment_results {
        let bests = &result.best_fitnesses;
        let best_seen = bests.iter().copied().fold(f64::MIN, f64::max);
        lost |= *bests.last().unwrap() < best_seen;

        assert_eq!(result.best_found.get_fitness(), best_seen);
        assert_eq!(NearOrigin.calculate_fitness(&result.best_found), best_seen);
    }
    assert!(lost, "no run lost its best, the test proves nothing");
}

#[test]
fn an_invalid_builder_returns_the_error_without_running() {
    let builder = EvolutionBuilder::new(20, 16, GeneCod::Bin, ())
        .with_selection(TournamentSelection::default())
        .with_crossover(NPointsCrossover::default())
        .with_mutation(BitFlipMutation::default())
        .with_stop_condition(|report| report.generation >= GENERATIONS);
    let mut runner = ExperimentRunner::new("invalid".to_string(), RUNS, builder);

    assert_eq!(runner.run(), Err(BuildError::MissingFitness));
    assert!(runner.experiment_results.is_empty());
}

#[test]
fn writes_no_file_unless_asked() {
    let name = "experiment_runner_writes_no_file";
    run(name, 9);

    // Where versions before 0.5 saved the chart with the `plot` feature.
    assert!(!std::path::Path::new("results").join(name).exists());
}

#[cfg(feature = "plot")]
#[test]
fn saves_the_chart_where_asked() {
    let runner = run("experiment_runner_saves_the_chart", 11);
    let dir = std::env::temp_dir().join(format!(
        "evolutionary_experiment_runner_chart_{}",
        std::process::id()
    ));
    let path = dir.join("nested").join("chart.png");

    runner.save_chart(&path).unwrap();

    assert!(path.exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(feature = "plot")]
#[test]
fn saving_a_chart_without_results_is_an_error() {
    let runner = ExperimentRunner::new("empty".to_string(), RUNS, builder(13));
    let path = std::env::temp_dir().join("evolutionary_experiment_runner_empty_chart.png");

    assert!(runner.save_chart(&path).is_err());
    assert!(!path.exists());
}
