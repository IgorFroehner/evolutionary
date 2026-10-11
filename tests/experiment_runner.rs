//! `ExperimentRunner` runs one builder many times in parallel. A seeded experiment must replay
//! identically, while its runs still differ from each other.

use evolutionary::experiment_runner::ExperimentResult;
use evolutionary::prelude::*;

const RUNS: u32 = 4;
const GENERATIONS: u32 = 10;

#[derive(Clone)]
struct CountOnes;

impl Fitness<Bin> for CountOnes {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        individual.get_chromosome().iter().filter(|&&gene| gene).count() as f64
    }
}

/// Runs a seeded experiment called `name`. With `plot` it also saves a chart to
/// `results/<name>/`, which is checked and then removed.
fn run(name: &str, seed: u64) -> ExperimentRunner<Bin> {
    let builder = EvolutionBuilder::new(20, 16, GeneCod::Bin, ())
        .with_fitness(CountOnes)
        .with_selection(TournamentSelection::default())
        .with_crossover(NPointsCrossover::default())
        .with_mutation(BitFlipMutation::default())
        .with_stop_condition(|report| report.generation >= GENERATIONS)
        .with_seed(seed);

    let mut runner = ExperimentRunner::new(name.to_string(), RUNS, builder);
    runner.run();

    #[cfg(feature = "plot")]
    {
        let dir = format!("results/{name}");
        assert!(std::path::Path::new(&format!("{dir}/{name}.png")).exists());
        std::fs::remove_dir_all(&dir).unwrap();
        // Only goes when no other results are left in it.
        let _ = std::fs::remove_dir("results");
    }

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
