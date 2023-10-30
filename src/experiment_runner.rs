use std::fs::create_dir_all;
use std::time::{Duration, Instant};

use crate::utils::plot_chart;
use crate::{coding::Coding, evolution_builder::EvolutionBuilder, population::Individual};

use rayon::prelude::*;

struct ExperimentMetrics {
    total_time: u128,
}

pub struct ExperimentRunner<T: Individual, C: Coding<T>> {
    runs: u32,
    name: String,
    evolution_builder: EvolutionBuilder<T, C>,
    experiment_metrics: ExperimentMetrics,
    pub experiment_results: Vec<ExperimentResult<T>>,
}

pub struct ExperimentResult<T> {
    total_time: u128,
    average_fitnesses: Vec<f64>,
    best_fitnesses: Vec<f64>,
    pub iterations: u32,
    pub best_found: T,
}

impl<T: Individual, C: Coding<T>> ExperimentRunner<T, C> {
    pub fn new(name: String, runs: u32, evolution_builder: EvolutionBuilder<T, C>) -> Self {
        Self {
            name,
            runs,
            evolution_builder,
            experiment_metrics: ExperimentMetrics { total_time: 0 },
            experiment_results: Vec::new(),
        }
    }

    pub fn run(&mut self) {
        let path = format!("results/{}", self.name);
        let _ = create_dir_all(path.clone());

        let results: Vec<ExperimentResult<T>> = (0..self.runs)
            .into_par_iter()
            .map(|_| {
                let mut evolution = self.evolution_builder.build().unwrap();

                let start_time = Instant::now();

                evolution.run();

                let total_time = start_time.elapsed().as_nanos();

                // let test_path = &format!("{}/{} run_{}.png", &path, self.name, run);
                // evolution.plot_chart(&test_path, &self.name).unwrap();

                let average_fitnesses = evolution.metrics.avg_fitnesses.clone();
                let best_fitnesses = evolution.metrics.best_fitnesses.clone();
                let best_found = evolution.current_best().clone();

                ExperimentResult {
                    average_fitnesses,
                    best_fitnesses,
                    total_time,
                    iterations: evolution.metrics.iterations,
                    best_found,
                }
            })
            .collect();

        let iterations = results.iter().map(|r| r.iterations).max().unwrap();

        self.experiment_results.extend(results);

        let mut average_fitness_sum_per_generation = vec![0f64; iterations as usize];
        let mut best_fitness_sum_per_generation = vec![0f64; iterations as usize];

        for result in &self.experiment_results {
            for i in 0..(result.iterations) as usize {
                average_fitness_sum_per_generation[i] += result.average_fitnesses[i];
                best_fitness_sum_per_generation[i] += result.best_fitnesses[i];
            }

            for i in result.iterations as usize..iterations as usize {
                average_fitness_sum_per_generation[i] +=
                    result.average_fitnesses[(result.iterations - 1) as usize];
                best_fitness_sum_per_generation[i] +=
                    result.best_fitnesses[(result.iterations - 1) as usize];
            }

            self.experiment_metrics.total_time += result.total_time;
        }

        let average_fitness_per_generation = average_fitness_sum_per_generation
            .iter()
            .map(|a| a / self.runs as f64)
            .collect::<Vec<f64>>();
        let best_fitness_per_generation = best_fitness_sum_per_generation
            .iter()
            .map(|a| a / self.runs as f64)
            .collect::<Vec<f64>>();

        let path = format!("{}/{}.png", &path, self.name);
        plot_chart(
            &best_fitness_per_generation,
            &average_fitness_per_generation,
            &path,
            &self.name,
        )
        .unwrap();

        println!(
            "Average time for {}: {:?}",
            self.name,
            Duration::from_nanos(self.experiment_metrics.total_time as u64 / self.runs as u64)
        );
    }

    pub fn experiment_digest(&self) {
        let avg_max_iterations = self
            .experiment_results
            .iter()
            .map(|result| result.iterations as f64)
            .sum::<f64>()
            / self.experiment_results.len() as f64;
        let avg_score = self
            .experiment_results
            .iter()
            .map(|result| result.best_found.get_fitness())
            .sum::<f64>()
            / self.experiment_results.len() as f64;

        println!("Average max iterations: {}", avg_max_iterations);
        println!("Average score: {}", avg_score);
    }
}
