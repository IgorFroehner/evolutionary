use std::fs::create_dir_all;
use std::time::{Duration, Instant};

use crate::{population::Individual, coding::Coding, evolution_builder::EvolutionBuilder};
use crate::utils::plot_chart;

use rayon::prelude::*;

struct ExperimentMetrics {
    total_time: u128,
}

pub struct ExperimentRunner<T: Individual, C: Coding<T>> {
    runs: u32,
    name: String,
    evolution_builder: EvolutionBuilder<T, C>,
    experiment_metrics: ExperimentMetrics,
    pub experiment_results: Vec<ExperimentResult>
}

pub struct ExperimentResult {
    total_time: u128,
    average_fitnesses: Vec<f64>,
    best_fitnesses: Vec<f64>,
    pub iterations: u32,
}

impl<T: Individual, C: Coding<T>> ExperimentRunner<T, C> {
    pub fn new(name: String, runs: u32, evolution_builder: EvolutionBuilder<T, C>) -> Self {
        Self {
            name,
            runs,
            evolution_builder,
            experiment_metrics: ExperimentMetrics {
                total_time: 0,
            },
            experiment_results: Vec::new(),
        }
    }

    pub fn run(&mut self) {
        let path = format!("results/{}", self.name);
        let _ = create_dir_all(path.clone());

        let results: Vec<ExperimentResult> = (0..self.runs).into_par_iter().map(|_| {
            let mut evolution = self.evolution_builder.build().unwrap();

            let start_time = Instant::now();

            evolution.run();

            let total_time = start_time.elapsed().as_nanos();

            // let test_path = &format!("{}/{} run_{}.png", &path, self.name, run);
            // evolution.plot_chart(&test_path, &self.name).unwrap();

            let average_fitnesses = evolution.metrics.avg_fitnesses;
            let best_fitnesses = evolution.metrics.best_fitnesses;

            ExperimentResult {
                average_fitnesses,
                best_fitnesses,
                total_time,
                iterations: evolution.metrics.iterations,
            }
        }).collect();

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
                average_fitness_sum_per_generation[i] += result.average_fitnesses[(result.iterations - 1) as usize];
                best_fitness_sum_per_generation[i] += result.best_fitnesses[(result.iterations - 1) as usize];
            }

            self.experiment_metrics.total_time += result.total_time;
        }

        let average_fitness_per_generation = average_fitness_sum_per_generation.iter().map(|a| a / self.runs as f64).collect::<Vec<f64>>();
        let best_fitness_per_generation = best_fitness_sum_per_generation.iter().map(|a| a / self.runs as f64).collect::<Vec<f64>>();

        let path = format!("{}/{}.png", &path, self.name);
        plot_chart(&best_fitness_per_generation, &average_fitness_per_generation, &path, &self.name).unwrap();

        println!("Average time for {}: {:?}", self.name, Duration::from_nanos(self.experiment_metrics.total_time as u64 / self.runs as u64));
    }
}
