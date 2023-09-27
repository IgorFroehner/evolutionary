use std::{collections::HashMap, time::Instant};

use crate::utils::plot_chart;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Steps {
    Selection,
    Crossover,
    Mutation,
    Fitness,
}

pub struct Metrics {
    pub best_fitnesses: Vec<f64>,
    pub avg_fitnesses: Vec<f64>,
    pub iterations: u32,
    start_time: Instant,
    end_time: Instant,
    pub step_times: HashMap<Steps, (bool, Instant, u128)>,
}

impl Metrics {
    pub fn new() -> Self {
        let mut step_times = HashMap::new();
        step_times.insert(Steps::Selection, (false, Instant::now(), 0));
        step_times.insert(Steps::Crossover, (false, Instant::now(), 0));
        step_times.insert(Steps::Mutation, (false, Instant::now(), 0));
        step_times.insert(Steps::Fitness, (false, Instant::now(), 0));

        Self {
            best_fitnesses: Vec::new(),
            avg_fitnesses: Vec::new(),
            iterations: 0,
            start_time: Instant::now(),
            end_time: Instant::now(),
            step_times: step_times,
        }
    }

    pub fn record(&mut self, best_fitness: f64, avg_fitness: f64) {
        self.best_fitnesses.push(best_fitness);
        self.avg_fitnesses.push(avg_fitness);
        self.iterations += 1;
    }

    pub fn start_clock(&mut self) {
        self.start_time = Instant::now();
    }

    pub fn end_clock(&mut self) {
        self.end_time = Instant::now();
    }

    pub fn step_start(&mut self, step: Steps) {
        self.step_times.get_mut(&step).map(|a| {
            a.0 = true;
            a.1 = Instant::now()
        });
    }

    pub fn step_end(&mut self, step: Steps) {
        self.step_times.get_mut(&step).map(|a| {
            a.0 = false;
            a.2 += a.1.elapsed().as_millis()
        });
    }

    pub fn get_elapsed_time(&self) -> u128 {
        self.end_time.duration_since(self.start_time).as_millis()
    }

    pub fn plot_chart(&self, path: &String, test_name: &String) -> Result<(), Box<dyn std::error::Error>> {
        plot_chart(&self.best_fitnesses, &self.avg_fitnesses, path, test_name)
    }
}