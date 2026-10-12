use std::time::Duration;
use std::{collections::HashMap, time::Instant};

use crate::fitness::fitness_cmp;
#[cfg(feature = "plot")]
use crate::plotting::plot_chart;

/// The steps of a generation whose time [`Metrics`] measures.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Steps {
    Selection,
    Crossover,
    Mutation,
    Fitness,
    Elitism,
}

impl Steps {
    /// Every step, in the order a generation runs them.
    const ALL: [Steps; 5] = [
        Steps::Selection,
        Steps::Crossover,
        Steps::Mutation,
        Steps::Fitness,
        Steps::Elitism,
    ];
}

/// The time spent in one step, over the whole run.
#[derive(Clone, Copy, Default)]
struct StepTimer {
    /// When the step started, while it runs.
    started: Option<Instant>,
    total: Duration,
}

/// The fitness history and the timings of a run.
pub struct Metrics {
    pub best_fitnesses: Vec<f64>,
    pub avg_fitnesses: Vec<f64>,
    pub iterations: u32,
    /// Generations since the best fitness seen in the run last improved.
    pub gens_without_improvement: u32,
    best_ever_fitness: Option<f64>,
    start_time: Instant,
    end_time: Instant,
    step_times: HashMap<Steps, StepTimer>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        let step_times = Steps::ALL
            .into_iter()
            .map(|step| (step, StepTimer::default()))
            .collect();

        Self {
            best_fitnesses: Vec::new(),
            avg_fitnesses: Vec::new(),
            iterations: 0,
            gens_without_improvement: 0,
            best_ever_fitness: None,
            start_time: Instant::now(),
            end_time: Instant::now(),
            step_times,
        }
    }

    /// The metrics of a run resumed from a [`Checkpoint`](crate::Checkpoint): its fitness
    /// history, as if it had been recorded generation by generation, and fresh timings.
    /// `best_ever_fitness` is the fitness of the run's best individual, which the next
    /// [`record`](Self::record) compares with.
    pub(crate) fn resumed(
        best_fitnesses: Vec<f64>,
        avg_fitnesses: Vec<f64>,
        gens_without_improvement: u32,
        best_ever_fitness: f64,
    ) -> Self {
        // One generation recorded per fitness: `build` checks that the history has one entry
        // per generation of the checkpoint, the initial population included.
        let iterations = best_fitnesses.len() as u32;

        Self {
            best_fitnesses,
            avg_fitnesses,
            iterations,
            gens_without_improvement,
            best_ever_fitness: Some(best_ever_fitness),
            ..Self::new()
        }
    }

    pub fn record(&mut self, best_fitness: f64, avg_fitness: f64) {
        let improved = self
            .best_ever_fitness
            .is_none_or(|best_ever| fitness_cmp(best_fitness, best_ever).is_gt());

        if improved {
            self.best_ever_fitness = Some(best_fitness);
            self.gens_without_improvement = 0;
        } else {
            self.gens_without_improvement += 1;
        }

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

    /// Starts timing `step`, until [`step_end`](Self::step_end).
    ///
    /// # Panics
    ///
    /// If `step` was already started and not ended.
    pub fn step_start(&mut self, step: Steps) {
        let timer = self.step_times.entry(step).or_default();
        assert!(timer.started.is_none(), "Step already started");
        timer.started = Some(Instant::now());
    }

    /// Stops timing `step` and adds the time since [`step_start`](Self::step_start) to it.
    ///
    /// # Panics
    ///
    /// If `step` was not started.
    pub fn step_end(&mut self, step: Steps) {
        let timer = self.step_times.entry(step).or_default();
        let started = timer.started.take().expect("Step not started");
        timer.total += started.elapsed();
    }

    /// The time spent in `step` since the run started, zero until the step first runs. It is
    /// `Some` for every step.
    pub fn step_time(&self, step: Steps) -> Option<Duration> {
        self.step_times.get(&step).map(|timer| timer.total)
    }

    /// The time from the start of the run to the end of its last generation.
    pub fn total_time(&self) -> Duration {
        self.end_time.duration_since(self.start_time)
    }

    /// Every step with the time spent in it, in the order a generation runs them.
    pub(crate) fn step_times(&self) -> impl Iterator<Item = (Steps, Duration)> + '_ {
        Steps::ALL
            .into_iter()
            .map(|step| (step, self.step_time(step).unwrap_or_default()))
    }

    /// Saves a chart of the best and average fitness per generation. Needs the `plot` feature.
    #[cfg(feature = "plot")]
    pub fn plot_chart(
        &self,
        path: &str,
        test_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        plot_chart(&self.best_fitnesses, &self.avg_fitnesses, path, test_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_times_add_up_and_start_at_zero() {
        let mut metrics = Metrics::new();

        for step in Steps::ALL {
            assert_eq!(metrics.step_time(step), Some(Duration::ZERO));
        }

        metrics.step_start(Steps::Mutation);
        std::thread::sleep(Duration::from_millis(2));
        metrics.step_end(Steps::Mutation);
        let first = metrics.step_time(Steps::Mutation).unwrap();
        metrics.step_start(Steps::Mutation);
        metrics.step_end(Steps::Mutation);

        assert!(first >= Duration::from_millis(2));
        assert!(metrics.step_time(Steps::Mutation).unwrap() >= first);
        assert_eq!(metrics.step_time(Steps::Selection), Some(Duration::ZERO));
    }

    #[test]
    #[should_panic(expected = "Step already started")]
    fn starting_a_running_step_panics() {
        let mut metrics = Metrics::new();

        metrics.step_start(Steps::Fitness);
        metrics.step_start(Steps::Fitness);
    }

    #[test]
    #[should_panic(expected = "Step not started")]
    fn ending_a_step_that_did_not_start_panics() {
        Metrics::new().step_end(Steps::Fitness);
    }

    #[test]
    fn generations_without_improvement_counts_from_the_best_ever() {
        let mut metrics = Metrics::new();
        let mut counters = Vec::new();

        for best in [1.0, 3.0, 2.0, 3.0, 4.0, f64::NAN, 4.0] {
            metrics.record(best, 0.0);
            counters.push(metrics.gens_without_improvement);
        }

        // Falling back to 2.0 and climbing back to 3.0 is not an improvement.
        assert_eq!(counters, vec![0, 0, 1, 2, 0, 1, 2]);
    }
}
