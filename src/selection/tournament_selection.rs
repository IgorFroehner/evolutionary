use rand::{seq::SliceRandom, thread_rng, Rng};

use crate::fitness::fitness_cmp;
use crate::population::Individual;

use super::Selection;

use rayon::prelude::*;

/// # Tournament Selection
///
/// For each slot of the mating pool it draws `k` distinct individuals at random and picks the
/// fittest of them with probability `kp`, or else the second fittest. A bigger `k` puts more
/// pressure towards the best individuals.
#[derive(Clone, Debug)]
pub struct TournamentSelection {
    k: usize,
    kp: f64,
}

impl TournamentSelection {
    /// Creates a tournament of `k` individuals where the fittest wins with probability `kp`.
    ///
    /// # Panics
    ///
    /// If `k` is zero or `kp` is not within `[0, 1]`.
    pub fn new(k: usize, kp: f64) -> Self {
        assert!(k > 0, "the tournament size `k` must be at least 1");
        assert!(
            (0.0..=1.0).contains(&kp),
            "the winning probability `kp` must be within [0, 1], got {kp}"
        );

        Self { k, kp }
    }

    /// Size of each tournament.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Probability of the fittest contestant winning the tournament.
    pub fn kp(&self) -> f64 {
        self.kp
    }
}

impl Default for TournamentSelection {
    /// A deterministic binary tournament: `k = 2` and `kp = 1.0`.
    fn default() -> Self {
        Self::new(2, 1.0)
    }
}

impl<T: Individual> Selection<T> for TournamentSelection {
    fn get_mating_pool(&self, initial_population: &Vec<T>) -> Vec<T> {
        initial_population
            .par_iter()
            .map_init(thread_rng, |rng, _| {
                let mut tournament: Vec<&T> =
                    initial_population.choose_multiple(rng, self.k).collect();

                tournament.sort_unstable_by(|a, b| fitness_cmp(b.get_fitness(), a.get_fitness()));

                let winner = if tournament.len() > 1 && !rng.gen_bool(self.kp) {
                    tournament[1]
                } else {
                    tournament[0]
                };

                winner.clone()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::population::Real;

    fn population(fitnesses: &[f64]) -> Vec<Real> {
        fitnesses
            .iter()
            .map(|&fitness| {
                let mut individual = Real::new(vec![fitness], (0.0, 1.0));
                individual.set_fitness(fitness);
                individual
            })
            .collect()
    }

    #[test]
    fn whole_population_tournament_always_picks_the_best() {
        let population = population(&[1.0, 5.0, 3.0, 2.0]);

        let pool = TournamentSelection::new(4, 1.0).get_mating_pool(&population);

        assert_eq!(pool.len(), population.len());
        assert!(pool
            .iter()
            .all(|individual| individual.get_fitness() == 5.0));
    }

    #[test]
    fn single_contestant_tournament_with_any_kp() {
        let population = population(&[1.0, 2.0, 3.0]);

        let pool = TournamentSelection::new(1, 0.0).get_mating_pool(&population);

        assert_eq!(pool.len(), population.len());
    }

    #[test]
    fn tournament_bigger_than_population() {
        let population = population(&[1.0, 2.0]);

        let pool = TournamentSelection::new(10, 1.0).get_mating_pool(&population);

        assert!(pool
            .iter()
            .all(|individual| individual.get_fitness() == 2.0));
    }

    #[test]
    fn nan_fitness_never_wins() {
        let population = population(&[f64::NAN, 1.0]);

        let pool = TournamentSelection::new(2, 1.0).get_mating_pool(&population);

        assert!(pool
            .iter()
            .all(|individual| individual.get_fitness() == 1.0));
    }

    #[test]
    #[should_panic(expected = "at least 1")]
    fn zero_sized_tournament_panics() {
        TournamentSelection::new(0, 1.0);
    }

    #[test]
    #[should_panic(expected = "within [0, 1]")]
    fn out_of_range_kp_panics() {
        TournamentSelection::new(2, 1.5);
    }
}
