//! `Real` at the scale of neuroevolution: a few hundred individuals of a few hundred genes, with
//! Gaussian mutation, more than one elitist and tournaments of more than two.

use evolutionary::prelude::*;

const POPULATION: u32 = 300;
const GENES: u32 = 300;

/// Distance to a fixed target in `(-1, 1)`, like weights a network should learn.
#[derive(Clone)]
struct Target(Vec<f64>);

impl Target {
    fn new() -> Self {
        let mut rng = Rngs::new(99).rng(0);
        Self((0..GENES).map(|_| rng.random_range(-1.0..1.0)).collect())
    }
}

impl Fitness<Real> for Target {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        let squared: f64 = individual
            .get_chromosome()
            .iter()
            .zip(&self.0)
            .map(|(gene, target)| (gene - target).powi(2))
            .sum();
        -squared / GENES as f64
    }
}

fn builder(elitism: u32) -> EvolutionBuilder<Real> {
    EvolutionBuilder::new(POPULATION, GENES, GeneCod::Real, (-1.0, 1.0))
        .with_fitness(Target::new())
        .with_selection(TournamentSelection::new(3, 0.9))
        .with_crossover(UniformCrossover::default())
        .with_mutation(GaussianMutation::new(1.0 / GENES as f64, 0.1))
        .with_elitism(elitism)
        .with_seed(2026)
}

fn ranked(population: &[Real]) -> Vec<&Real> {
    let mut ranked: Vec<&Real> = population.iter().collect();
    ranked.sort_by(|a, b| b.get_fitness().total_cmp(&a.get_fitness()));
    ranked
}

#[test]
fn converges_towards_the_target() {
    let mut evolution = builder(5)
        .with_stop_condition(|report| report.generation >= 60)
        .build()
        .unwrap();

    evolution.run();

    let first = evolution.metrics.best_fitnesses[0];
    let last = evolution.current_best_fitness();
    // The mean squared error of the best individual falls at least five times (about 13 times
    // with this seed).
    assert!(last > first / 5.0, "from {first} to {last}");
    assert_eq!(evolution.current_population().len(), POPULATION as usize);
    assert!(evolution
        .current_population()
        .iter()
        .all(|individual| individual.get_chromosome().len() == GENES as usize));
}

#[test]
fn every_elitist_survives_to_the_next_generation() {
    const ELITISM: usize = 8;
    let mut evolution = builder(ELITISM as u32)
        .with_stop_condition(|report| report.generation >= 20)
        .build()
        .unwrap();
    evolution.start();

    for _ in 0..20 {
        let before = evolution.current_population();
        let elitists: Vec<&Real> = ranked(&before)[..ELITISM].to_vec();

        evolution.next();

        let after = evolution.current_population();
        for elitist in &elitists {
            assert!(
                after
                    .iter()
                    .any(|individual| individual.get_chromosome() == elitist.get_chromosome()),
                "an elitist was lost"
            );
        }
        // So the best fitnesses can only improve, rank by rank.
        let after_ranked = ranked(&after);
        for rank in 0..ELITISM {
            assert!(after_ranked[rank].get_fitness() >= elitists[rank].get_fitness());
        }
    }
}

fn scored_population() -> Vec<Real> {
    let target = Target::new();
    let rngs = Rngs::new(3);
    (0..POPULATION as usize)
        .map(|index| {
            let mut individual =
                Real::generate_member(GENES, &(-1.0, 1.0), &mut rngs.rng(index));
            individual.set_fitness(target.calculate_fitness(&individual));
            individual
        })
        .collect()
}

fn mean_fitness(population: &[Real]) -> f64 {
    population.iter().map(Individual::get_fitness).sum::<f64>() / population.len() as f64
}

#[test]
fn bigger_tournaments_select_harder() {
    let population = scored_population();
    let rngs = Rngs::new(4);

    let pool = |k| TournamentSelection::new(k, 1.0).get_mating_pool(&population, &rngs);
    let (two, five, ten) = (pool(2), pool(5), pool(10));

    assert_eq!(five.len(), population.len());
    assert!(mean_fitness(&population) < mean_fitness(&two));
    assert!(mean_fitness(&two) < mean_fitness(&five));
    assert!(mean_fitness(&five) < mean_fitness(&ten));
}

#[test]
fn the_k_minus_one_worst_never_win_a_deterministic_tournament() {
    const K: usize = 5;
    let population = scored_population();
    let worst: Vec<f64> = ranked(&population)
        .iter()
        .rev()
        .take(K - 1)
        .map(|individual| individual.get_fitness())
        .collect();

    let pool = TournamentSelection::new(K, 1.0).get_mating_pool(&population, &Rngs::new(5));

    // Every tournament has K distinct contestants, so one of them beats these.
    assert!(pool
        .iter()
        .all(|individual| !worst.contains(&individual.get_fitness())));
}
