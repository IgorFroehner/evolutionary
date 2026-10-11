use evolutionary::prelude::*;

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

const SIZE: u32 = 20;
const DIMENSION: u32 = 12;

fn builder() -> EvolutionBuilder<Bin> {
    EvolutionBuilder::new(SIZE, DIMENSION, GeneCod::Bin, ())
        .with_fitness(CountOnes)
        .with_selection(TournamentSelection::default())
        .with_crossover(NPointsCrossover::default())
        .with_mutation(BitFlipMutation::default())
        .with_stop_condition(|report| report.generation >= 30)
        .with_seed(5)
}

fn chromosomes(population: &[Bin]) -> Vec<Vec<bool>> {
    population
        .iter()
        .map(|individual| individual.get_chromosome().clone())
        .collect()
}

#[test]
fn a_few_individuals_seed_the_population_and_random_ones_fill_it() {
    let mut champion = Bin::new(vec![true; DIMENSION as usize]);
    // A stale fitness from a saved run must not survive.
    champion.set_fitness(1000.0);

    let mut evolution = builder()
        .with_initial_population(vec![champion])
        .build()
        .unwrap();
    evolution.start();

    assert_eq!(evolution.current_population().len(), SIZE as usize);
    assert_eq!(evolution.current_best_fitness(), DIMENSION as f64);
    assert_eq!(
        *evolution.current_best().get_chromosome(),
        vec![true; DIMENSION as usize]
    );
}

#[test]
fn a_saved_population_resumes_the_run() {
    let mut first = builder().build().unwrap();
    first.run();
    let saved = first.current_population();

    let mut resumed = builder()
        .with_initial_population(saved.clone())
        .build()
        .unwrap();
    resumed.start();

    assert_eq!(
        chromosomes(&resumed.current_population()),
        chromosomes(&saved)
    );
    assert_eq!(resumed.current_best_fitness(), first.current_best_fitness());
}

#[test]
fn an_empty_initial_population_is_the_same_as_none() {
    let mut plain = builder().build().unwrap();
    plain.run();

    let mut empty = builder()
        .with_initial_population(Vec::new())
        .build()
        .unwrap();
    empty.run();

    assert_eq!(
        chromosomes(&plain.current_population()),
        chromosomes(&empty.current_population())
    );
}

#[test]
fn every_start_begins_from_the_initial_population() {
    let champion = Bin::new(vec![true; DIMENSION as usize]);
    let mut evolution = builder()
        .with_initial_population(vec![champion])
        .with_elitism(0)
        .build()
        .unwrap();

    evolution.run();
    evolution.start();

    assert_eq!(evolution.current_best_fitness(), DIMENSION as f64);
}

#[test]
fn too_many_initial_individuals_is_an_error() {
    let population = vec![Bin::new(vec![false; DIMENSION as usize]); SIZE as usize + 1];

    let error = builder()
        .with_initial_population(population)
        .build()
        .err()
        .unwrap();

    assert_eq!(
        error,
        BuildError::InitialPopulationTooLarge {
            given: SIZE as usize + 1,
            population_size: SIZE
        }
    );
}

#[test]
fn an_initial_individual_of_the_wrong_dimension_is_an_error() {
    let population = vec![
        Bin::new(vec![false; DIMENSION as usize]),
        Bin::new(vec![false; 3]),
    ];

    let error = builder()
        .with_initial_population(population)
        .build()
        .err()
        .unwrap();

    assert_eq!(
        error,
        BuildError::InitialIndividualDimension {
            index: 1,
            genes: 3,
            dimension: DIMENSION
        }
    );
    assert!(error.to_string().contains("individual 1 has 3 genes"));
}

#[test]
fn a_permutation_can_seed_a_population() {
    let sorted = IntPerm::new((1..=DIMENSION as i64).collect());

    assert_eq!(*sorted.get_chromosome(), (1..=12).collect::<Vec<i64>>());
}

#[test]
#[should_panic(expected = "must be a permutation")]
fn a_chromosome_that_is_not_a_permutation_panics() {
    IntPerm::new(vec![1, 2, 2, 4]);
}
