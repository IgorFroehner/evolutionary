//! Crossovers pair the population two by two, so with an odd size the last individual has no
//! partner. It must pass through unchanged instead of panicking.

use evolutionary::prelude::*;

const SIZE: usize = 5;

fn assert_last_untouched<T: Individual, C: Crossover<T>>(crossover: C, population: Vec<T>)
where
    T::Gene: PartialEq + std::fmt::Debug,
{
    let last = population.last().unwrap().get_chromosome().clone();
    let mut population = population;

    crossover.crossover(&mut population, &Rngs::new(0));

    assert_eq!(population.len(), SIZE);
    assert_eq!(*population.last().unwrap().get_chromosome(), last);
}

fn bins() -> Vec<Bin> {
    (0..SIZE)
        .map(|i| Bin::new(vec![i % 2 == 0; 8]))
        .collect()
}

fn ints() -> Vec<Int> {
    (0..SIZE as i64)
        .map(|i| Int::new(vec![i; 8], (0, 10)))
        .collect()
}

fn reals() -> Vec<Real> {
    (0..SIZE)
        .map(|i| Real::new(vec![i as f64; 8], (0.0, 10.0)))
        .collect()
}

fn perms() -> Vec<IntPerm> {
    let rngs = Rngs::new(1);
    (0..SIZE)
        .map(|i| IntPerm::generate_member(8, &(), &mut rngs.rng(i)))
        .collect()
}

fn n_points() -> NPointsCrossover {
    NPointsCrossover {
        crossover_rate: 1.0,
        n_points: 2,
    }
}

fn uniform() -> UniformCrossover {
    UniformCrossover {
        crossover_rate: 1.0,
        toss_probability: 0.5,
    }
}

#[test]
fn bin_crossovers() {
    assert_last_untouched(n_points(), bins());
    assert_last_untouched(uniform(), bins());
}

#[test]
fn int_crossovers() {
    assert_last_untouched(n_points(), ints());
    assert_last_untouched(uniform(), ints());
}

#[test]
fn real_crossovers() {
    assert_last_untouched(n_points(), reals());
    assert_last_untouched(uniform(), reals());
    assert_last_untouched(BlendCrossover::new(1.0, 0.5), reals());
    assert_last_untouched(
        ArithmeticCrossover {
            crossover_rate: 1.0,
            alpha: 0.3,
        },
        reals(),
    );
}

#[test]
fn perm_crossovers() {
    assert_last_untouched(OrderedCrossover { crossover_rate: 1.0 }, perms());
    assert_last_untouched(PartiallyMappedCrossover { crossover_rate: 1.0 }, perms());
    assert_last_untouched(CycleCrossover { crossover_rate: 1.0 }, perms());
}

#[test]
fn odd_population_evolves() {
    let mut evolution = EvolutionBuilder::new(SIZE as u32, 8, GeneCod::Bin, ())
        .with_fitness(CountOnes)
        .with_selection(TournamentSelection::default())
        .with_crossover(n_points())
        .with_mutation(BitFlipMutation::default())
        .with_stop_condition(|report| report.generation >= 10)
        .with_seed(3)
        .build()
        .unwrap();

    evolution.run();

    assert_eq!(evolution.current_population().len(), SIZE);
}

#[derive(Clone)]
struct CountOnes;

impl Fitness<Bin> for CountOnes {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        individual.get_chromosome().iter().filter(|&&gene| gene).count() as f64
    }
}
