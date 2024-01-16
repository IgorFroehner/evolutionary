use evolutionary::prelude::*;

fn f(individual: &Bin) -> f64 {
    let mut sum = 0.;

    for i in 0..individual.get_chromosome().len() {
        if individual.get_gene(i) {
            sum += 1.;
        }
    }

    sum
}

#[derive(Clone)]
pub struct MaxFitness;

impl Fitness<Bin> for MaxFitness {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        f(individual)
    }
}

fn main() {
    let mut evolution = EvolutionBuilder::new(30, 10, GeneCod::Bin, ())
        .with_fitness(MaxFitness)
        .with_selection(TournamentSelection::default())
        .with_crossover(UniformCrossover::default())
        .with_mutation(BitSwapMutation::default())
        .with_title("Max".to_string())
        .with_stop_condition(move |_, iterations, _| iterations >= 1000)
        .build()
        .unwrap();

    evolution.run();

    println!("Best individual: {:?}", evolution.current_best());
    println!("Best fitness: {}", evolution.current_best_fitness());

    // evolution.population_digest();
    evolution.time_digest();
}
