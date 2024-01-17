use evolutionary::prelude::*;

// It needs to implement clone because it will be used in the EvolutionBuilder.
#[derive(Clone)]
struct PartyFitness;

impl Fitness<Bin> for PartyFitness {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        let mut sum = 0.;

        for i in 0..individual.get_chromosome().len() {
            if individual.get_gene(i) {
                sum += 1.;
            }
        }

        sum
    }
}

fn main() {
    let max_guests = 5;

    // Set an arbitrary number of individuals: 30
    // Set the size of the chromosome: max_guests to meet our constraints
    // And set this evolution to use the Binary encoding
    let mut evolution = EvolutionBuilder::new(30, max_guests, GeneCod::Bin, ())
        // Set the fitness function
        .with_fitness(PartyFitness)
        // Use the Tournament Selection
        .with_selection(TournamentSelection::default())
        // Use the Binary Crossover
        .with_crossover(NPointsCrossover::default())
        // Use the Binary Mutation with a mutation rate of 0.05 (which is the default)
        .with_mutation(BitFlipMutation { mutation_rate: 0.05 })
        // Set the stop condition based on the number of iterations
        .with_stop_condition(move |_, iterations, _| iterations >= 1000)
        .build().unwrap();

    // The run method of the Evolution will run the evolution until the stop condition is met
    evolution.run();

    // After the evolution is done, we can get the best individual and its fitness:
    let best = evolution.current_best();

    println!("Best individual: {:?}", best);
    println!("Best fitness: {}", evolution.current_best_fitness());

    // Prints the best and average fitness over the generations
    evolution.plot_chart("party_fitness.png", "Party Fitness").unwrap();

    // Gives an output of the time spent digested by each evolution step
    evolution.time_digest();
}
