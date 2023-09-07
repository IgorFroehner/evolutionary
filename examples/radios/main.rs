use evolutionary_computing::{
    crossover::BinCrossover,
    evolution_builder::EvolutionBuilder,
    io::{read_config, Config},
    mutation::BinMutation,
    population::GeneCod,
    selection::RouletteSelection,
};
use radio_fitness::RadioFitness;

use crate::radio_fitness::{get_lx, get_st};

mod radio_fitness;

const L: f64 = 5.;

const RANGE_ST: (f64, f64) = (0., 24.);
const RANGE_LX: (f64, f64) = (0., 16.);
const C: f64 = -1.;

fn main() {
    let file_name = "examples/radios/Config.toml";

    let Config {
        population_size,
        dimension,
        gene_cod,
        runs,
        ..
    } = read_config(file_name).expect("Failed to read config file");

    if let GeneCod::Bin = gene_cod {
        let fitness = RadioFitness;

        let crossover = BinCrossover {
            crossover_rate: 0.8,
            n_points: 1,
        };

        let mutation = BinMutation {
            mutation_rate: 0.05,
        };

        let mut evolution = EvolutionBuilder::new(population_size, dimension, gene_cod, ())
            .set_selection(RouletteSelection::default())
            .set_fitness(fitness)
            .set_crossover(crossover)
            .set_mutation(mutation)
            .set_title("Radios".to_string())
            .set_stop_condition(move |_, iterations| iterations > runs as u32)
            .build().unwrap();
        
        evolution.run();

        evolution.plot_chart().unwrap();

        let best_found = evolution.current_best();
        println!("Best found: {:?}", best_found);
        println!("Best found fitness: {}", evolution.current_best_fitness());
        println!("ST: {}", get_st(&best_found));
        println!("LX: {}", get_lx(&best_found));
    }
}
