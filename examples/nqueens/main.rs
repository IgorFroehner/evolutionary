use std::time::Instant;

use evolutionary_computing::{
    crossover::PMXCrossover,
    evolution_builder::EvolutionBuilder,
    fitness::Fitness,
    io::{read_config, Config},
    mutation::PermMutation,
    population::{GeneCod, IntPerm},
    selection::RouletteSelection,
};

#[derive(Clone)]
struct NQueensFitness {
    c_max: f64,
}

fn count_colisions(individual: &IntPerm) -> f64 {
    let mut cont = 0.0;
    let genes = &individual.0;
    for i in 0..genes.len() {
        for j in 0..genes.len() {
            if i == j {
                continue;
            }
            let dist = i.abs_diff(j);
            let diff = genes[i].abs_diff(genes[j]);

            if diff == dist as u64 {
                cont += 1.0;
            }
        }
    }

    cont
}

impl Fitness<IntPerm> for NQueensFitness {
    fn calculate_fitness(&self, individual: &IntPerm) -> f64 {
        let cont = count_colisions(individual);

        if cont < self.c_max {
            self.c_max - cont
        } else {
            0.0
        }
    }
}

fn main() {
    let file_name = "examples/nqueens/Config.toml";

    let Config {
        population_size,
        dimension,
        gene_cod,
        ..
    } = read_config(file_name).expect("Failed to read config file");

    if let GeneCod::IntPerm = gene_cod {
        let max_colisions = dimension as f64 * (dimension as f64 - 1.0);
        let fitness = NQueensFitness {
            c_max: max_colisions,
        };

        let selection: RouletteSelection = RouletteSelection::default();

        let mut evolution = EvolutionBuilder::new(population_size, dimension, gene_cod, ())
            .set_fitness(fitness)
            .set_selection(selection)
            .set_crossover(PMXCrossover::default())
            .set_mutation(PermMutation::default())
            .set_stop_condition(move |best_fitness, _| best_fitness == max_colisions)
            .set_title("NQueens".to_string())
            .build()
            .unwrap();

        let start_time = Instant::now();

        evolution.run();

        let best_solution = evolution.current_best();
        println!("Best found: {:?}", best_solution);
        println!("Colisions number: {}", count_colisions(&best_solution));
        println!("Time elapsed: {}", start_time.elapsed().as_secs_f64());

        evolution.plot_chart().unwrap();
    }
}
