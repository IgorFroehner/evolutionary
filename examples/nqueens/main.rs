use evolutionary::prelude::*;

#[derive(Clone)]
struct NQueensFitness {
    c_max: f64,
}

fn count_collisions(individual: &IntPerm) -> f64 {
    let mut cont = 0.0;
    let genes = &individual.chromosome;
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
        let cont = count_collisions(individual);

        if cont < self.c_max {
            self.c_max - cont
        } else {
            0.0
        }
    }
}

fn main() {
    let file_name = "examples/nqueens/Config.toml";

    let config = read_config(file_name).expect("Failed to read config file");

    if let GeneCod::IntPerm = config.gene_cod {
        let crossover = PMXCrossover {
            crossover_rate: 0.8,
        };

        let mutation = SwapMutation::new(0.02);

        let max_colisions = config.dimension as f64 * (config.dimension as f64 - 1.0);

        let fitness = NQueensFitness {
            c_max: max_colisions,
        };

        let evolution_builder = EvolutionBuilder::from_config(config.into())
            .with_fitness(fitness)
            .with_selection(TournamentSelection::default())
            .with_crossover(crossover.clone())
            .with_mutation(mutation.clone())
            .with_elitism(1)
            .with_stop_condition(move |best_fitness, _, _| best_fitness == max_colisions)
            .with_title("NQueens".to_string());

        let mut evolution = evolution_builder.build().unwrap();

        evolution.run();

        evolution
            .plot_chart("NQueens.png", "NQueens Problem")
            .unwrap();

        println!("Best individual: {:?}", evolution.current_best());
        println!("Best fitness: {}", evolution.current_best_fitness());
        evolution.time_digest();
    }
}
