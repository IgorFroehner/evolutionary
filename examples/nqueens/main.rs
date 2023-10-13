
use evolutionary_computing::{
    crossover::PMXCrossover,
    evolution_builder::EvolutionBuilder,
    fitness::Fitness,
    config_read::read_config,
    mutation::PermMutation,
    population::{GeneCod, IntPerm},
    coding::Coding,
};
use evolutionary_computing::experiment_runner::ExperimentRunner;
use evolutionary_computing::selection::TournamentSelection;

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

#[derive(Clone)]
struct NQueensCoding;

impl Coding<IntPerm> for NQueensCoding {
    type Output = f64;

    fn decode(&self, individual: &IntPerm) -> Self::Output {
        count_colisions(individual)
    }
}

fn main() {
    let file_name = "examples/nqueens/Config.toml";

    let config = read_config(file_name).expect("Failed to read config file");

    if let GeneCod::IntPerm = config.gene_cod {
        let crossover = PMXCrossover {
            crossover_rate: 1.0,
        };

        let mutation = PermMutation::new(0.02);

        let max_colisions = config.dimension as f64 * (config.dimension as f64 - 1.0);

        let fitness = NQueensFitness {
            c_max: max_colisions,
        };

        let dimension = config.dimension;
        let evolution_builder = EvolutionBuilder::from_config(config.into())
            .with_fitness(fitness)
            .with_selection(TournamentSelection::default())
            .with_crossover(crossover.clone())
            .with_mutation(mutation.clone())
            .with_coding(NQueensCoding)
            .with_stop_condition(|_best_fitness, _, gens_without_improvement| gens_without_improvement >= 1000)
            .with_title("NQueens".to_string());

        let mut experiment = ExperimentRunner::new(format!("{}Queens.png", dimension), 10, evolution_builder);

        experiment.run();
    }
}
