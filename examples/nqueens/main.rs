use plotters::prelude::Quartiles;
use evolutionary_computing::{
    crossover::PMXCrossover,
    evolution_builder::EvolutionBuilder,
    fitness::Fitness,
    config_read::read_config,
    mutation::PermMutation,
    population::{GeneCod, IntPerm},
    selection::RouletteSelection, coding::Coding,
};
use evolutionary_computing::experiment_runner::ExperimentRunner;
use evolutionary_computing::plot_evolution::Steps::Selection;
use evolutionary_computing::selection::TournamentSelection;
use evolutionary_computing::utils::plot_boxplot;

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

        let mutation = PermMutation {
            mutation_rate: 0.02,
        };

        let mut iterations_number: Vec<Vec<u32>> = Vec::new();
        let mut labels = Vec::new();
        let mut dimension = 128;
        while dimension <= 128 {
            let mut new_config = config.clone();

            new_config.dimension = dimension;
            let max_colisions = new_config.dimension as f64 * (config.dimension as f64 - 1.0);

            let fitness = NQueensFitness {
                c_max: max_colisions,
            };

            let evolution_builder = EvolutionBuilder::from_config(new_config.into())
                .with_fitness(fitness)
                .with_selection(TournamentSelection::default())
                .with_crossover(crossover.clone())
                .with_mutation(mutation.clone())
                .with_coding(NQueensCoding)
                .with_stop_condition(move |best_fitness, _| best_fitness == max_colisions)
                .with_title("NQueens".to_string());

            let label = format!("{}Queens", dimension);
            labels.push(label.clone());
            let mut experiment = ExperimentRunner::new(label, 10, evolution_builder);

            experiment.run();

            iterations_number.push(experiment.experiment_results.iter().map(|r| r.iterations).collect());
            dimension *= 2;
        }

        println!("max iterations: {}", iterations_number.iter().map(|v| v.iter().max().unwrap()).max().unwrap());

        let sum: f32 = iterations_number[0].iter().map(|&v| v as f32).sum();
        let len = iterations_number[0].len() as f32;

        println!("average iterations: {}", sum / len);

        let quartiles = iterations_number.iter().map(|v| Quartiles::new(v)).collect();
        plot_boxplot(&quartiles, &labels).unwrap();
    }
}
