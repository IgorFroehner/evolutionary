
use evolutionary::prelude::*;

mod nqueens_fitness;

use nqueens_fitness::{NQueensFitness};

#[derive(Clone)]
struct NQueensCoding;

impl Coding<IntPerm> for NQueensCoding {
    type Output = f64;

    fn decode(&self, individual: &IntPerm) -> Self::Output {
        // count_colisions(individual)
        0.0
    }
}

fn main() {
    let file_name = "examples/nqueens_valued/Config.toml";

    let config = read_config(file_name).expect("Failed to read config file");

    let runs_array = vec![100, 500, 1000, 10000, 100000];

    if let GeneCod::IntPerm = config.gene_cod {
        let crossover = PMXCrossover {
            crossover_rate: 0.8,
        };

        let mutation = PermMutation::new(0.02);
        let dimension = config.dimension;

        let fitness = NQueensFitness::new(dimension as usize);

        let dimension = config.dimension;
        let runs = config.runs;

        let evolution_builder = EvolutionBuilder::from_config(config.clone().into())
            .with_fitness(fitness.clone())
            .with_selection(TournamentSelection::default())
            .with_crossover(crossover.clone())
            .with_mutation(mutation.clone())
            .with_coding(NQueensCoding)
            .with_stop_condition(move |_, _max_iterations, gens_not_improving| gens_not_improving == 20_000)
            .with_title("NQueens".to_string());

        let mut experiment = ExperimentRunner::new(format!("{}Queens", dimension), 30, evolution_builder);

        experiment.run();

        let avg_max_iterations = experiment.experiment_results.iter().map(|result| result.iterations as f64).sum::<f64>() / experiment.experiment_results.len() as f64;
        let avg_score = experiment.experiment_results.iter().filter(|result| fitness.count_collisions(&result.best_found) <= 0.0).map(|result| fitness.score(&result.best_found)).sum::<f64>() / experiment.experiment_results.len() as f64;
        let count_with_collisions = experiment.experiment_results.iter().filter(|result| fitness.count_collisions(&result.best_found) > 0.0).count();

        println!("Average max iterations: {}", avg_max_iterations);
        println!("Average score: {}", avg_score);
        println!("Count with collisions: {}", count_with_collisions);

        // let quartiles = Quartiles::new(&best_scores);
        // plot_boxplot(&vec![quartiles], &vec![format!("{}Queens", dimension)]).unwrap();

        // for result in experiment.experiment_results {
        //     println!("Best individual: {:?}", result.best_found);
        //     println!("Best individual collisions: {}", fitness.count_collisions(&result.best_found));
        //     println!("Best individual score: {}", fitness.score(&result.best_found));
        // }
    }
}
