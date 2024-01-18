mod datasets;

use datasets::*;
use evolutionary::prelude::*;

#[derive(Clone)]
struct TSPFitness {
    matrix: Vec<Vec<f64>>,
    worst: f64,
}

fn calculate_distance(path: &Vec<i64>, matrix: &Vec<Vec<f64>>) -> f64 {
    let mut distance = 0.;

    for i in 0..path.len() - 1 {
        distance += matrix[path[i] as usize - 1][path[i + 1] as usize - 1];
    }
    distance += matrix[path[path.len() - 1] as usize - 1][path[0] as usize - 1];

    distance
}

impl Fitness<IntPerm> for TSPFitness {
    fn calculate_fitness(&self, individual: &IntPerm) -> f64 {
        let distance = calculate_distance(individual.get_chromosome(), &self.matrix);

        self.worst - distance
    }
}

fn main() {
    let dataset = load_matrix("p01".to_string());

    let max_dist = dataset
        .iter()
        .map(|x| x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let size = dataset.len();

    let min_dist_possible = 291.0;
    let worst = max_dist * size as f64;
    let best_fitness_possible = worst - min_dist_possible;

    let fitness = TSPFitness {
        matrix: dataset.clone(),
        worst,
    };

    println!("Best possible: {}", best_fitness_possible);

    let mut evolution = EvolutionBuilder::new(30, size as u32, GeneCod::IntPerm, ())
        .with_selection(TournamentSelection::default())
        .with_fitness(fitness)
        .with_crossover(OXCrossover::default())
        .with_mutation(SwapMutation::default())
        .with_title("TSP".to_string())
        .with_stop_condition(move |best_fitness, _, _| best_fitness >= best_fitness_possible)
        .with_elitism(3)
        .build()
        .unwrap();

    evolution.run();

    let best = evolution.current_best();
    println!("Best found: {:?}", best);
    println!(
        "Distance: {}",
        calculate_distance(best.get_chromosome(), &dataset)
    );
    evolution.time_digest();

    evolution.plot_chart("tsp.png", "TSP").unwrap();
}
