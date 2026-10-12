use evolutionary::prelude::*;
use maze::read_matrix_from_file;
use maze_fitness::MazeFitness;

mod maze;
mod maze_fitness;

fn main() {
    let maze = read_matrix_from_file("examples/maze/maze0.in");

    let mut start = (0, 0);
    let mut end = (0, 0);
    for (i, row) in maze.iter().enumerate() {
        for (j, &cell) in row.iter().enumerate() {
            if cell == 2 {
                start = (i, j);
            }
            if cell == 3 {
                end = (i, j);
            }
        }
    }

    let n = maze.len();
    let m = maze[0].len();

    let max_score = (n + m) as f64;

    let crossover = UniformCrossover {
        crossover_rate: 0.8,
        toss_probability: 0.3,
    };

    let evolution_builder = EvolutionBuilder::new(50, 100, GeneCod::Real, (0.0, 1.0))
        .with_fitness(MazeFitness {
            max_dist: max_score,
            start,
            end,
            maze: maze.clone(),
        })
        .with_selection(RouletteSelection)
        .with_crossover(crossover)
        .with_mutation(RandomResettingMutation::default())
        .with_title("Maze".to_string())
        .with_stop_condition(move |report| report.best_fitness == max_score);
    // .with_stop_condition(|report| report.generation >= 10_000);

    let mut evolution = evolution_builder.build().unwrap();

    evolution.run();

    evolution.time_digest();

    let _experiment =
        ExperimentRunner::new("Maze  - Caminho Minimo".to_string(), 30, evolution_builder);

    // experiment.run().expect("invalid evolution parameters");

    // experiment.experiment_digest();

    // let best_founds = experiment
    //     .experiment_results
    //     .iter()
    //     .map(|result| result.best_found.clone())
    //     .collect::<Vec<Real>>();
    //
    // let mut paths_size = vec![];
    // for best_found in &best_founds {
    //     let path = MazeFitness::calculate_path(&best_found.chromosome, &maze, start);
    //
    //     if *path.last().unwrap() != end {
    //         continue;
    //     }
    //     paths_size.push(path.len());
    // }
    //
    // let mean = paths_size.iter().sum::<usize>() as f64 / paths_size.len() as f64;
    // let variance = paths_size
    //     .iter()
    //     .map(|x| (*x as f64 - mean).powi(2))
    //     .sum::<f64>()
    //     / paths_size.len() as f64;
    // let std_dev = variance.sqrt();

    // println!("Not found paths: {}", best_founds.len() - paths_size.len());
    // println!("Mean: {}", mean);
    // println!("Variance: {}", std_dev);
}
