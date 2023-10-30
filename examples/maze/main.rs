use colored::Colorize;
use evolutionary::prelude::*;
use maze::MAZE;
use maze_fitness::MazeFitness;

mod maze;
mod maze_fitness;

#[derive(Clone)]
pub struct MazeCoding;

impl Coding<Real> for MazeCoding {
    type Output = ();

    fn decode(&self, _individual: &Real) -> Self::Output {
        ()
    }
}

fn main() {
    let mut maze = MAZE
        .clone()
        .into_iter()
        .map(|row| row.into_iter().map(|cell| cell as i32).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let mut start = (0, 0);
    let mut end = (0, 0);
    for i in 0..maze.len() {
        for j in 0..maze[0].len() {
            if maze[i][j] == 2 {
                start = (i, j);
            }
            if maze[i][j] == 3 {
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

    let mut evolution_builder = EvolutionBuilder::new(50, 100, GeneCod::Real, (0.0, 1.0))
        .with_fitness(MazeFitness {
            max_dist: max_score,
            start,
            end,
            maze: maze.clone(),
        })
        .with_selection(RouletteSelection::default())
        .with_crossover(crossover)
        .with_mutation(SubstituteMutation::default())
        .with_title("Maze".to_string())
        .with_stop_condition(move |best_fitness, _, _|  best_fitness == max_score - 1.0)
        .with_coding(MazeCoding);

    let mut experiment = ExperimentRunner::new("Maze 1".to_string(), 100, evolution_builder);

    experiment.run();

    experiment.experiment_digest();

    // evolution.plot_chart("maze.png", "Maze").unwrap();

    // let best = evolution.current_best();

    // let path = calculate_path(best.get_chromosome(), &maze, start);

    // for i in path {
    //     maze[i.0][i.1] = 4;
    // }

    // for i in 0..maze.len() {
    //     for j in 0..maze[0].len() {
    //         if maze[i][j] == 1 {
    //             print!("{} ", format!("{}", maze[i][j]).black());
    //         } else if maze[i][j] == 0 {
    //             print!("{} ", maze[i][j]);
    //         } else {
    //             print!("{} ", format!("{}", maze[i][j]).red());
    //         }
    //     }
    //     println!();
    // }
}
