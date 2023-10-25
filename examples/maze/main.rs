use colored::Colorize;
use evolutionary::prelude::*;
use maze::MAZE;
use rayon::vec;

mod maze;

#[derive(Clone)]
pub struct MazeFitness {
    max_dist: f64,
    start: (usize, usize),
    end: (usize, usize),
    pub maze: Vec<Vec<i32>>,
}

fn calculate_path(vec: &Vec<f64>, maze: &Vec<Vec<i32>>, start: (usize, usize)) -> Vec<(usize, usize)> {
    let mut x = start.0;
    let mut y = start.1;

    let mut vis = vec![vec![false; maze[0].len()]; maze.len()];

    let mut path = vec![start];

    for i in 0..vec.len() {
        let step = vec[i];

        vis[x][y] = true;
        let mut possibilities = vec![];
        if x as i32 + 1 < maze.len() as i32 && maze[x + 1][y] == 1 && !vis[x + 1][y] {
            possibilities.push((x + 1, y));
        }
        if x as i32 - 1 > 0 && maze[x - 1][y] == 1 && !vis[x - 1][y] {
            possibilities.push((x - 1, y));
        }
        if y as i32 - 1 > 0 && maze[x][y - 1] == 1 && !vis[x][y - 1] {
            possibilities.push((x, y - 1));
        }
        if y as i32 + 1 < maze[0].len() as i32 && maze[x][y + 1] == 1 && !vis[x][y + 1] {
            possibilities.push((x, y + 1));
        }

        if possibilities.len() == 0 {
            break;
        }

        let frac = 1.0 / possibilities.len() as f64;
        let index = (step / frac).floor() as usize;
        let selected = possibilities.get(index).or(possibilities.first()).unwrap();

        x = selected.0;
        y = selected.1;
        path.push(*selected);
    }

    path
}

impl Fitness<Real> for MazeFitness {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        let path = calculate_path(&individual.chromosome, &self.maze, self.start);

        let last = path.last().unwrap();
        let (x, y) = last;

        let dist = ((*x as i32 - self.end.0 as i32).abs() + (*y as i32 - self.end.1 as i32).abs()) as f64;

        self.max_dist - dist
    }
}

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
        toss_probability: 0.1,
    };

    let mut evolution = EvolutionBuilder::new(50, 100, GeneCod::Real, (0.0, 1.0))
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
        .with_coding(MazeCoding)
        .build()
        .unwrap();

    evolution.run();

    evolution.plot_chart("maze.png", "Maze").unwrap();

    let best = evolution.current_best();

    let path = calculate_path(best.get_chromossome(), &maze, start);

    for i in path {
        maze[i.0][i.1] = 4;
    }

    for i in 0..maze.len() {
        for j in 0..maze[0].len() {
            if maze[i][j] != 4 {
                print!("{} ", format!("{}", maze[i][j]).blue());
            } else {
                print!("{} ", format!("{}", maze[i][j]).red());
            }
        }
        println!();
    }
}
