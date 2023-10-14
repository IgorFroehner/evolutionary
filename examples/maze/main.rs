use evolutionary::prelude::*;
use maze::MAZE;

mod maze;

#[derive(Clone)]
pub struct MazeFitness {
    max_score: f64,
    n: usize,
    m: usize,
    pub board: Vec<f64>,
}

impl Fitness<Bin> for MazeFitness {
    fn calculate_fitness(&self, _individual: &Bin) -> f64 {
        0.0
    }
}

#[derive(Clone)]
pub struct MazeCoding;

impl Coding<Bin> for MazeCoding {
    type Output = ();

    fn decode(&self, _individual: &Bin) -> Self::Output {
        ()
    }
}

fn main() {
    let maze = MAZE;

    let n = maze.len();
    let m = maze[0].len();

    let max_score = (n * m) as f64;

    
}