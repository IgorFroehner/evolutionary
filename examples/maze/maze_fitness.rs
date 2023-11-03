use evolutionary::{prelude::Real, Fitness};

#[derive(Clone)]
pub struct MazeFitness {
    pub max_dist: f64,
    pub start: (usize, usize),
    pub end: (usize, usize),
    pub maze: Vec<Vec<i32>>,
}

impl MazeFitness {
    pub fn calculate_path(
        vec: &Vec<f64>,
        maze: &Vec<Vec<i32>>,
        start: (usize, usize),
    ) -> Vec<(usize, usize)> {
        let mut x = start.0;
        let mut y = start.1;

        let mut vis = vec![vec![false; maze[0].len()]; maze.len()];

        let mut path = vec![start];

        for i in 0..vec.len() {
            let step = vec[i];

            vis[x][y] = true;
            let mut possibilities = vec![];
            if x as i32 + 1 < maze.len() as i32 && maze[x + 1][y] != 0 && !vis[x + 1][y] {
                possibilities.push((x + 1, y));
            }
            if x as i32 - 1 > 0 && maze[x - 1][y] != 0 && !vis[x - 1][y] {
                possibilities.push((x - 1, y));
            }
            if y as i32 - 1 > 0 && maze[x][y - 1] != 0 && !vis[x][y - 1] {
                possibilities.push((x, y - 1));
            }
            if y as i32 + 1 < maze[0].len() as i32 && maze[x][y + 1] != 0 && !vis[x][y + 1] {
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
}

impl Fitness<Real> for MazeFitness {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        let path = Self::calculate_path(&individual.chromosome, &self.maze, self.start);

        let last = path.last().unwrap();
        let (x, y) = last;

        let dist =
            ((*x as i32 - self.end.0 as i32).abs() + (*y as i32 - self.end.1 as i32).abs()) as f64;

        self.max_dist - dist // - (path.len() as f64 / individual.chromosome.len() as f64)
    }
}
