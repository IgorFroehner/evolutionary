use evolutionary_computing::fitness::Fitness;
use evolutionary_computing::population::IntPerm;

#[derive(Clone)]
pub struct NQueensFitness {
    c_max: f64,
    max_score: f64,
    dimension: usize,
    pub board: Vec<f64>,
}

impl NQueensFitness {
    pub fn new(dimension: usize) -> Self {
        let mut board = vec![0.0; dimension * dimension];

        let mut odd = false;
        let mut best_for_column = vec![-1.0; dimension];
        for i in 0..(dimension * dimension) {
            if i % dimension == 0 {
                odd = !odd;
            }
            if odd {
                board[i] = (i as f64 + 1.0).sqrt();
            } else {
                board[i] = (i as f64 + 1.0).log10();
            }
            best_for_column[i % dimension] = board[i].max(best_for_column[i % dimension]);
        }

        let max_score = best_for_column.iter().sum();

        NQueensFitness {
            c_max: dimension as f64 * (dimension as f64 - 1.0),
            dimension,
            max_score,
            board,
        }
    }

    pub fn count_collisions(&self, individual: &IntPerm) -> f64 {
        let genes = &individual.0;

        let mut colisions = 0.0;
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                if i == j {
                    continue;
                }
                let dist = i.abs_diff(j);
                let diff = genes[i].abs_diff(genes[j]);

                if diff == dist as u64 {
                    colisions += 1.0;
                }
            }
        }

        colisions
    }

    pub fn score(&self, individual: &IntPerm) -> f64 {
        let genes = &individual.0;

        let mut total_score = 0.0;
        for i in 0..self.dimension {
            total_score += self.board[((genes[i] as usize - 1) * self.dimension) + i];
        }

        total_score
    }
}

impl Fitness<IntPerm> for NQueensFitness {
    fn calculate_fitness(&self, individual: &IntPerm) -> f64 {
        let c = -1.3;
        let norm_colision = self.count_collisions(individual) / self.c_max;
        let norm_score = self.score(individual) / self.max_score;

        let fitness = norm_score + c * norm_colision;

        if fitness > 0.0 {
            fitness
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::nqueens_fitness::NQueensFitness;

    #[test]
    fn nqueens_board_filling() {
        let fitness = NQueensFitness::new(4);

        println!("{:?}", fitness.board);
        assert_eq!(fitness.board.len(), 16);
        for i in 0..4 {
            assert_eq!(fitness.board[i], (i as f64 + 1.0).sqrt());
        }
        for i in 4..8 {
            assert_eq!(fitness.board[i], (i as f64 + 1.0).log10());
        }
        for i in 8..12 {
            assert_eq!(fitness.board[i], (i as f64 + 1.0).sqrt());
        }
    }
}
