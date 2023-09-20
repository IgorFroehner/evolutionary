use rand::{seq::IteratorRandom, thread_rng, Rng};

use crate::population::Bin;

use super::Crossover;

#[derive(Clone)]
pub struct  BinCrossover {
    pub crossover_rate: f64,
    pub n_points: usize,
}

impl Default for BinCrossover {
    fn default() -> Self {
        BinCrossover {
            crossover_rate: 0.8,
            n_points: 1,
        }
    }
}

impl BinCrossover {
    fn swap_sections(vec1: &mut Vec<bool>, vec2: &mut Vec<bool>, start: usize, end: usize) {
        for i in start..end {
            let temp = vec1[i];
            vec1[i] = vec2[i];
            vec2[i] = temp;
        }
    }
}

impl Crossover<Bin> for BinCrossover {
    fn crossover(&self, population: &mut Vec<Bin>) {
        let mut rng = thread_rng();

        for i in 0..(population.len() / 2) {
            if rng.gen_bool(self.crossover_rate) {
                let mut parent1 = population[i * 2].clone();
                let mut parent2 = population[i * 2 + 1].clone();

                let len = parent1.0.len();

                let mut points: Vec<usize> = (0..len).choose_multiple(&mut rng, self.n_points);

                points.sort();

                let mut start = 0;

                for point in points {
                    BinCrossover::swap_sections(&mut parent1.0, &mut parent2.0, start, point);

                    start = point;
                }

                population[i * 2] = parent1;
                population[i * 2 + 1] = parent2;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn swap_sections() {
        let mut vec1 = vec![false, false, false, false, false, false];
        let mut vec2 = vec![true, true, true, true, true, true];

        super::BinCrossover::swap_sections(&mut vec1, &mut vec2, 2, 4);

        assert_eq!(vec1, vec![false, false, true, true, false, false]);
        assert_eq!(vec2, vec![true, true, false, false, true, true]);
    }
}
