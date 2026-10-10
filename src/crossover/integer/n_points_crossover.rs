use rand::Rng;
use rand::prelude::IteratorRandom;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;
use crate::{Crossover, Individual, Rngs};
use crate::crossover::NPointsCrossover;
use crate::population::Int;

impl Crossover<Int> for NPointsCrossover {
    fn crossover(&self, population: &mut Vec<Int>, rngs: &Rngs) {
        population
            .par_chunks_mut(2)
            .enumerate()
            .for_each(|(pair, chunk)| {
                let mut rng = rngs.rng(pair);
                if rng.random_bool(self.crossover_rate) {
                    let mut parent1 = chunk[0].clone();
                    let mut parent2 = chunk[1].clone();

                    let len = parent1.get_chromosome().len();

                    let mut points: Vec<usize> = (0..len).choose_multiple(&mut rng, self.n_points);

                    points.sort();

                    let mut start = 0;

                    for point in points {
                        NPointsCrossover::swap_sections(
                            &mut parent1.get_mut_chromosome(),
                            &mut parent2.get_mut_chromosome(),
                            start,
                            point,
                        );

                        start = point;
                    }

                    chunk[0] = parent1;
                    chunk[1] = parent2;
                }
            });
    }
}
