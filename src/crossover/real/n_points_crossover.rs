use rand::{Rng, thread_rng};
use rand::prelude::IteratorRandom;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;
use crate::{Crossover, Individual};
use crate::crossover::NPointsCrossover;
use crate::population::Real;

impl Crossover<Real> for NPointsCrossover {
    fn crossover(&self, population: &mut Vec<Real>) {
        population.par_chunks_mut(2).for_each_init(
            || thread_rng(),
            |mut rng, chunk| {
                if rng.gen_bool(self.crossover_rate) {
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
            },
        );
    }
}
