use rand::Rng;
use rayon::{
    prelude::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::{population::IntPerm, Individual};

use super::{assert_permutations, Crossover};
use crate::Rngs;

/// # Partially Mapped Crossover (PMX)
///
/// With the `crossover_rate` probability, swaps a random section between the parents, then
/// repairs the genes outside of it through the mapping the swap defines, so both children stay
/// permutations.
///
/// # Panics
///
/// `crossover` panics if `crossover_rate` is not within `[0, 1]`, or if two parents it crosses
/// are not permutations of `1..=n`.
#[derive(Clone)]
pub struct PartiallyMappedCrossover {
    pub crossover_rate: f64,
}

impl Default for PartiallyMappedCrossover {
    fn default() -> Self {
        PartiallyMappedCrossover {
            crossover_rate: 0.8,
        }
    }
}

impl PartiallyMappedCrossover {
    fn pmx_matching(parent1: &mut [i64], parent2: &mut [i64], start: usize, end: usize) {
        assert_permutations(parent1, parent2);

        // The mapping the swapped section defines, as mapping[gene - 1], 0 where a gene has none:
        // mapping_1 takes the genes parent1 gave away to those it got in their place, and
        // mapping_2 the other way.
        let mut mapping_1 = vec![0; parent1.len()];
        let mut mapping_2 = vec![0; parent1.len()];

        for i in start..end {
            std::mem::swap(&mut parent1[i], &mut parent2[i]);

            mapping_1[(parent2[i] - 1) as usize] = parent1[i];
            mapping_2[(parent1[i] - 1) as usize] = parent2[i];
        }

        // A gene outside of the section that the section now holds too follows the mapping until
        // it reaches one the section does not hold. The genes are permutations, so it gets there.
        for i in (0..start).chain(end..parent1.len()) {
            while mapping_2[(parent1[i] - 1) as usize] != 0 {
                parent1[i] = mapping_2[(parent1[i] - 1) as usize];
            }

            while mapping_1[(parent2[i] - 1) as usize] != 0 {
                parent2[i] = mapping_1[(parent2[i] - 1) as usize];
            }
        }
    }
}

impl Crossover<IntPerm> for PartiallyMappedCrossover {
    fn crossover(&self, population: &mut [IntPerm], rngs: &Rngs) {
        population
            .par_chunks_exact_mut(2)
            .enumerate()
            .for_each(|(pair, chunk)| {
                let mut rng = rngs.rng(pair);
                if rng.random_bool(self.crossover_rate) {
                    let (first, second) = chunk.split_at_mut(1);

                    let len = first[0].get_chromosome().len();

                    let start = rng.random_range(0..len);
                    let end = rng.random_range(start..len);

                    PartiallyMappedCrossover::pmx_matching(
                        &mut first[0].chromosome,
                        &mut second[0].chromosome,
                        start,
                        end,
                    );
                }
            });
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pmx_matching_1() {
        let mut parent1 = vec![9, 8, 4, 5, 6, 7, 1, 3, 2, 10];
        let mut parent2 = vec![8, 7, 1, 2, 3, 10, 9, 5, 4, 6];

        super::PartiallyMappedCrossover::pmx_matching(&mut parent1, &mut parent2, 3, 6);

        assert_eq!(parent1, vec![9, 8, 4, 2, 3, 10, 1, 6, 5, 7]);
        assert_eq!(parent2, vec![8, 10, 1, 5, 6, 7, 9, 2, 4, 3]);
    }

    #[test]
    fn pmx_matching_2() {
        let mut parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut parent2 = vec![9, 3, 7, 8, 2, 6, 5, 1, 4];

        super::PartiallyMappedCrossover::pmx_matching(&mut parent1, &mut parent2, 2, 6);

        assert_eq!(parent1, vec![1, 5, 7, 8, 2, 6, 3, 4, 9]);
        assert_eq!(parent2, vec![9, 7, 3, 4, 5, 6, 2, 1, 8]);
    }

    #[test]
    fn pmx_matching_with_loop() {
        let mut parent1 = vec![4, 2, 3, 1, 5];
        let mut parent2 = vec![2, 3, 1, 4, 5];

        super::PartiallyMappedCrossover::pmx_matching(&mut parent1, &mut parent2, 1, 4);

        assert_eq!(parent1, vec![2, 3, 1, 4, 5]);
        assert_eq!(parent2, vec![4, 2, 3, 1, 5]);
    }

    #[test]
    #[should_panic(expected = "the chromosome must be a permutation of 1..=3, got [1, 2, 1]")]
    fn pmx_rejects_a_repeated_gene() {
        // The mapping of this section, 1 to 2 and 2 to 1, used to send the last gene round it
        // forever.
        super::PartiallyMappedCrossover::pmx_matching(&mut [1, 2, 1], &mut [2, 1, 3], 0, 2);
    }
}
