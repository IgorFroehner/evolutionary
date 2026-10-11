use rand::distr::{Bernoulli, Distribution};
use rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{population::Real, Mutation};
use crate::Rngs;

#[derive(Clone)]
pub struct DeltaMutation {
    pub mutation_rate: f64,
    pub frac: f64,
}

impl Default for DeltaMutation {
    fn default() -> Self {
        DeltaMutation {
            mutation_rate: 0.05,
            frac: 10.0,
        }
    }
}

impl Mutation<Real> for DeltaMutation {
    fn mutate(&self, population: &mut [Real], rngs: &Rngs) {
        // Built once: `Rng::random_bool` builds the same distribution again at every call.
        let mutates = Bernoulli::new(self.mutation_rate).unwrap();

        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, individual)| {
                let mut rng = rngs.rng(index);
                for j in 0..individual.chromosome.len() {
                    if mutates.sample(&mut rng) {
                        let delta =
                            rng.random_range(individual.range.0..individual.range.1) / self.frac;

                        let result = if rng.random_bool(0.5) {
                            individual.chromosome[j] - delta
                        } else {
                            individual.chromosome[j] + delta
                        };

                        if result < individual.range.0 {
                            individual.chromosome[j] = individual.range.0;
                        } else if result > individual.range.1 {
                            individual.chromosome[j] = individual.range.1;
                        } else {
                            individual.chromosome[j] = result;
                        }
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Individual;

    fn genes(mutation: &DeltaMutation, start: f64, range: (f64, f64)) -> Vec<f64> {
        let mut population = vec![Real::new(vec![start; 300], range); 300];
        mutation.mutate(&mut population, &Rngs::new(7));

        population
            .iter()
            .flat_map(|individual| individual.get_chromosome().iter().copied())
            .collect()
    }

    #[test]
    fn moves_each_gene_by_less_than_the_range_over_frac() {
        let mutation = DeltaMutation {
            mutation_rate: 1.0,
            frac: 10.0,
        };

        let genes = genes(&mutation, 5.0, (0.0, 10.0));

        assert!(genes.iter().all(|gene| (gene - 5.0).abs() < 1.0));
        assert!(genes.iter().any(|&gene| gene < 5.0) && genes.iter().any(|&gene| gene > 5.0));
    }

    #[test]
    fn clamps_to_the_range_of_each_individual() {
        let mutation = DeltaMutation {
            mutation_rate: 1.0,
            frac: 1.0,
        };

        let genes = genes(&mutation, 1.0, (-0.5, 2.0));

        assert!(genes.iter().all(|gene| (-0.5..=2.0).contains(gene)));
        assert!(genes.contains(&-0.5) && genes.contains(&2.0));
    }

    #[test]
    fn a_zero_rate_changes_nothing() {
        let mutation = DeltaMutation {
            mutation_rate: 0.0,
            frac: 1.0,
        };

        assert!(genes(&mutation, 0.5, (0.0, 1.0)).iter().all(|&gene| gene == 0.5));
    }
}
