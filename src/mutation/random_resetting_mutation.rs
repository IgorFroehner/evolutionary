// Random Resetting mutation is implemented for Real and Int chromosome types.

/// # Substitute Mutation
///
/// For each gene in the real representation it has `mutation_rate` probability of replacing
/// the gene with a random value within the range.
///
/// # Panics
///
/// `mutate` panics if `mutation_rate` is not within `[0, 1]`, or when it resets a gene of an
/// individual whose `range` is empty: `range.0 > range.1` for [`Int`](crate::prelude::Int),
/// `range.0 >= range.1` (or a bound that is not finite) for [`Real`](crate::prelude::Real).
#[derive(Clone)]
pub struct RandomResettingMutation {
    pub mutation_rate: f64,
}

impl Default for RandomResettingMutation {
    fn default() -> Self {
        RandomResettingMutation {
            mutation_rate: 0.05,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::population::{Int, Real};
    use crate::{Individual, Mutation, Rngs};

    /// Genes start outside of the range, so every reset gene can be told apart.
    fn reset_ints(mutation_rate: f64) -> Vec<i64> {
        let mut population = vec![Int::new(vec![100; 300], (0, 9)); 300];
        RandomResettingMutation { mutation_rate }.mutate(&mut population, &Rngs::new(2));

        population
            .iter()
            .flat_map(|individual| individual.get_chromosome().iter().copied())
            .filter(|&gene| gene != 100)
            .collect()
    }

    #[test]
    fn resets_the_expected_share_of_int_genes_within_the_range() {
        let reset = reset_ints(0.05);

        // 90 000 genes at 5%: 4 500 expected, with a standard deviation of about 65.
        assert!(
            (reset.len() as f64 - 4500.0).abs() < 4.0 * 65.0,
            "{}",
            reset.len()
        );
        assert!(reset.iter().all(|gene| (0..=9).contains(gene)));
    }

    #[test]
    fn int_genes_can_be_reset_to_either_bound() {
        let reset = reset_ints(1.0);

        assert!(reset.contains(&0) && reset.contains(&9));
    }

    #[test]
    fn a_zero_rate_resets_nothing() {
        assert!(reset_ints(0.0).is_empty());
    }

    #[test]
    fn resets_the_expected_share_of_real_genes() {
        let mut population = vec![Real::new(vec![100.0; 300], (-1.0, 1.0)); 300];
        RandomResettingMutation {
            mutation_rate: 0.05,
        }
        .mutate(&mut population, &Rngs::new(2));

        let reset = population
            .iter()
            .flat_map(|individual| individual.get_chromosome())
            .filter(|&&gene| gene != 100.0)
            .count() as f64;

        // 90 000 genes at 5%: 4 500 expected, with a standard deviation of about 65.
        assert!((reset - 4500.0).abs() < 4.0 * 65.0, "{reset}");
    }

    #[test]
    fn resets_real_genes_within_the_range() {
        let mut population = vec![Real::new(vec![100.0; 300], (-1.0, 1.0)); 300];
        RandomResettingMutation { mutation_rate: 1.0 }.mutate(&mut population, &Rngs::new(2));

        assert!(population
            .iter()
            .flat_map(|individual| individual.get_chromosome())
            .all(|gene| (-1.0..=1.0).contains(gene)));
    }
}
