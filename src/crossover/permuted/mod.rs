use crate::population::assert_permutation;
use crate::*;

mod cycle_crossover;
mod ordered_crossover;
mod partially_mapped_crossover;

pub use cycle_crossover::*;
pub use ordered_crossover::*;
pub use partially_mapped_crossover::*;

/// Checks that both parents are permutations of `1..=n` of the same length, so the operators
/// can index their tables by `gene - 1`. `IntPerm::chromosome` is a public field, so nothing
/// else guarantees it.
///
/// # Panics
///
/// If their lengths differ, or one of them is not a permutation.
fn assert_permutations(parent1: &[i64], parent2: &[i64]) {
    assert!(
        parent1.len() == parent2.len(),
        "the parents must have the same length, got {} and {} genes",
        parent1.len(),
        parent2.len()
    );

    assert_permutation(parent1);
    assert_permutation(parent2);
}

#[cfg(test)]
mod tests {
    use super::assert_permutations;

    #[test]
    fn accepts_permutations() {
        assert_permutations(&[3, 1, 2], &[1, 2, 3]);
        assert_permutations(&[1], &[1]);
        assert_permutations(&[], &[]);
    }

    #[test]
    #[should_panic(expected = "the chromosome must be a permutation of 1..=3, got [0, 1, 2]")]
    fn rejects_a_gene_below_one() {
        assert_permutations(&[0, 1, 2], &[1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "the chromosome must be a permutation of 1..=3, got [1, 2, 4]")]
    fn rejects_a_gene_above_the_length() {
        assert_permutations(&[1, 2, 3], &[1, 2, 4]);
    }

    #[test]
    #[should_panic(expected = "the parents must have the same length, got 3 and 2 genes")]
    fn rejects_parents_of_different_lengths() {
        assert_permutations(&[1, 2, 3], &[1, 2]);
    }
}
