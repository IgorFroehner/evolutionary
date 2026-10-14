use crate::*;

mod cycle_crossover;
mod ordered_crossover;
mod partially_mapped_crossover;

pub use cycle_crossover::*;
pub use ordered_crossover::*;
pub use partially_mapped_crossover::*;

/// Checks that both parents are permutations of `1..=n`, `n` the length of `parent1`, so the
/// operators can index their tables by `gene - 1`. `IntPerm::chromosome` is a public field, so
/// nothing else guarantees it.
///
/// # Panics
///
/// If one of them is not.
fn assert_permutations(parent1: &[i64], parent2: &[i64]) {
    let n = parent1.len();

    // Set by the genes of parent1 and cleared by those of parent2, so each gene of 1..=n must be
    // in both exactly once.
    let mut seen = vec![false; n];

    for &gene in parent1 {
        match gene_index(gene, n) {
            Some(index) if !seen[index] => seen[index] = true,
            _ => not_a_permutation(parent1, n),
        }
    }

    if parent2.len() != n {
        not_a_permutation(parent2, n);
    }
    for &gene in parent2 {
        match gene_index(gene, n) {
            Some(index) if seen[index] => seen[index] = false,
            _ => not_a_permutation(parent2, n),
        }
    }
}

/// `gene - 1`, if `gene` is within `1..=n`.
fn gene_index(gene: i64, n: usize) -> Option<usize> {
    usize::try_from(gene)
        .ok()
        .and_then(|gene| gene.checked_sub(1))
        .filter(|&index| index < n)
}

#[cold]
fn not_a_permutation(chromosome: &[i64], n: usize) -> ! {
    panic!("the chromosome must be a permutation of 1..={n}, got {chromosome:?}");
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
    #[should_panic(expected = "the chromosome must be a permutation of 1..=3, got [1, 2]")]
    fn rejects_parents_of_different_lengths() {
        assert_permutations(&[1, 2, 3], &[1, 2]);
    }
}
