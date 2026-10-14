use rand::{seq::SliceRandom, Rng};

use super::Individual;

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(serde_derive::Serialize, serde_derive::Deserialize)
)]
pub struct IntPerm {
    pub chromosome: Vec<i64>,
    #[cfg_attr(
        feature = "serde",
        serde(deserialize_with = "crate::fitness::deserialize_fitness")
    )]
    fitness: f64,
}

impl IntPerm {
    /// Creates an individual from a known chromosome, for example to seed a population with
    /// saved solutions. Its fitness is `NaN` until it is evaluated.
    ///
    /// # Panics
    ///
    /// If `chromosome` is not a permutation of `1..=chromosome.len()`, which every permutation
    /// operator relies on.
    pub fn new(chromosome: Vec<i64>) -> Self {
        assert_permutation(&chromosome);

        Self {
            chromosome,
            fitness: f64::NAN,
        }
    }
}

/// Whether `chromosome` is a permutation of `1..=chromosome.len()`, in one pass.
pub(crate) fn is_permutation(chromosome: &[i64]) -> bool {
    let len = chromosome.len();

    // seen[gene - 1]: each gene of 1..=len must come exactly once.
    let mut seen = vec![false; len];

    chromosome.iter().all(|&gene| {
        let index = usize::try_from(gene)
            .ok()
            .and_then(|gene| gene.checked_sub(1))
            .filter(|&index| index < len);

        match index {
            Some(index) if !seen[index] => {
                seen[index] = true;
                true
            }
            _ => false,
        }
    })
}

/// Checks the chromosome of [`IntPerm::new`] and the parents of the permutation crossovers, with
/// one message for all of them.
///
/// # Panics
///
/// If `chromosome` is not a permutation of `1..=chromosome.len()`.
pub(crate) fn assert_permutation(chromosome: &[i64]) {
    if !is_permutation(chromosome) {
        not_a_permutation(chromosome);
    }
}

#[cold]
fn not_a_permutation(chromosome: &[i64]) -> ! {
    panic!(
        "the chromosome must be a permutation of 1..={}, got {chromosome:?}",
        chromosome.len()
    );
}

impl Individual for IntPerm {
    type Gene = i64;
    type RangeType = ();

    fn generate_member<R: Rng + ?Sized>(dimension: u32, _: &Self::RangeType, rng: &mut R) -> Self {
        let mut member = (1..=dimension as i64).collect::<Vec<i64>>();
        member.shuffle(rng);

        Self {
            chromosome: member,
            fitness: f64::NAN,
        }
    }

    fn get_chromosome(&self) -> &[Self::Gene] {
        &self.chromosome
    }

    fn get_mut_chromosome(&mut self) -> &mut [Self::Gene] {
        &mut self.chromosome
    }

    fn set_gene(&mut self, index: usize, value: Self::Gene) {
        self.chromosome[index] = value;
    }

    fn get_gene(&self, index: usize) -> Self::Gene {
        self.chromosome[index]
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }

    fn get_fitness(&self) -> f64 {
        self.fitness
    }
}

#[cfg(test)]
mod tests {
    use super::is_permutation;

    #[test]
    fn is_permutation_of_one_to_the_length() {
        assert!(is_permutation(&[]));
        assert!(is_permutation(&[1]));
        assert!(is_permutation(&[3, 1, 2]));

        assert!(!is_permutation(&[1, 1, 3]));
        assert!(!is_permutation(&[0, 1, 2]));
        assert!(!is_permutation(&[1, 2, 4]));
        assert!(!is_permutation(&[-1, 1, 2]));
        assert!(!is_permutation(&[i64::MIN, 1, 2]));
    }
}
