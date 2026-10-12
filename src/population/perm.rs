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
        assert!(
            is_permutation(&chromosome),
            "the chromosome must be a permutation of 1..={}, got {chromosome:?}",
            chromosome.len()
        );

        Self {
            chromosome,
            fitness: f64::NAN,
        }
    }
}

/// Whether `chromosome` is a permutation of `1..=chromosome.len()`.
pub(crate) fn is_permutation(chromosome: &[i64]) -> bool {
    let mut sorted = chromosome.to_vec();
    sorted.sort_unstable();
    sorted.iter().copied().eq(1..=chromosome.len() as i64)
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
