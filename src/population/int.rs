use rand::Rng;

use super::Individual;

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(serde_derive::Serialize, serde_derive::Deserialize)
)]
pub struct Int {
    chromosome: Vec<i64>,
    fitness: f64,
    pub range: <Int as Individual>::RangeType,
}

impl Int {
    /// Creates an individual from a known chromosome, for example to seed a population with
    /// saved solutions. `range` is the range the genes were generated in.
    pub fn new(chromosome: Vec<i64>, range: <Int as Individual>::RangeType) -> Self {
        Self {
            chromosome,
            fitness: 0.0,
            range,
        }
    }
}

impl Individual for Int {
    type Gene = i64;
    type RangeType = (i64, i64);

    fn generate_member<R: Rng + ?Sized>(
        dimension: u32,
        range: &Self::RangeType,
        rng: &mut R,
    ) -> Self {
        Self {
            chromosome: (0..dimension)
                .map(|_| rng.random_range(range.0..=range.1))
                .collect::<Vec<i64>>(),
            fitness: 0.0,
            range: *range,
        }
    }

    fn get_chromosome(&self) -> &Vec<Self::Gene> {
        &self.chromosome
    }

    fn get_mut_chromosome(&mut self) -> &mut Vec<Self::Gene> {
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
