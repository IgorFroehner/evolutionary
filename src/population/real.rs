use rand::Rng;

use super::Individual;

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(serde_derive::Serialize, serde_derive::Deserialize)
)]
pub struct Real {
    pub chromosome: Vec<f64>,
    fitness: f64,
    pub range: <Real as Individual>::RangeType,
}

impl Real {
    /// Creates an individual from a known chromosome, for example to seed a population with
    /// saved solutions. `range` is the range the genes were generated in.
    pub fn new(chromosome: Vec<f64>, range: <Real as Individual>::RangeType) -> Self {
        Self {
            chromosome,
            fitness: 0.0,
            range,
        }
    }
}

impl Individual for Real {
    type Gene = f64;
    type RangeType = (f64, f64);

    fn generate_member<R: Rng + ?Sized>(
        dimension: u32,
        (a, b): &Self::RangeType,
        rng: &mut R,
    ) -> Self {
        Self {
            chromosome: (0..dimension)
                .map(|_| rng.random_range(*a..=*b))
                .collect::<Vec<f64>>(),
            fitness: 0.0,
            range: (*a, *b),
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
