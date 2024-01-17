use rand::{thread_rng, Rng};

use super::Individual;

#[derive(Debug, Clone)]
pub struct Int {
    chromosome: Vec<i64>,
    fitness: f64,
    pub range: <Int as Individual>::RangeType,
}

impl Individual for Int {
    type Gene = i64;
    type RangeType = (i64, i64);

    fn generate_member(dimension: u32, range: &Self::RangeType) -> Self {
        let mut rng = thread_rng();

        Self {
            chromosome: (0..dimension)
                .map(|_| rng.gen_range(range.0..=range.1))
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
