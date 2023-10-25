use rand::{thread_rng, Rng};

use super::Individual;

#[derive(Debug, Clone)]
pub struct Bin {
    pub chromosome: Vec<bool>,
    fitness: f64,
}

impl Individual for Bin {
    type Gene = bool;
    type RangeType = ();

    fn generate_member(dimension: u32, _: &Self::RangeType) -> Self {
        let mut rng = thread_rng();

        Self {
            chromosome: (0..dimension).map(|_| rng.gen()).collect::<Vec<bool>>(),
            fitness: 0.0,
        }
    }

    fn get_chromossome(&self) -> &Vec<Self::Gene> {
        &self.chromosome
    }

    fn set_gene(&mut self, index: usize, value: Self::Gene) {
        self.chromosome[index] = value;
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }

    fn get_fitness(&self) -> f64 {
        self.fitness
    }
}
