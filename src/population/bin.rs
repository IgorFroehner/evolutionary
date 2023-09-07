use rand::{thread_rng, Rng};

use super::Individual;

#[derive(Debug, Clone)]
pub struct Bin(pub Vec<bool>, pub f64);

impl Individual for Bin {
    type Gene = bool;
    type RangeType = ();

    fn generate_member(dimension: u32, _: &Self::RangeType) -> Self {
        let mut rng = thread_rng();

        Self(
            (0..dimension).map(|_| rng.gen()).collect::<Vec<bool>>(),
            0.0,
        )
    }

    fn get_vec(&self) -> &Vec<Self::Gene> {
        &self.0
    }

    fn get_fitness(&self) -> f64 {
        self.1
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.1 = fitness;
    }
}
