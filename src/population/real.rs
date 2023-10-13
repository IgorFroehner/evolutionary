use rand::{thread_rng, Rng};

use super::Individual;

#[derive(Debug, Clone)]
pub struct Real(pub Vec<f64>, f64);

impl Individual for Real {
    type Gene = f64;
    type RangeType = (f64, f64);

    fn generate_member(dimension: u32, (a, b): &Self::RangeType) -> Self {
        let mut rng = thread_rng();

        Self(
            (0..dimension)
                .map(|_| rng.gen_range(*a..=*b))
                .collect::<Vec<f64>>(),
            0.0,
        )
    }

    fn get_vec(&self) -> &Vec<Self::Gene> {
        &self.0
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.1 = fitness;
    }

    fn get_fitness(&self) -> f64 {
        self.1
    }
}
