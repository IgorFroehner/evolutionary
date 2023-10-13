use rand::{thread_rng, Rng};

use super::Individual;

#[derive(Debug, Clone)]
pub struct Int(pub Vec<i64>, f64);

impl Individual for Int {
    type Gene = i64;
    type RangeType = (i64, i64);

    fn generate_member(dimension: u32, range: &Self::RangeType) -> Self {
        let mut rng = thread_rng();

        Self(
            (0..dimension)
                .map(|_| rng.gen_range(range.0..=range.1))
                .collect::<Vec<i64>>(),
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
