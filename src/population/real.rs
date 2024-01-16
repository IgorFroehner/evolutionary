use rand::{thread_rng, Rng};

use super::Individual;

#[derive(Debug, Clone)]
pub struct Real {
    pub chromosome: Vec<f64>,
    fitness: f64,
    pub range: <Real as Individual>::RangeType,
}

impl Individual for Real {
    type Gene = f64;
    type RangeType = (f64, f64);

    fn generate_member(dimension: u32, (a, b): &Self::RangeType) -> Self {
        let mut rng = thread_rng();

        Self {
            chromosome: (0..dimension)
                .map(|_| rng.gen_range(*a..=*b))
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
