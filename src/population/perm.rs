use rand::{seq::SliceRandom, Rng};

use super::Individual;

#[derive(Debug, Clone)]
pub struct IntPerm {
    pub chromosome: Vec<i64>,
    fitness: f64,
}

impl Individual for IntPerm {
    type Gene = i64;
    type RangeType = ();

    fn generate_member<R: Rng + ?Sized>(dimension: u32, _: &Self::RangeType, rng: &mut R) -> Self {
        let mut member = (1..=dimension as i64).collect::<Vec<i64>>();
        member.shuffle(rng);

        Self {
            chromosome: member,
            fitness: 0.0,
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
