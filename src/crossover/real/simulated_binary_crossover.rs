use crate::Crossover;
use crate::population::Real;

#[derive(Clone)]
pub struct SimulatedBinaryCrossover;

impl Crossover<Real> for SimulatedBinaryCrossover {
    fn crossover(&self, population: &mut Vec<Real>) {
        todo!()
    }
}