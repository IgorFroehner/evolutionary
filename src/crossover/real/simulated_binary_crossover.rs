use crate::{Crossover, Rngs};
use crate::population::Real;

#[derive(Clone)]
pub struct SimulatedBinaryCrossover;

impl Crossover<Real> for SimulatedBinaryCrossover {
    fn crossover(&self, _population: &mut Vec<Real>, _rngs: &Rngs) {
        todo!()
    }
}