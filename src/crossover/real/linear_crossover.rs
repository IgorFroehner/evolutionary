use crate::{Crossover, Rngs};
use crate::population::Real;

#[derive(Clone)]
pub struct LinearCrossover;

impl Crossover<Real> for LinearCrossover {
    fn crossover(&self, _population: &mut [Real], _rngs: &Rngs) {
        todo!()
    }
}