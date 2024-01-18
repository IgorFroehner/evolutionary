use crate::Crossover;
use crate::population::Real;

#[derive(Clone)]
pub struct LinearCrossover;

impl Crossover<Real> for LinearCrossover {
    fn crossover(&self, _population: &mut Vec<Real>) {
        todo!()
    }
}