use crate::Crossover;
use crate::population::Real;

#[derive(Clone)]
pub struct LinearCrossover;

impl Crossover<Real> for LinearCrossover {
    fn crossover(&self, population: &mut Vec<Real>) {
        todo!()
    }
}