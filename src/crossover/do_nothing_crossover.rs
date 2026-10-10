use crate::{Crossover, Individual, Rngs};

#[derive(Clone)]
pub struct DoNothingCrossover;

impl<T: Individual> Crossover<T> for DoNothingCrossover {
    fn crossover(&self, _population: &mut Vec<T>, _rngs: &Rngs) {}
}
