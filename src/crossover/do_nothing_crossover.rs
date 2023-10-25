
use crate::{Individual, Crossover};

#[derive(Clone)]
pub struct DoNothingCrossover;

impl<T: Individual> Crossover<T> for DoNothingCrossover {
    fn crossover(&self, _population: &mut Vec<T>) {}
}
