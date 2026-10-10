use crate::{Individual, Mutation, Rngs};

#[derive(Clone)]
pub struct DoNothingMutation;

impl<T: Individual> Mutation<T> for DoNothingMutation {
    fn mutate(&self, _population: &mut [T], _rngs: &Rngs) {}
}
