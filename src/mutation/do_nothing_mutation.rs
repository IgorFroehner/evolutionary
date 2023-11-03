use crate::{Individual, Mutation};

#[derive(Clone)]
pub struct DoNothingMutation;

impl<T: Individual> Mutation<T> for DoNothingMutation {
    fn mutate(&self, _population: &mut Vec<T>) {}
}
