use crate::{Individual, Rngs, Selection};

#[derive(Clone)]
pub struct DoNothingSelection;

impl<T: Individual> Selection<T> for DoNothingSelection {
    fn get_mating_pool(&self, initial_population: &Vec<T>, _rngs: &Rngs) -> Vec<T> {
        initial_population.clone()
    }
}
