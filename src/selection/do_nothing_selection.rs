use crate::{Individual, Selection};

#[derive(Clone)]
pub struct DoNothingSelection;

impl<T: Individual> Selection<T> for DoNothingSelection {
    fn get_mating_pool(&self, initial_population: &Vec<T>) -> Vec<T> {
        initial_population.clone()
    }
}
