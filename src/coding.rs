use crate::population::Individual;

pub trait Coding<T: Individual> {
    type Output;

    fn decode(&self, individual: &T) -> Self::Output;
}
