use dyn_clone::DynClone;

use crate::population::Individual;

pub trait Coding<T: Individual>: 'static + DynClone + Send + Sync  {
    type Output;

    fn decode(&self, individual: &T) -> Self::Output;
}
