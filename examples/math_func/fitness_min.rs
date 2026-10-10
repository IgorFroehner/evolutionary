use evolutionary::prelude::*;

use crate::{f, L, RANGE};

const C_MAX: f64 = 2.;

#[derive(Clone)]
struct MathFitnessMin;

impl MathFitnessMin {
    fn get_x(&self, bin: &Bin) -> f64 {
        let d = convert_bin(bin.get_chromosome());

        within_range(RANGE, L, d)
    }
}

impl Fitness<Bin> for MathFitnessMin {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        let x = self.get_x(individual);

        if f(x) < C_MAX {
            C_MAX - f(x)
        } else {
            0.
        }
    }
}
