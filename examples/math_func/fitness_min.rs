use evolutionary::prelude::*;

use crate::{f, C_MAX, L, RANGE};

#[derive(Clone)]
struct MathFitnessMin;

impl MathFitnessMin {
    fn get_x(&self, bin: &Bin) -> f64 {
        let d = convert_bin(&bin.chromosome);

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
