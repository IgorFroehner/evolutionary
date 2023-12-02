use evolutionary::prelude::*;

use crate::{f, C_MIN, L, RANGE};

#[derive(Clone)]
pub struct MathFitnessMax;

impl MathFitnessMax {
    pub fn get_x(bin: &Bin) -> f64 {
        let d = convert_bin(&bin.get_chromosome());

        within_range(RANGE, L, d)
    }
}

impl Fitness<Bin> for MathFitnessMax {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        let x = MathFitnessMax::get_x(individual);

        if f(x) - C_MIN > 0. {
            f(x) - C_MIN
        } else {
            0.
        }
    }
}
