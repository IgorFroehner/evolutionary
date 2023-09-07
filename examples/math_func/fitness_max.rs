use evolutionary_computing::{population::Bin, fitness::Fitness, utils::{convert_bin, within_range}};

use crate::{RANGE, L, f, C_MIN};

#[derive(Clone)]
pub struct MathFitnessMax;

impl MathFitnessMax {
    pub fn get_x(bin: &Bin) -> f64 {
        let d = convert_bin(&bin.0);

        within_range(RANGE, L, d)
    }
}

impl Fitness<Bin> for MathFitnessMax {
    fn calculate_fitness(&self, individual:  &Bin) -> f64 {
        let x = MathFitnessMax::get_x(individual);

        if f(x) - C_MIN > 0. { f(x) - C_MIN }
        else { 0. }
    }
}

