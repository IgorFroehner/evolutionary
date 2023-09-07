use evolutionary_computing::{population::Bin, fitness::Fitness, utils::{convert_bin, within_range}};

use crate::{f, RANGE, L, C_MAX};

#[derive(Clone)]
struct MathFitnessMin;

impl MathFitnessMin {
    fn get_x(&self, bin: &Bin) -> f64 {
        let d = convert_bin(&bin.0);

        within_range(RANGE, L, d)
    }
}

impl Fitness<Bin> for MathFitnessMin {
    fn calculate_fitness(&self, individual:  &Bin) -> f64 {
        let x = self.get_x(individual);

        if f(x) < C_MAX { C_MAX - f(x) }
        else { 0. }
    }
}
