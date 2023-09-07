use evolutionary_computing::{
    population::Bin,
    utils::{convert_bin, within_range}, fitness::Fitness,
};

use crate::{RANGE_LX, RANGE_ST, L, C};

pub fn get_st(bin: &Bin) -> f64 {
    let d = convert_bin(&bin.0[0..5].to_vec());
    let x = within_range(RANGE_ST, L, d);

    x.round()
}

pub fn get_lx(bin: &Bin) -> f64 {
    let d = convert_bin(&bin.0[5..10].to_vec());
    let x = within_range(RANGE_LX, L, d);

    x.round()
}

pub fn f(st: f64, lx: f64) -> f64 {
    30. * st + 40. * lx
}

fn f_norm(st: f64, lx: f64) -> f64 {
    f(st, lx) / 1360.
}

fn r_norm(st: f64, lx: f64) -> f64 {
    0f64.max((st + 2. * lx - 40.0) / 16.0)
}

#[derive(Clone)]
pub struct RadioFitness;

impl Fitness<Bin> for RadioFitness {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        let st = get_st(individual);
        let lx: f64 = get_lx(individual);
        let y = f_norm(st, lx);
        let r = r_norm(st, lx);

        y + C * r
    }
}

#[cfg(test)]
mod tests {
    use evolutionary_computing::{population::Bin, fitness::Fitness};

    use crate::{radio_fitness::{get_st, get_lx}, RANGE_ST, RANGE_LX};

    use super::RadioFitness;

    #[test]
    fn st_is_generated_between_range_st() {
        let max_individual = Bin(vec![true, true, true, true, true], 0);
        let min_individual = Bin(vec![false, false, false, false, false], 0);

        assert_eq!(get_st(&min_individual), RANGE_ST.0);
        assert_eq!(get_st(&max_individual), RANGE_ST.1);
    }

    #[test]
    fn lx_is_generated_between_range_lx() {
        let min_individual = Bin(vec![false, false, false, false, false, false, false, false, false, false], 0);
        let max_individual = Bin(vec![true, true, true, true, true, true, true, true, true, true], 0);

        assert_eq!(get_lx(&min_individual), RANGE_LX.0);
        assert_eq!(get_lx(&max_individual), RANGE_LX.1);
    }

    #[test]
    fn fitness() {
        let fitness = RadioFitness;

        let individual_0 = Bin(vec![true, false, true, true, false, false, true, true, true, true], 0);
        let individual_1 = Bin(vec![true, false, true, true, true, true, true, true, true, false], 0);

        let st = get_st(&individual_1);
        let lx = get_lx(&individual_1);

        println!("st: {}", st);
        println!("lx: {}", lx);
        println!("lucro: {}", 30. * st + 40. * lx);

        assert_eq!(fitness.calculate_fitness(&individual_0), 0.56);
    }
}
