use plotters::prelude::*;
use plotters::style::{RED, WHITE};

pub fn convert_bin(vec: &Vec<bool>) -> f64 {
    let mut res = 0.0;
    for i in 0..vec.len() {
        res += (if vec[i] { 1.0 } else { 0.0 }) * 2f64.powf(i as f64);
    }

    res
}

pub fn within_range(range: (f64, f64), l: f64, d: f64) -> f64 {
    let a = range.1 - range.0;
    let b = 2f64.powf(l) - 1.;

    (range.0 + (a / b) * d).floor()
}

#[cfg(test)]
mod test {
    use crate::utils::{convert_bin, within_range};

    #[test]
    fn convert_bin_test() {
        let seven_vec = vec![true, true, true, false]; // 1 + 2 + 4
        assert_eq!(convert_bin(&seven_vec), 7.0);

        let thirteen_vec: Vec<bool> = vec![true, false, true, true]; // 1 + 0 + 4 + 8
        assert_eq!(convert_bin(&thirteen_vec), 13.0);

        let sixty_four_vec: Vec<bool> = vec![false, false, false, false, false, false, true];
        assert_eq!(convert_bin(&sixty_four_vec), 64.0);
    }

    #[test]
    fn within_range_test_1() {
        let value_vec = vec![true, true, true, true, true, true, true, true, true, true];
        let value = convert_bin(&value_vec);

        let range = (0., 1.);
        let l = value_vec.len() as f64;
        let d = value;

        println!("value: {}", value);

        assert_eq!(within_range(range, l, d), range.1);
    }

    #[test]
    fn within_range_test_2() {
        let range = (0., 16.);
        let l = 5.0;
        let d = 30.0;

        assert_eq!(within_range(range, l, d), 15.0);
    }
}
