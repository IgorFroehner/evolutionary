//! Helpers to decode a [`Bin`](crate::prelude::Bin) chromosome into a number.

/// Reads `bits` as an unsigned binary number, least significant bit first: `[true, false, true]`
/// is 5.
///
/// The result is exact up to 53 bits, the precision of an `f64`. With more, the lowest bits
/// are lost to rounding (each set bit is added to the total in turn), and from 1 024 bits on a
/// set high bit makes the result infinite.
///
/// ```
/// use evolutionary::prelude::*;
///
/// assert_eq!(convert_bin(&[true, false, true]), 5.0);
/// assert_eq!(convert_bin(&[true; 53]), 2f64.powi(53) - 1.0);
/// ```
pub fn convert_bin(bits: &[bool]) -> f64 {
    let mut res = 0.0;
    for (i, &bit) in bits.iter().enumerate() {
        if bit {
            res += 2f64.powi(i as i32);
        }
    }

    res
}

/// Maps `d`, a number of `l` bits (as [`convert_bin`] decodes it, so from 0 to `2^l - 1`),
/// linearly onto `range`: 0 gives `range.0`, `2^l - 1` gives `range.1`, and the values in
/// between are spread evenly, `(range.1 - range.0) / (2^l - 1)` apart.
///
/// The result is not rounded: with 16 bits over `(-2.0, 2.0)` it takes 65 536 values, not the
/// five integers. Round it (`.round()` or `.floor()`) to decode an integer.
///
/// ```
/// use evolutionary::prelude::*;
///
/// assert_eq!(within_range((-2.0, 2.0), 16.0, 0.0), -2.0);
/// assert_eq!(within_range((-2.0, 2.0), 16.0, 65_535.0), 2.0);
/// // 2 bits over (0, 3): 0, 1, 2 and 3 give 0.0, 1.0, 2.0 and 3.0.
/// assert_eq!(within_range((0.0, 3.0), 2.0, 2.0), 2.0);
/// // Not rounded: 30 of the 31 steps of 5 bits over (0, 16).
/// assert!((within_range((0.0, 16.0), 5.0, 30.0) - 15.48).abs() < 0.01);
/// ```
pub fn within_range(range: (f64, f64), l: f64, d: f64) -> f64 {
    let steps = 2f64.powf(l) - 1.;

    range.0 + (range.1 - range.0) * (d / steps)
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
    fn convert_bin_is_exact_up_to_53_bits() {
        assert_eq!(convert_bin(&[true; 53]), 9_007_199_254_740_991.0);

        let mut high_bit = vec![false; 64];
        high_bit[63] = true;
        assert_eq!(convert_bin(&high_bit), 2f64.powi(63));
    }

    #[test]
    fn within_range_test_1() {
        let value_vec = vec![true, true, true, true, true, true, true, true, true, true];
        let value = convert_bin(&value_vec);

        let range = (0., 1.);
        let l = value_vec.len() as f64;
        let d = value;

        assert_eq!(within_range(range, l, d), range.1);
    }

    #[test]
    fn within_range_test_2() {
        let range = (0., 16.);
        let l = 5.0;
        let d = 30.0;

        // 30 of the 31 steps of 16 / 31, not floored to 15.
        assert!((within_range(range, l, d) - 480.0 / 31.0).abs() < 1e-12);
    }

    #[test]
    fn within_range_reaches_both_ends() {
        assert_eq!(within_range((-2.0, 2.0), 16.0, 0.0), -2.0);
        assert_eq!(within_range((-2.0, 2.0), 16.0, 65_535.0), 2.0);
        assert_eq!(within_range((0.0, 24.0), 5.0, 31.0), 24.0);
    }
}
