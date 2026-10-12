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
/// linearly onto `range`: 0 gives exactly `range.0`, `2^l - 1` gives exactly `range.1`, the
/// values in between are spread evenly, `(range.1 - range.0) / (2^l - 1)` apart, and no result
/// falls outside of the range.
///
/// A `d` above `2^l - 1` saturates at `range.1`, and a negative one at `range.0`, instead of
/// extrapolating beyond the range. A `NaN` `d`, or a `NaN` bound, gives `NaN`, and so can an
/// infinite bound (at the other end, infinity times zero is `NaN`).
///
/// The result is not rounded: with 16 bits over `(-2.0, 2.0)` it takes 65 536 values, not the
/// five integers. Round it (`.round()` or `.floor()`) to decode an integer.
///
/// ```
/// use evolutionary::prelude::*;
///
/// assert_eq!(within_range((-2.0, 2.0), 16.0, 0.0), -2.0);
/// assert_eq!(within_range((-2.0, 2.0), 16.0, 65_535.0), 2.0);
/// // Bounds that are not exact in binary are reached too.
/// assert_eq!(within_range((-3.3, 7.1), 16.0, 65_535.0), 7.1);
/// // 2 bits over (0, 3): 0, 1, 2 and 3 give 0.0, 1.0, 2.0 and 3.0.
/// assert_eq!(within_range((0.0, 3.0), 2.0, 2.0), 2.0);
/// // Not rounded: 30 of the 31 steps of 5 bits over (0, 16).
/// assert!((within_range((0.0, 16.0), 5.0, 30.0) - 15.48).abs() < 0.01);
/// ```
pub fn within_range(range: (f64, f64), l: f64, d: f64) -> f64 {
    let steps = 2f64.powf(l) - 1.;
    let x = interpolate(range, d / steps);

    // A last guard, so that no rounding in between can leave the range (the tests find no
    // case where `interpolate` needs it), which also saturates a `d` outside of `0..=steps`.
    // Comparisons rather than `f64::max` and `min`, which would turn a `NaN` into a bound.
    let (low, high) = (range.0.min(range.1), range.0.max(range.1));
    if x < low {
        low
    } else if x > high {
        high
    } else {
        x
    }
}

/// The point at `t` (from 0 to 1) between the ends of `range`. Exact at both ends: there `t`
/// is exactly 0 or 1, so one term vanishes and the other is the bound itself, where
/// `range.0 + (range.1 - range.0) * t` rounds `range.1 - range.0` first and so misses
/// `range.1`, and can step out of the range.
fn interpolate(range: (f64, f64), t: f64) -> f64 {
    range.0 * (1.0 - t) + range.1 * t
}

#[cfg(test)]
mod test {
    use crate::utils::{convert_bin, interpolate, within_range};

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

    /// Ranges of every sign and scale, some whose bounds are not exact in binary.
    const RANGES: [(f64, f64); 12] = [
        (-3.3, 7.1),
        (-0.1, 0.2),
        (0.1, 0.2),
        (0.0, 1.0),
        (-2.0, 2.0),
        (0.0, 24.0),
        (0.0, 16.0),
        (-7.5, -0.25),
        (-1e6, -1.0),
        (1e-3, 1e3),
        (0.7, 0.7000001),
        (-1e300, 1e300),
    ];

    #[test]
    fn within_range_stays_in_the_range_and_hits_both_ends() {
        for range in RANGES {
            for bits in [1u32, 2, 3, 5, 8, 10, 12, 16] {
                let steps = 2u64.pow(bits) - 1;
                let l = f64::from(bits);

                assert_eq!(within_range(range, l, 0.0), range.0, "{range:?} {bits}");
                assert_eq!(
                    within_range(range, l, steps as f64),
                    range.1,
                    "{range:?} {bits}"
                );
                for d in 0..=steps {
                    // Unguarded, so the test checks the interpolation itself.
                    let x = interpolate(range, d as f64 / steps as f64);
                    assert!(
                        (range.0..=range.1).contains(&x),
                        "{range:?} {bits} bits, d = {d}: {x}"
                    );
                    assert_eq!(within_range(range, l, d as f64), x);
                }
            }
        }
    }

    #[test]
    fn within_range_saturates_outside_of_the_bits() {
        assert_eq!(within_range((-0.1, 0.2), 4.0, 16.0), 0.2);
        assert_eq!(within_range((-0.1, 0.2), 4.0, 1e9), 0.2);
        assert_eq!(within_range((-0.1, 0.2), 4.0, -1.0), -0.1);
        // A reversed range saturates at its ends too.
        assert_eq!(within_range((5.0, 1.0), 4.0, 100.0), 1.0);
        assert_eq!(within_range((5.0, 1.0), 4.0, -3.0), 5.0);
    }

    #[test]
    fn within_range_lets_nan_through() {
        assert!(within_range((0.0, 1.0), 4.0, f64::NAN).is_nan());
        assert!(within_range((f64::NAN, 1.0), 4.0, 3.0).is_nan());
        assert!(within_range((0.0, f64::NAN), 4.0, 15.0).is_nan());
        assert!(within_range((f64::NAN, f64::NAN), 4.0, 0.0).is_nan());
        // Infinity times zero.
        assert!(within_range((0.0, f64::INFINITY), 4.0, 0.0).is_nan());
        assert_eq!(within_range((0.0, f64::INFINITY), 4.0, 3.0), f64::INFINITY);
    }

    #[test]
    fn within_range_reaches_both_ends() {
        assert_eq!(within_range((-2.0, 2.0), 16.0, 0.0), -2.0);
        assert_eq!(within_range((-2.0, 2.0), 16.0, 65_535.0), 2.0);
        assert_eq!(within_range((0.0, 24.0), 5.0, 31.0), 24.0);
    }
}
