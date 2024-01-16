// N Points Crossover works for Binary, Integer and Reals but not for Permuted Integers, so the
// implementations are in each module.

/// # N Points Crossover
///
/// Generates `n` random points to "crossover" the sections between the parents. This can be
/// a `Single-Point Crossover` using `n_points = 1` a `Two-Points Crossover` using `n_points = 2`
/// or any number you'd like.
#[derive(Clone)]
pub struct NPointsCrossover {
    /// The probability of crossover occurring.
    pub crossover_rate: f64,
    /// The number of points to crossover.
    pub n_points: usize,
}

impl Default for NPointsCrossover {
    fn default() -> Self {
        NPointsCrossover {
            crossover_rate: 0.8,
            n_points: 1,
        }
    }
}

impl NPointsCrossover {
    pub fn swap_sections<T: Copy>(vec1: &mut Vec<T>, vec2: &mut Vec<T>, start: usize, end: usize) {
        for i in start..end {
            let temp = vec1[i];
            vec1[i] = vec2[i];
            vec2[i] = temp;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn swap_sections() {
        let mut vec1 = vec![false, false, false, false, false, false];
        let mut vec2 = vec![true, true, true, true, true, true];

        super::NPointsCrossover::swap_sections(&mut vec1, &mut vec2, 2, 4);

        assert_eq!(vec1, vec![false, false, true, true, false, false]);
        assert_eq!(vec2, vec![true, true, false, false, true, true]);
    }
}
