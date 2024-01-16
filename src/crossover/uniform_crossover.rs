// The Uniform Crossover works for Binary, Integer and Reals but not for Permuted Integers,
// so the implementations are in each module.

/// # Uniform Crossover
///
/// For each gene, it selects whether to swap them between the parents based on the `toss_probability`.
#[derive(Clone)]
pub struct UniformCrossover {
    /// The probability of crossover occurring.
    pub crossover_rate: f64,
    /// The probability of swapping the genes.
    pub toss_probability: f64,
}

impl Default for UniformCrossover {
    fn default() -> Self {
        UniformCrossover {
            crossover_rate: 0.8,
            toss_probability: 0.5,
        }
    }
}
