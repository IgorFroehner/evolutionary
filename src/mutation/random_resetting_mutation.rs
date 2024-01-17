// Random Resetting mutation is implemented for Real and Int chromosome types.

/// # Substitute Mutation
///
/// For each gene in the real representation it has `mutation_rate` probability of replacing
/// the gene with a random value within the range.
#[derive(Clone)]
pub struct RandomResettingMutation {
    pub mutation_rate: f64,
}

impl Default for RandomResettingMutation {
    fn default() -> Self {
        RandomResettingMutation {
            mutation_rate: 0.05,
        }
    }
}
