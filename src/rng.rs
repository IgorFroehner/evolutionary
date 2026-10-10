use rand::RngCore;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

/// The random number generators of one step of a generation, handed to every [`Selection`],
/// [`Crossover`] and [`Mutation`].
///
/// Each unit of work (an individual, a pair of parents, a slot of the mating pool) asks for its
/// own generator with [`rng`](Self::rng), passing its index. The generator depends only on the
/// seed of the run, the generation, the step and that index, so an operator that runs in
/// parallel gives the same result however rayon schedules its work.
///
/// # Example
///
/// ```
/// use evolutionary::prelude::*;
/// use rayon::prelude::*;
///
/// #[derive(Clone)]
/// struct FlipFirstGene;
///
/// impl Mutation<Bin> for FlipFirstGene {
///     fn mutate(&self, population: &mut [Bin], rngs: &Rngs) {
///         population
///             .par_iter_mut()
///             .enumerate()
///             .for_each(|(index, individual)| {
///                 let mut rng = rngs.rng(index);
///                 if rng.random_bool(0.5) {
///                     individual.set_gene(0, !individual.get_gene(0));
///                 }
///             });
///     }
/// }
/// ```
///
/// [`Selection`]: crate::Selection
/// [`Crossover`]: crate::Crossover
/// [`Mutation`]: crate::Mutation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rngs {
    seed: u64,
    generation: u32,
    step: Step,
}

/// Which part of a generation the generators are for, so the steps of one generation never
/// share a stream. The values are part of every generator's key: changing one changes every
/// seeded run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// Generators built by [`Rngs::new`], outside of an evolution.
    Standalone = 0,
    Initialization = 1,
    Selection = 2,
    Crossover = 3,
    Mutation = 4,
    /// Seeds of the runs of an [`ExperimentRunner`](crate::prelude::ExperimentRunner).
    Experiment = 5,
}

impl Rngs {
    /// Generators for `seed`, for testing an operator or running one outside of an evolution.
    pub fn new(seed: u64) -> Self {
        Self::for_step(seed, 0, Step::Standalone)
    }

    pub(crate) fn for_step(seed: u64, generation: u32, step: Step) -> Self {
        Self {
            seed,
            generation,
            step,
        }
    }

    /// The generator for the unit of work at `index`. Calling it twice with the same index
    /// returns two generators that produce the same numbers.
    pub fn rng(&self, index: usize) -> TaskRng {
        // Every input gets its own bytes of the ChaCha key, so no two of them share a stream.
        let mut key = [0u8; 32];
        key[0..8].copy_from_slice(&self.seed.to_le_bytes());
        key[8..12].copy_from_slice(&self.generation.to_le_bytes());
        key[12..16].copy_from_slice(&(self.step as u32).to_le_bytes());
        key[16..24].copy_from_slice(&(index as u64).to_le_bytes());

        TaskRng(ChaCha8Rng::from_seed(key))
    }
}

/// A random number generator for one unit of work, made by [`Rngs::rng`]. Use it through the
/// [`Rng`](rand::Rng) trait.
///
/// It is a ChaCha8 generator, so the numbers it produces are the same on every platform.
#[derive(Clone, Debug)]
pub struct TaskRng(ChaCha8Rng);

impl RngCore for TaskRng {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
        self.0.fill_bytes(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn draws(mut rng: TaskRng) -> Vec<u64> {
        (0..4).map(|_| rng.random()).collect()
    }

    #[test]
    fn same_index_same_numbers() {
        let rngs = Rngs::new(7);

        assert_eq!(draws(rngs.rng(3)), draws(rngs.rng(3)));
    }

    #[test]
    fn every_input_changes_the_numbers() {
        let base = draws(Rngs::for_step(7, 1, Step::Selection).rng(0));

        assert_ne!(base, draws(Rngs::for_step(8, 1, Step::Selection).rng(0)));
        assert_ne!(base, draws(Rngs::for_step(7, 2, Step::Selection).rng(0)));
        assert_ne!(base, draws(Rngs::for_step(7, 1, Step::Mutation).rng(0)));
        assert_ne!(base, draws(Rngs::for_step(7, 1, Step::Selection).rng(1)));
    }

    #[test]
    fn numbers_are_stable_across_versions() {
        // Changing how generators are derived changes every seeded run, which is a breaking
        // change for anyone replaying one. Update this value only on purpose.
        let first: u64 = Rngs::new(42).rng(0).random();

        assert_eq!(first, FIRST_DRAW_OF_SEED_42);
    }

    const FIRST_DRAW_OF_SEED_42: u64 = 6424161053832095879;
}
