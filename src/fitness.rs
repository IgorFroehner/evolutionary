use dyn_clone::DynClone;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

use crate::population::Individual;

/// Trait that defines the fitness function and that will be used by the evolutionary algorithm.
/// You need to implement your own fitness function for each problem you're trying to solve. The
/// fitness function is directly related to the type of individual you're using and how your
/// mapping the answers.
///
/// Example:
///
/// ```
/// use evolutionary::prelude::*;
///
/// #[derive(Clone)]
/// struct SumFitness;
///
/// impl Fitness<Bin> for SumFitness {
///     fn calculate_fitness(&self, individual: &Bin) -> f64 {
///         let mut sum = 0.0;
///
///         for i in 0..individual.get_chromosome().len() {
///             if individual.get_gene(i) {
///                 sum += 1.0;
///             }
///         }
///
///         sum
///     }
/// }
/// ```
///
/// The evolution maximizes the fitness: the best individual is the one with the highest, and
/// [`GenerationReport::best_fitness`](crate::GenerationReport::best_fitness) is the highest
/// value. To minimize something (an error, a cost, a time), wrap the function that computes it
/// in [`Minimize`] instead of negating it by hand.
///
/// # Evaluating the whole population at once
///
/// The evolution evaluates a generation with one call to
/// [`calculate_population`](Self::calculate_population), which by default calls
/// [`calculate_fitness`](Self::calculate_fitness) on every individual in parallel. Override it
/// when the population is cheaper to evaluate together: a simulation that steps every
/// individual at once, a batched forward pass, a GPU kernel.
///
/// # Randomness
///
/// A seeded run (see [`EvolutionBuilder::with_seed`](crate::EvolutionBuilder::with_seed))
/// replays only if the fitness is deterministic: the same individual must get the same number
/// every time, whatever the thread that evaluates it. Do not draw from `rand::random()`,
/// `rand::rng()` or a generator shared between threads; the fitness gets no [`Rngs`](crate::Rngs)
/// of its own.
///
/// A noisy fitness can still replay: derive its noise from a hash of the chromosome, with a
/// hash whose output is fixed (not [`DefaultHasher`](std::hash::DefaultHasher), which may
/// change between Rust versions). Noise from the chromosome alone is the same at every
/// evaluation, though. For noise that changes from one generation to the next, as
/// [`EvolutionBuilder::with_reevaluate_elitists`](crate::EvolutionBuilder::with_reevaluate_elitists)
/// needs to be of any use, count the generations in `calculate_population`, which the
/// evolution calls once per generation, from the thread that runs it, and mix the count into
/// the hash. The same count can pick a scenario per generation (a track, a set of test cases)
/// shared by the whole population:
///
/// ```
/// use evolutionary::prelude::*;
/// use evolutionary::rayon::prelude::*;
/// use std::sync::atomic::{AtomicU64, Ordering};
///
/// /// The sum of the genes, plus noise in `[-0.5, 0.5)` that changes every generation.
/// struct NoisySum {
///     generation: AtomicU64,
/// }
///
/// impl Clone for NoisySum {
///     fn clone(&self) -> Self {
///         let generation = self.generation.load(Ordering::Relaxed);
///         NoisySum { generation: AtomicU64::new(generation) }
///     }
/// }
///
/// /// SplitMix64's finalizer: a hash that gives the same numbers in every Rust version.
/// fn mix(mut x: u64) -> u64 {
///     x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
///     x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
///     x ^ (x >> 31)
/// }
///
/// impl Fitness<Real> for NoisySum {
///     fn calculate_fitness(&self, individual: &Real) -> f64 {
///         let genes = individual.get_chromosome();
///         let generation = self.generation.load(Ordering::Relaxed);
///         let hash = genes.iter().fold(mix(generation), |hash, gene| mix(hash ^ gene.to_bits()));
///         let noise = (hash >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
///
///         genes.iter().sum::<f64>() + noise
///     }
///
///     fn calculate_population(&self, population: &mut [Real]) {
///         self.generation.fetch_add(1, Ordering::Relaxed);
///         population.par_iter_mut().for_each(|individual| {
///             let fitness = self.calculate_fitness(individual);
///             individual.set_fitness(fitness);
///         });
///     }
/// }
///
/// let builder = EvolutionBuilder::new(20, 5, (-1.0, 1.0))
///     .with_fitness(NoisySum { generation: AtomicU64::new(0) })
///     .with_selection(TournamentSelection::default())
///     .with_crossover(UniformCrossover::default())
///     .with_mutation(GaussianMutation::default())
///     .with_reevaluate_elitists(true)
///     .with_seed(7)
///     .with_stop_condition(|report| report.generation >= 30);
///
/// // Each build clones the fitness the builder holds, with its count still at zero.
/// let mut first = builder.build().unwrap();
/// let mut second = builder.build().unwrap();
/// first.run();
/// second.run();
///
/// assert_eq!(first.metrics.best_fitnesses, second.metrics.best_fitnesses);
/// ```
///
/// The count must live in the fitness value itself and be copied by its `Clone`, as above,
/// never kept behind an `Arc` or in a `static`. Every [`Evolution`](crate::Evolution) gets its
/// own clone of the fitness, and a shared count would be advanced by every run at once: two
/// identical seeded [`ExperimentRunner`](crate::prelude::ExperimentRunner)s, whose runs go in
/// parallel, would then give different histories.
///
/// Because the count lives in the fitness, a second [`start`](crate::Evolution::start) of the
/// same `Evolution` carries on from where the first run left it: build a new `Evolution` from
/// the builder to replay a run.
///
/// A run resumed with
/// [`EvolutionBuilder::with_checkpoint`](crate::EvolutionBuilder::with_checkpoint) gets a new
/// clone of the fitness too, and does not evaluate the checkpoint's population again. To go on
/// with the same noise, start the count where the saved run had it: one call for the initial
/// population and one per generation since, so `checkpoint.generation + 1` in the example
/// above.
///
/// A fitness of this kind has no seed of its own, so two runs with different seeds see the
/// same noise in the same generation; put a seed of your own in the fitness if that matters.
pub trait Fitness<T: Individual>: 'static + DynClone + Send + Sync {
    /// Method that calculates the fitness of an individual.
    fn calculate_fitness(&self, individual: &T) -> f64;

    /// Calculates the fitness of every individual in `population` and stores it with
    /// [`Individual::set_fitness`]. The evolution calls it once per generation (and once for the
    /// initial population), with every individual that needs a fitness.
    ///
    /// The default calls [`calculate_fitness`](Self::calculate_fitness) on each individual, in
    /// parallel with rayon. Override it to evaluate the population as a batch. An override must
    /// set the fitness of every individual: one it skips keeps the fitness it had, which for a
    /// new child is its parent's.
    ///
    /// The order of `population` is fixed, so a fitness that depends on positions (a batch
    /// split into fixed lanes, noise keyed by index) still replays:
    ///
    /// - For the initial population, the individuals given to
    ///   [`EvolutionBuilder::with_initial_population`](crate::EvolutionBuilder::with_initial_population)
    ///   come first, in their order, then the random ones.
    /// - For every later generation, the new individuals come first, in the order of the mating
    ///   pool after crossover and mutation. With
    ///   [`EvolutionBuilder::with_reevaluate_elitists`](crate::EvolutionBuilder::with_reevaluate_elitists)
    ///   on, the elitists of the previous generation follow, at most as many as there are new
    ///   individuals, in the order elitism found them: not sorted by fitness, and the same in
    ///   every replay built with the same Rust version (elitism partitions the population with
    ///   `select_nth_unstable_by`, whose order the standard library may change).
    ///
    /// The new individuals are the same, at the same positions, with re-evaluation on or off.
    ///
    /// # Example
    ///
    /// ```
    /// use evolutionary::prelude::*;
    ///
    /// #[derive(Clone)]
    /// struct BatchedSum;
    ///
    /// impl Fitness<Real> for BatchedSum {
    ///     fn calculate_fitness(&self, individual: &Real) -> f64 {
    ///         individual.get_chromosome().iter().sum()
    ///     }
    ///
    ///     fn calculate_population(&self, population: &mut [Real]) {
    ///         // One pass over the whole population, as a batched evaluation would do.
    ///         let sums: Vec<f64> = population
    ///             .iter()
    ///             .map(|individual| individual.get_chromosome().iter().sum())
    ///             .collect();
    ///
    ///         for (individual, sum) in population.iter_mut().zip(sums) {
    ///             individual.set_fitness(sum);
    ///         }
    ///     }
    /// }
    /// ```
    fn calculate_population(&self, population: &mut [T]) {
        population.par_iter_mut().for_each(|individual| {
            let fitness = self.calculate_fitness(individual);
            individual.set_fitness(fitness);
        });
    }
}

/// Turns a value to minimize into a fitness, by negating it: the individual with the smallest
/// value gets the highest fitness, and the evolution, the stop condition and the reports keep
/// maximizing.
///
/// Every fitness the evolution reports is the negated value, so negate it again to read the
/// value itself: `-report.best_fitness`. Negation is exact, so that gives back exactly what
/// the wrapped function returned. A `NaN` stays `NaN`, the worst fitness either way.
///
/// The fitnesses are negative when the values are positive, so pair `Minimize` with a
/// selection that ranks individuals rather than weighs them (`TournamentSelection`,
/// `RankSelection`), or with a scaled fitness-proportional one
/// ([`RouletteSelection::scaled`](crate::prelude::RouletteSelection::scaled)).
///
/// ```
/// use evolutionary::prelude::*;
///
/// /// The distance of the genes from the origin, to bring down to 0.
/// #[derive(Clone)]
/// struct Distance;
///
/// impl Fitness<Real> for Distance {
///     fn calculate_fitness(&self, individual: &Real) -> f64 {
///         individual.chromosome.iter().map(|gene| gene * gene).sum::<f64>().sqrt()
///     }
/// }
///
/// let mut evolution = EvolutionBuilder::new(50, 3, (-10.0, 10.0))
///     .with_fitness(Minimize(Distance))
///     .with_selection(TournamentSelection::default())
///     .with_crossover(UniformCrossover::default())
///     .with_mutation(GaussianMutation::new(0.3, 0.1))
///     .with_seed(1)
///     .with_stop_condition(|report| -report.best_fitness < 0.5 || report.generation >= 500)
///     .build()
///     .unwrap();
///
/// evolution.run();
///
/// let distance = -evolution.report().best_fitness;
/// assert!(distance < 0.5);
/// assert_eq!(distance, Distance.calculate_fitness(evolution.current_best()));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Minimize<F>(pub F);

impl<T: Individual, F: Fitness<T> + Clone> Fitness<T> for Minimize<F> {
    fn calculate_fitness(&self, individual: &T) -> f64 {
        -self.0.calculate_fitness(individual)
    }

    /// Hands the population to the wrapped fitness's own `calculate_population`, so a batched
    /// fitness stays batched, then negates what it stored. The fitnesses are negated before the
    /// call too, so the wrapped fitness sees raw values throughout, and one it leaves unset keeps
    /// its fitness exactly.
    fn calculate_population(&self, population: &mut [T]) {
        negate_fitnesses(population);
        self.0.calculate_population(population);
        negate_fitnesses(population);
    }
}

fn negate_fitnesses<T: Individual>(population: &mut [T]) {
    for individual in population.iter_mut() {
        individual.set_fitness(-individual.get_fitness());
    }
}

/// Total order over fitness values where `NaN` ranks below every number, so an individual whose
/// fitness could not be computed is always treated as the worst one instead of panicking a
/// comparison.
pub(crate) fn fitness_cmp(a: f64, b: f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a.total_cmp(&b),
    }
}

/// Weight of an individual for fitness-proportional selection. `NaN` gets no chance of being
/// picked, the same as the worst fitness.
pub(crate) fn selection_weight(fitness: f64) -> f64 {
    if fitness.is_nan() {
        0.0
    } else {
        fitness
    }
}

/// Reads the fitness of a saved individual, taking `null` as `NaN`.
///
/// An individual that was never evaluated has a `NaN` fitness, and JSON has no `NaN`:
/// `serde_json` writes it as `null`, which a plain `f64` refuses to read back. Formats that
/// are not human-readable (bincode and the like) store the `f64` as it is, so they read it as
/// one too, and the format of every saved individual stays the same.
///
/// `serde_json` writes `+inf` and `-inf` as `null` too, and there is no telling them apart
/// from a `NaN` once written: an infinite fitness saved to JSON loads back as `NaN`, silently,
/// where the file failed to load before. Binary formats keep infinities.
#[cfg(feature = "serde")]
pub(crate) fn deserialize_fitness<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct FitnessVisitor;

    impl<'de> serde::de::Visitor<'de> for FitnessVisitor {
        type Value = f64;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a fitness: a number, or null for one never evaluated")
        }

        fn visit_f64<E>(self, value: f64) -> Result<f64, E> {
            Ok(value)
        }

        fn visit_i64<E>(self, value: i64) -> Result<f64, E> {
            Ok(value as f64)
        }

        fn visit_u64<E>(self, value: u64) -> Result<f64, E> {
            Ok(value as f64)
        }

        fn visit_unit<E>(self) -> Result<f64, E> {
            Ok(f64::NAN)
        }

        fn visit_none<E>(self) -> Result<f64, E> {
            Ok(f64::NAN)
        }

        fn visit_some<D>(self, deserializer: D) -> Result<f64, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            deserializer.deserialize_any(self)
        }
    }

    if deserializer.is_human_readable() {
        deserializer.deserialize_any(FitnessVisitor)
    } else {
        deserializer.deserialize_f64(FitnessVisitor)
    }
}

/// Reads a saved fitness history, taking each `null` as `NaN`, as
/// [`deserialize_fitness`] does for one fitness: the average of a generation where no fitness
/// is a number is `NaN`.
#[cfg(feature = "serde")]
pub(crate) fn deserialize_fitnesses<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct SavedFitness(f64);

    impl<'de> serde::Deserialize<'de> for SavedFitness {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            deserialize_fitness(deserializer).map(SavedFitness)
        }
    }

    let fitnesses: Vec<SavedFitness> = serde::Deserialize::deserialize(deserializer)?;

    Ok(fitnesses.into_iter().map(|fitness| fitness.0).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::population::Real;
    use std::cmp::Ordering;

    #[test]
    fn nan_ranks_below_every_number() {
        assert_eq!(fitness_cmp(f64::NAN, f64::NEG_INFINITY), Ordering::Less);
        assert_eq!(fitness_cmp(f64::NEG_INFINITY, f64::NAN), Ordering::Greater);
        assert_eq!(fitness_cmp(f64::NAN, f64::NAN), Ordering::Equal);
        assert_eq!(fitness_cmp(-f64::NAN, 0.0), Ordering::Less);
    }

    #[test]
    fn numbers_keep_their_order() {
        assert_eq!(fitness_cmp(1.0, 2.0), Ordering::Less);
        assert_eq!(fitness_cmp(2.0, 1.0), Ordering::Greater);
        assert_eq!(fitness_cmp(1.0, 1.0), Ordering::Equal);
    }

    /// The first gene, as a value to minimize.
    #[derive(Clone)]
    struct FirstGene;

    impl Fitness<Real> for FirstGene {
        fn calculate_fitness(&self, individual: &Real) -> f64 {
            individual.chromosome[0]
        }
    }

    #[test]
    fn minimize_ranks_the_smaller_value_first() {
        let small = Real::new(vec![-3.5], (-10.0, 10.0));
        let big = Real::new(vec![2.0], (-10.0, 10.0));
        let minimize = Minimize(FirstGene);

        let (small, big) = (
            minimize.calculate_fitness(&small),
            minimize.calculate_fitness(&big),
        );

        assert_eq!(fitness_cmp(small, big), Ordering::Greater);
        assert_eq!((-small, -big), (-3.5, 2.0));
    }

    /// A batch fitness that counts its calls, skips its last individual and gives the others
    /// their first gene plus the number of individuals in the batch.
    #[derive(Clone, Default)]
    struct Batched {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Fitness<Real> for Batched {
        fn calculate_fitness(&self, _individual: &Real) -> f64 {
            panic!("a batch fitness evaluates through calculate_population");
        }

        fn calculate_population(&self, population: &mut [Real]) {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let size = population.len() as f64;
            let last = population.len() - 1;
            for individual in &mut population[..last] {
                individual.set_fitness(individual.chromosome[0] + size);
            }
        }
    }

    #[test]
    fn minimize_hands_the_population_to_a_batch_fitness() {
        let batched = Batched::default();
        let mut population: Vec<Real> = [1.0, -2.0, 7.0]
            .iter()
            .map(|&gene| Real::new(vec![gene], (-10.0, 10.0)))
            .collect();
        population[2].set_fitness(-4.5);

        Minimize(batched.clone()).calculate_population(&mut population);

        assert_eq!(batched.calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        let fitnesses: Vec<f64> = population.iter().map(|i| i.get_fitness()).collect();
        // The one the batch skipped keeps its fitness exactly.
        assert_eq!(fitnesses, vec![-4.0, -1.0, -4.5]);
    }

    #[test]
    fn minimize_makes_the_smallest_value_the_best_of_a_run() {
        use crate::prelude::*;

        let mut evolution = EvolutionBuilder::new(30, 1, (-10.0, 10.0))
            .with_fitness(Minimize(FirstGene))
            .with_selection(TournamentSelection::default())
            .with_crossover(DoNothingCrossover)
            .with_mutation(GaussianMutation::new(1.0, 0.5))
            .with_seed(5)
            .with_stop_condition(|report| report.generation >= 30)
            .build()
            .unwrap();
        evolution.run();

        let smallest = evolution
            .population()
            .iter()
            .map(|individual| individual.chromosome[0])
            .fold(f64::INFINITY, f64::min);
        assert_eq!(evolution.current_best().chromosome[0], smallest);
        assert_eq!(-evolution.current_best_fitness(), smallest);
        // The run pushed the value down, below anything the initial range allowed.
        assert!(smallest < -10.0, "{smallest}");
    }

    #[test]
    fn nan_has_no_selection_weight() {
        assert_eq!(selection_weight(f64::NAN), 0.0);
        assert_eq!(selection_weight(3.0), 3.0);
    }
}
