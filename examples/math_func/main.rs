use evolutionary_computing::{
    config_read::{read_config, RawConfig},
    crossover::BinCrossover,
    evolution_builder::EvolutionBuilder,
    mutation::BinMutation,
    population::{GeneCod, Bin},
    selection::RouletteSelection,
    utils::{convert_bin, within_range}, coding::Coding,
};
use fitness_max::MathFitnessMax;

mod fitness_max;
mod fitness_min;

const L: f64 = 16.;
const RANGE: (f64, f64) = (-2., 2.);
const C_MAX: f64 = 2.;
const C_MIN: f64 = -4.;

pub fn f(x: f64) -> f64 {
    (20. * x).cos() - (x.abs() / 2.) + (x.powf(3.) / 4.)
}

#[derive(Clone)]
struct MathFuncCoding;

impl Coding<Bin> for MathFuncCoding {
    type Output = f64;

    fn decode(&self, individual: &Bin) -> Self::Output {
        within_range(RANGE, L, convert_bin(&individual.0))
    }
}

fn main() {
    let file_name = "examples/math_func/Config.toml";

    let RawConfig {
        population_size,
        dimension,
        gene_cod,
        runs,
        ..
    } = read_config(file_name).expect("Failed to read config file");

    if let GeneCod::Bin = gene_cod {
        let max_fitness = MathFitnessMax;

        let mut evolution = EvolutionBuilder::new(population_size, dimension, gene_cod, ())
            .with_fitness(max_fitness)
            .with_selection(RouletteSelection::default())
            .with_crossover(BinCrossover::default())
            .with_mutation(BinMutation::default())
            .with_title("Math Function".to_string())
            .with_stop_condition(move |_, iterations, _| iterations >= runs as u32)
            .with_coding(MathFuncCoding)
            .build()
            .unwrap();

        evolution.run();

        let best_found = evolution.current_best();

        println!("Best found: {:?}", best_found);
        let x = within_range(RANGE, L, convert_bin(&best_found.0));
        println!("x: {}", x);
        println!("f(x): {}", f(x));

        // evolution.plot_chart().unwrap();
    }
}
