use evolutionary::prelude::*;

use fitness_max::MathFitnessMax;

mod fitness_max;
// The minimisation variant of the example, to use in place of `MathFitnessMax`.
#[allow(dead_code)]
mod fitness_min;

const L: f64 = 16.;
const RANGE: (f64, f64) = (-2., 2.);
const C_MIN: f64 = -4.;

pub fn f(x: f64) -> f64 {
    (20. * x).cos() - (x.abs() / 2.) + (x.powf(3.) / 4.)
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

        let mut evolution = EvolutionBuilder::new(population_size, dimension, ())
            .with_fitness(max_fitness)
            .with_selection(RouletteSelection)
            .with_crossover(NPointsCrossover::default())
            .with_mutation(BitFlipMutation::default())
            .with_title("Math Function".to_string())
            .with_stop_condition(move |report| report.generation >= runs)
            .build()
            .unwrap();

        evolution.run();

        let best_found = evolution.current_best();

        println!("Best found: {:?}", best_found);
        let x = within_range(RANGE, L, convert_bin(best_found.get_chromosome()));
        println!("x: {}", x);
        println!("f(x): {}", f(x));

        // evolution.plot_chart().unwrap();
    }
}
