use evolutionary::prelude::*;

use radio_fitness::RadioFitness;

use crate::radio_fitness::{get_lx, get_st};

mod radio_fitness;

const L: f64 = 5.;

const RANGE_ST: (f64, f64) = (0., 24.);
const RANGE_LX: (f64, f64) = (0., 16.);
const C: f64 = -1.;

#[derive(Clone)]
struct RadioCoding;

impl Coding<Bin> for RadioCoding {
    type Output = (f64, f64);

    fn decode(&self, individual: &Bin) -> Self::Output {
        let st = get_st(individual);
        let lx = get_lx(individual);

        (st, lx)
    }
}

fn main() {
    let file_name = "examples/radios/Config.toml";

    let config = read_config(file_name).expect("Error while reading config file");

    if let GeneCod::Bin = config.gene_cod {
        let fitness = RadioFitness;

        let crossover = NPointsCrossover {
            crossover_rate: 0.8,
            n_points: 1,
        };

        let mutation = BitSwapMutation {
            mutation_rate: 0.05,
        };

        let max_runs = config.runs;

        let evolution_builder = EvolutionBuilder::from_config(config.into())
            .with_selection(TournamentSelection::default())
            .with_fitness(fitness)
            .with_crossover(crossover)
            .with_mutation(mutation)
            .with_title("Radios".to_string())
            .with_stop_condition(move |_, iterations, _| iterations > max_runs as u32)
            .with_coding(RadioCoding)
            // .set_elitism(false)
            ;

        let mut experiment = ExperimentRunner::new("radios".to_string(), 10, evolution_builder);

        experiment.run();

        // evolution.plot_chart(&"radios".to_string()).unwrap();
        //
        // let best_found = evolution.current_best();
        // println!("Best found: {:?}", best_found);
        // println!("Best found fitness: {}", evolution.current_best_fitness());
        // println!("ST: {}", get_st(&best_found));
        // println!("LX: {}", get_lx(&best_found));
        // println!("Lucro: {}", f(get_st(&best_found), get_lx(&best_found)));
    }
}
