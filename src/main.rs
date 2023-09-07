use evolutionary_computing::{
    evolution_builder::EvolutionBuilder, fitness::Fitness, population::{Bin, GeneCod},
};

mod io;
mod population;

#[derive(Clone)]
struct MyFitness {
    value: f64,
}

impl Fitness<Bin> for MyFitness {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        if individual.0[0] {
            0.0 * self.value
        } else {
            1.0 * self.value
        }
    }
}

fn main() {
    let evolution_builer = EvolutionBuilder::new(
        10,
        10,
        GeneCod::IntPerm,
        (),
    );

    let fitness = MyFitness { value: 1.0 };

    let evolution = evolution_builer.set_fitness(fitness).build().unwrap();

    println!("{:?}", evolution.current_population);
}
