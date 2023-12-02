use std::fs;
use evolutionary::experiment_runner::ExperimentResult;
use evolutionary::prelude::*;

#[derive(Clone)]
pub struct MetroFitness {
    pub adj: Vec<Vec<(usize, f64)>>,
    pub dists: Vec<Vec<f64>>,
    pub start: usize,
    pub end: usize,
    pub max_cost: f64,
    pub max_real_dist: f64,
}

fn get_path(adj: &Vec<Vec<(usize, f64)>>, individual: &Real, start: usize, end: usize) -> Vec<(usize, f64)> {
    let mut current = start;

    let mut path: Vec<(usize, f64)> = vec![];
    let mut visited = vec![false; adj.len()];
    path.push((current, 0.0));
    visited[current] = true;

    for &gene in individual.get_chromosome() {
        let possibilities = adj[current]
            .iter()
            .filter(|(i, _)| !visited[*i])
            .collect::<Vec<&(usize, f64)>>();

        if possibilities.is_empty() {
            break;
        }

        let frac = 1.0 / possibilities.len() as f64;
        let index = (gene / frac).floor() as usize;
        let a = possibilities.get(index).or(possibilities.first()).unwrap();

        current = a.0;
        path.push(**a);

        visited[current] = true;
        if current == end {
            break;
        }
    }

    path
}

impl Fitness<Real> for MetroFitness {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        let path = get_path(&self.adj, individual, self.start, self.end);

        let path_cost = path.iter().map(|(_, c)| c).sum::<f64>();

        let path_end = path.last().unwrap().0;

        let real_dist = if self.dists[self.end][path_end] != -1.0 {
            self.dists[self.end][path_end]
        } else {
            self.dists[path_end][self.end]
        };

        self.max_cost + self.max_real_dist - (real_dist / self.max_real_dist) - (path_cost / self.max_cost)
    }
}

#[derive(Clone)]
struct MetroCoding;

impl Coding<Real> for MetroCoding {
    type Output = f64;

    fn decode(&self, individual: &Real) -> Self::Output {
        0.0
    }
}

fn main() {
    let filename = "examples/metro/metro.in";
    let content = fs::read_to_string(filename).expect("Failed to read the file");

    let first_line = content.lines().next().unwrap().split(" ").collect::<Vec<&str>>();
    let n = first_line[0].parse::<usize>().unwrap();
    let m = first_line[1].parse::<usize>().unwrap();

    let mut adj = vec![vec![]; n + 5];

    for line in content.lines().skip(1) {
        let line = line.split(" ").collect::<Vec<&str>>();
        let x = line.get(0).unwrap().parse::<usize>().unwrap();
        let y = line.get(1).unwrap().parse::<usize>().unwrap();
        let c = line.get(2).unwrap().parse::<f64>().unwrap();

        adj[x].push((y, c));
        adj[y].push((x, c));
    }

    let sum_cost = adj.iter().map(|v| v.iter().map(|(_, c)| c).sum::<f64>()).sum::<f64>();

    let start = 6usize;
    let end = 11usize;

    println!("n: {}, m: {}", n, m);
    println!("adj: {:?}", adj);
    println!("start: {}, end: {}", start, end);

    let dists_file = "examples/metro/dists.in";
    let dists_content = fs::read_to_string(dists_file).expect("Failed to read the file");

    let mut dists = vec![vec![-1.0; n + 5]; n + 5];
    let mut max_real_dist: f64 = -1.0;

    let mut i = 1;
    for line in dists_content.lines() {
        let line = line.split(",").collect::<Vec<&str>>();

        let mut j = 1;
        for n in line {
            if n != "-" {
                let a = n.parse::<f64>();
                if a.is_ok() {
                    dists[i][j] = a.unwrap();
                    max_real_dist = max_real_dist.max(dists[i][j]);
                }
            }
            j += 1;
        }
        i += 1;
    }

    let mut evolution_builder = EvolutionBuilder::new(30, 30,    GeneCod::Real, (0.0, 1.0))
        .with_fitness(MetroFitness {
            start,
            end,
            dists: dists.clone(),
            adj: adj.clone(),
            max_real_dist,
            max_cost: sum_cost,
        })
        .with_selection(TournamentSelection::default())
        .with_crossover(UniformCrossover::default())
        .with_mutation(SubstituteMutation::default())
        .with_title("Metro".to_string())
        .with_stop_condition(move |_, iterations, _| iterations >= 30)
        .with_coding(MetroCoding)
        // .build()
        // .unwrap()
        ;

    let mut experiment = ExperimentRunner::new("Metro - E6 -> E11".to_string(), 30, evolution_builder);

    experiment.run();

    experiment.experiment_digest();

    // let mut evolution = evolution_builder.build().unwrap();

    // evolution.run();

    // let best_found = evolution.current_best();
    //
    // let path = get_path(&adj, &best_found, start, end);
    //
    // println!("Path: {:?}", path);
    //
    // let nro_nodes = path.len() - 2;
    //
    // let path_cost = path.iter().map(|(_, c)| c).sum::<f64>();
    // println!("Path cost: {}", path_cost);
    //
    // let speed = 40.0 / 60.0;
    // let path_in_minutes = path_cost / speed;
    // println!("Path in minutes: {} minutes", path_in_minutes + nro_nodes as f64 * 5.0);
    //
    // println!("Best individual: {:?}", evolution.current_best());
    // println!("Best fitness: {}", evolution.current_best_fitness());
    //
    // // evolution.population_digest();
    // evolution.time_digest();
    //
    // evolution.plot_chart("Metro.png", "Metro E14 -> E7").unwrap();
}
