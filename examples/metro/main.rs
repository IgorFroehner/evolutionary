use evolutionary::prelude::*;
use std::fs;

#[derive(Clone)]
pub struct MetroFitness {
    pub adj: Vec<Vec<(usize, f64, usize)>>,
    pub dists: Vec<Vec<f64>>,
    pub start: usize,
    pub end: usize,
    pub max_time: f64,
    pub max_real_dist: f64,
}

fn get_path(
    adj: &Vec<Vec<(usize, f64, usize)>>,
    individual: &Real,
    start: usize,
    end: usize,
) -> Vec<(usize, f64, usize)> {
    let mut current = start;

    let mut path: Vec<(usize, f64, usize)> = vec![];
    let mut visited = vec![false; adj.len()];
    visited[current] = true;

    for &gene in individual.get_chromosome() {
        let possibilities = adj[current]
            .iter()
            .filter(|(i, _, _)| !visited[*i])
            .collect::<Vec<&(usize, f64, usize)>>();

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

fn calc_path_time(path: &Vec<(usize, f64, usize)>) -> f64 {
    let mut path_dist = 0.0;
    let mut line_change = 0;
    let mut last_line = 0;
    for i in 0..path.len() {
        path_dist += path[i].1;
        if i == 0 {
            last_line = path[i].2;
            continue;
        }
        if path[i].2 != last_line {
            line_change += 1;
        }
        last_line = path[i].2;
    }

    path_dist / (2.0 / 3.0) + line_change as f64 * 5.0
}

impl Fitness<Real> for MetroFitness {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        let path = get_path(&self.adj, individual, self.start, self.end);

        let path_time = calc_path_time(&path);

        let path_end = path.last().unwrap().0;

        let real_dist = if self.dists[self.end][path_end] != -1.0 {
            self.dists[self.end][path_end]
        } else {
            self.dists[path_end][self.end]
        };

        2.0 - (real_dist / self.max_real_dist) - (path_time / self.max_time)
    }
}

fn main() {
    let filename = "examples/metro/metro.in";
    let content = fs::read_to_string(filename).expect("Failed to read the file");

    let first_line = content
        .lines()
        .next()
        .unwrap()
        .split(" ")
        .collect::<Vec<&str>>();
    let n = first_line[0].parse::<usize>().unwrap();
    let m = first_line[1].parse::<usize>().unwrap();

    let mut adj = vec![vec![]; n + 5];

    for line in content.lines().skip(1) {
        let line = line.split(" ").collect::<Vec<&str>>();
        let x = line.get(0).unwrap().parse::<usize>().unwrap();
        let y = line.get(1).unwrap().parse::<usize>().unwrap();
        let c = line.get(2).unwrap().parse::<f64>().unwrap();
        let d = line.get(3).unwrap().parse::<usize>().unwrap();

        adj[x].push((y, c, d));
        adj[y].push((x, c, d));
    }

    let sum_cost = adj
        .iter()
        .map(|v| v.iter().map(|(_, c, _)| c).sum::<f64>())
        .sum::<f64>();

    let _max_time = sum_cost / 40.0 + (m as f64 - 1.0) * 5.0;

    let start = 14usize;
    let end = 7usize;

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

    let evolution_builder = EvolutionBuilder::new(30, 30,    GeneCod::Real, (0.0, 1.0))
        .with_fitness(MetroFitness {
            start,
            end,
            dists: dists.clone(),
            adj: adj.clone(),
            max_real_dist,
            max_time: sum_cost,
        })
        .with_selection(TournamentSelection::default())
        .with_crossover(UniformCrossover::default())
        .with_mutation(SubstituteMutation::default())
        .with_title("Metro".to_string())
        .with_stop_condition(move |_, iterations, _| iterations >= 100)
        // .build()
        // .unwrap()
        ;

    let mut experiment =
        ExperimentRunner::new("Metro - E14 -> E7".to_string(), 30, evolution_builder);

    experiment.run();

    experiment.experiment_digest();

    // let mut evolution = evolution_builder.build().unwrap();
    //
    // evolution.run();
    //
    // let best_found = evolution.current_best();
    //
    // let path = get_path(&adj, &best_found, start, end);
    //
    // let path_time = calc_path_time(&path);
    // let dist_in_path = path.iter().map(|(_, d, _)| d).sum::<f64>();
    //
    // println!("Path: {:?}", path);
    // println!("Distance in path: {}", dist_in_path);
    // println!("Path time: {}", path_time);
    //
    // println!("Best individual: {:?}", evolution.current_best());
    // println!("Best fitness: {}", evolution.current_best_fitness());
    //
    // // evolution.population_digest();
    // evolution.time_digest();
    //
    // evolution.plot_chart("Metro.png", "Metro E14 -> E7").unwrap();
}
