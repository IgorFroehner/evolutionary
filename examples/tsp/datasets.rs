use std::fs::read_to_string;

pub fn load_matrix(dataset: String) -> Vec<Vec<f64>> {
    let file = read_to_string(format!("examples/tsp/datasets/{}.in", dataset)).unwrap();

    let mut dataset = Vec::new();
    for line in file.lines() {
        let splitted_line = line.split_whitespace().collect::<Vec<&str>>();

        dataset.push(splitted_line
            .iter()
            .map(|x| x.parse::<f64>().unwrap())
            .collect::<Vec<f64>>());
    }

    dataset
}
