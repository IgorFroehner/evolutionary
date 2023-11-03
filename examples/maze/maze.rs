use std::fs;

pub fn read_matrix_from_file(filename: &str) -> Vec<Vec<i32>> {
    // Read the content of the file
    let content = fs::read_to_string(filename).expect("Failed to read the file");

    // Split the content into lines and map each line to a vector of integers
    content
        .lines()
        .map(|line| {
            line.chars()
                .map(|ch| {
                    ch.to_digit(10)
                        .expect("Failed to convert character to digit") as i32
                })
                .collect()
        })
        .collect()
}
