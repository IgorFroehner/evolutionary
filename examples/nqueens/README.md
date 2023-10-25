# N Queens Example

#### [Go to the implementation](./main.rs)

This example shows how to use the `evolutionary` library to solve the N Queens problem.

The N Queens problem is a classic problem in which you have to place N queens on a NxN chessboard such that no queen 
can attack another queen. This means that no two queens can be on the same row, column, or diagonal.

### Coding

This implemetation was done through the Permuted Integers coding, where in the chromosome every index is the column and
each value is the line. This already guarantees that no queen's will be in the same line or column, this way the
experiment must find only a configuration where the queens can't attack each other in the diagonals.

### Fitness

The fitness is `max_collisions - n_collisions`.

## Running the Example
```bash
cargo run --example nqueens
```
