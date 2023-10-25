# Maze Example

#### [Go to the implementation](./main.rs)

This example shows how to use the `evolutionary` library to solve the Maze problem.

The maze problem is basically to find a valid path between the entry and the end in a giving maze. The maze input
is a matrix of integers as follows ([maze example](./maze.rs)):

* 0 is a wall
* 1 is a free cell
* 2 is the entry
* 3 is the end

### Coding

This implementation was done through the Real coding, where at each index $i$ value corresponds to the $choice_i$ 
the actor will do where they are at step $i$. It calculates the possibility at a giving position looking around and
checking if it's a wall or a free cell. It also doesn't enable the actor to pass again in a already visited cell.

### Fitness

The fitness is `max_dist - dist`.

## Running the Example
```bash
cargo run --example maze
```
