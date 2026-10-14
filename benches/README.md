# Benchmarks

Criterion benchmarks, split by concern so a change shows up where it happens.

| File | Measures |
|---|---|
| `generation.rs` | One `Evolution::next` of `Real`, `Bin`, `Int` and `IntPerm`, each with a cheap fitness, and of `Real` without elitism, with re-evaluated elitists, with a fitness of about 50 µs an individual and with a batch fitness. |
| `selection.rs` | Every selection, the scaled roulette and SUS included, on fitnesses that are all positive. |
| `crossover.rs` | Every crossover, on each representation it works for. |
| `mutation.rs` | Every mutation, on each representation it works for. |
| `checkpoint.rs` | Taking a checkpoint and saving it to JSON, reading it back, and resuming from it. Needs the `serde` feature. |
| `common/mod.rs` | The sizes, the fitnesses and the seeded populations they share. Not a benchmark. |

The sizes are 50x50, 300x300 and 2000x500 for a generation, and 300x300 and 2000x500 for the
rest. Every population and run is seeded, and every iteration starts from the same input: a
generation benchmark measures generation 1 of a new run each time, an operator one works on a
fresh copy of the same population. A full run takes about 7 minutes on an M-series laptop.

## Names

Every benchmark is `<group>/<case>/<population>x<genes>`, where the group is the file:

```
generation/real/300x300
selection/tournament_k5/2000x500
crossover/pmx/300x300
mutation/gaussian/300x300
checkpoint/resume/2000x500
```

A case that exists for more than one representation starts with it: `crossover/real_uniform`,
`crossover/bin_uniform`, `mutation/int_random_resetting`, `mutation/perm_swap`. The names are
the keys of the saved baselines, so renaming a case loses its history.

Criterion also reports the throughput in individuals per second.

## Running

```bash
cargo bench --all-features                          # everything; without serde, checkpoint.rs is skipped
cargo bench --bench crossover                       # one file
cargo bench -- crossover/pmx                        # the cases whose name matches
cargo bench --bench generation -- 300x300           # one size of one file
RAYON_NUM_THREADS=1 cargo bench --bench generation  # without the thread pool
cargo bench -- crossover/pmx --warm-up-time 3 --measurement-time 5
```

Warm-up is 1 s, measurement 2 s and the noise threshold 3%, with 20 samples instead of 100 at
2000x500. `--warm-up-time`, `--measurement-time` and `--noise-threshold` after `--` change the
first three.

## Threads

Every benchmark runs on rayon's pool: selection, crossover, mutation and the default fitness
are parallel. The numbers depend on the number of threads and on what else the machine is
doing, and the 50x50 generations are mostly the cost of scheduling. Set `RAYON_NUM_THREADS`
explicitly, for example to the number of performance cores, and use the same value for the
runs you compare.

## Comparing with main

Criterion keeps baselines in `target/criterion`, on your machine only.

```bash
git switch main
RAYON_NUM_THREADS=4 cargo bench --all-features -- --save-baseline main
git switch my-branch
RAYON_NUM_THREADS=4 cargo bench --all-features -- --baseline main
```

The second run prints the change of every benchmark against main. For a PR that touches
performance, paste the comparison of the benchmarks it affects into the PR. Close other
programs while it runs: the numbers move by a few percent from one run to the next anyway.

The report for a PR follows `.claude/skills/bench-report/SKILL.md`.
