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
rest. Every population and run is seeded, so each run measures the same work. A full run takes
about 6 minutes on an M-series laptop.

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
`crossover/bin_uniform`, `mutation/int_random_resetting`. The names are the keys of the saved
baselines, so renaming a case loses its history.

Criterion also reports the throughput in individuals per second.

## Running

```bash
cargo bench --all-features                  # everything; without serde, checkpoint.rs is skipped
cargo bench --bench crossover               # one file
cargo bench -- crossover/pmx                # the cases whose name matches
cargo bench -- 300x300                      # one size
RAYON_NUM_THREADS=1 cargo bench             # without the thread pool
```

Warm-up is 1 s and measurement 2 s, with 20 samples instead of 100 at 2000x500.
`--warm-up-time` and `--measurement-time` after `--` change the first two.

## Comparing with main

Criterion keeps baselines in `target/criterion`, on your machine only.

```bash
git switch main
cargo bench --all-features -- --save-baseline main
git switch my-branch
cargo bench --all-features -- --baseline main
```

The second run prints the change of every benchmark against main. For a PR that touches
performance, paste the comparison of the benchmarks it affects into the PR. Close other
programs while it runs: the numbers move by a few percent from one run to the next anyway.
