---
name: bench-report
description: Produce a before-and-after criterion benchmark report for a PR of this crate and paste it into the PR. Use for any change under src/ that touches evolution.rs, a selection, crossover or mutation operator, a representation, fitness evaluation or the RNG (rng.rs), for a bump of rand, rand_distr or rayon, and whenever asked to benchmark, compare performance, or do a before-and-after.
---

# Bench report

The report compares the PR's base with its head on the criterion suite in `benches/` (layout,
names and settings in `benches/README.md`). `compare.py`, next to this file, renders it from
criterion's own output. The report is the record of whether a change kept or improved
performance: follow every step, and report what you did not do.

## 1. Pick the groups

Run only the groups the change touches. Criterion's filter, after `--`, is a regex on the id
`<group>/<case>/<pop>x<genes>`.

| The change touches | Run |
|---|---|
| one file of `src/crossover/`, `src/mutation/` or `src/selection/` | that group, narrowed to the cases of that operator: `--bench crossover`, `crossover/pmx/` |
| the `mod.rs` or the trait of one of those directories | the whole group |
| `src/fitness.rs`, fitness evaluation, `src/metrics.rs` | `--bench generation` |
| `src/population/<repr>.rs` | `generation/<repr>` (no trailing slash, so `real` also runs the `real_*` variants) and that representation's cases in `crossover` and `mutation`; for `real.rs` also `selection` and `checkpoint` |
| `src/population/mod.rs` (`Individual`) | everything |
| `src/evolution_builder.rs` | `--bench generation` and `checkpoint/resume/` |
| `src/checkpoint.rs` | `--all-features --bench checkpoint` |
| `src/utils.rs` | nothing measures it: say so, no report |
| `src/evolution.rs`, `src/rng.rs`, a bump of `rand`, `rand_distr` or `rayon` in `Cargo.toml` or `Cargo.lock`, a release | everything |

A shared operator file maps to the case of every representation it runs on:
`src/crossover/uniform_crossover.rs` is `real_uniform`, `bin_uniform` and `int_uniform`;
`swap_mutation.rs` is `real_swap`, `bin_swap`, `int_swap` and `perm_swap`. Check the bench file
for the cases that use the operator. Any run that includes `checkpoint` needs `--all-features`.

Every run also includes one control case that the change cannot affect:
`mutation/bin_swap/300x300`, with `--bench mutation`. It runs on the rayon pool, so it slows
down when something else takes the performance cores. When the change touches
`src/mutation/swap_mutation.rs`, `src/mutation/mod.rs` or `src/population/bin.rs`, use
`checkpoint/from_json/300x300` instead, which only parses JSON and needs `--all-features` and
`--bench checkpoint`; it is also the control of a full run. A run of several groups or cases
is one quoted regex, so a crossover change runs:

```bash
cargo bench --bench crossover --bench mutation -- 'crossover/(pmx|real_uniform)/|mutation/bin_swap/300x300'
```

A full run, `cargo bench --all-features` with no filter, includes `checkpoint/from_json/300x300`
already and takes about 7 minutes, so a full report about 15. One group takes one to two
minutes per run. Write down what you skip: the report says so.

## 2. Preconditions

- `git status --porcelain` prints nothing. Commit or set aside your changes first.
- Run `cargo test --test golden` on the branch. If it fails, the seeded results changed (for
  example an RNG stream, or the elitism order): see step 6.
- The machine is awake and on mains power. On macOS `pmset -g batt | head -1` says
  `AC Power`, and each `cargo bench` runs under `caffeinate -i`. On Linux,
  `/sys/class/power_supply/*/online` is 1, and it runs under `systemd-inhibit`.
- The machine is quiet, checked right before and right after each run (step 3 and 4):
  `ps -Ao pcpu,comm -r | head -5` shows no process above 50% CPU other than cargo and the bench
  itself, and the 1-minute load in `uptime` is below `RAYON_NUM_THREADS`. The run itself
  raises the load, so after it, wait for the load to fall back:

  ```bash
  until uptime | awk '{ exit !($(NF-2) + 0 < 4) }'; do sleep 5; done; uptime
  ```

  With a timeout of 2 minutes: if it does not get there, something else ran, so redo the run.
  You cannot close other sessions: if it is not quiet before a run, wait for it the same way,
  for at most 2 minutes. If it is still not quiet, stop and write "could not get a quiet
  machine" in the PR instead of a table.
- Both runs on the same machine, in the same session, one after the other.
- The same environment for both runs: `RAYON_NUM_THREADS=4 CARGO_INCREMENTAL=0` on each
  `cargo bench`, as below. Keep 4, at most the performance cores
  (`sysctl -n hw.perflevel0.physicalcpu` on macOS), unless the PR says why.
- The criterion settings of `benches/common/mod.rs`. If you pass `--warm-up-time`,
  `--measurement-time` or `--noise-threshold`, pass the same to both runs and report them.

Shell variables, `export` and `cd` may not survive between an agent's tool calls, so each
command below sets its own directory and environment. Run these one by one and write the
values literally where the commands below have `<...>`:

```bash
git fetch origin
git merge-base HEAD origin/main         # the base commit
git rev-parse --short <the base commit> # <base>
git rev-parse --short HEAD              # <head>
git rev-parse --show-toplevel           # <branch checkout>
git diff --stat <base> HEAD -- Cargo.toml Cargo.lock
```

For a PR stacked on another branch, use `origin/<that branch>` instead of `origin/main`.
`<target>` is `<branch checkout>/target`, or `$CARGO_TARGET_DIR` if that is set. If the last
command prints anything, the dependencies differ: the comparison is still valid, that is what
it measures, but the report says that the lockfile differed.

Empty criterion's results, so nothing from another run or day can end up in the report:

```bash
rm -rf <target>/criterion
```

## 3. Baseline

The baseline runs in a throwaway worktree of the base, built in its own target dir, with
`CRITERION_HOME` set so criterion saves the results in the branch's `<target>/criterion`,
where the branch run reads them. Never build both checkouts in one target dir: cargo then
finds the base's binaries up to date in the branch, does not rebuild, and the comparison
measures the base twice.

```bash
mktemp -d                                       # <tmp>
git worktree add --detach <tmp>/base <base>
```

The bench code must be the same in both runs, so the base always gets the head's `benches/`
and the `[lib]` and `[[bench]]` sections of its `Cargo.toml`. That also covers a base older
than the suite, and a PR that changes the benches:

```bash
cd <tmp>/base && rm -rf benches && cp -R <branch checkout>/benches .
cd <tmp>/base && awk '/^\[/ { skip = ($0 == "[lib]" || $0 == "[[bench]]") } !skip' Cargo.toml > Cargo.toml.new && awk '/^\[/ { keep = ($0 == "[lib]" || $0 == "[[bench]]") } keep' <branch checkout>/Cargo.toml >> Cargo.toml.new && mv Cargo.toml.new Cargo.toml
```

If the bench then does not compile against the base, because the base lacks an API it uses
or has another `criterion`, copy the head's `criterion` dev-dependency too. A case that still
does not compile cannot be compared: drop it from the filter and list it as skipped.

Check that the machine is quiet (step 2), then run the chosen groups there, saving the
baseline under `<base>`. Check it again after the run; if it was not quiet, run it again.

```bash
cd <tmp>/base && CARGO_TARGET_DIR=<tmp>/base/target CRITERION_HOME=<target>/criterion RAYON_NUM_THREADS=4 CARGO_INCREMENTAL=0 caffeinate -i cargo bench --bench crossover --bench mutation -- 'crossover/(pmx|real_uniform)/|mutation/bin_swap/300x300' --save-baseline <base>
```

The first build there compiles every dependency, about 15 seconds. Give the command a timeout
above the run time (10 minutes for one group), or run it in the background and wait for it.
It must exit 0.

## 4. Comparison

In the branch checkout: remove the crate's own build first, so the run below has to compile
it from this checkout. Then check that the machine is quiet, run the same groups in the same
environment against the baseline, keeping the log, and check the machine again.

```bash
cd <branch checkout> && cargo clean --release -p evolutionary
cd <branch checkout> && (set -o pipefail; RAYON_NUM_THREADS=4 CARGO_INCREMENTAL=0 caffeinate -i cargo bench --bench crossover --bench mutation -- 'crossover/(pmx|real_uniform)/|mutation/bin_swap/300x300' --baseline <base> 2>&1 | tee <target>/criterion/<head>.log); echo "exit $?"
grep -n 'Compiling evolutionary\|panicked\|^error' <target>/criterion/<head>.log
```

Some sandboxes refuse the inline `(set -o pipefail; …)`. Putting that command in a script file
under `<tmp>` and running it with `bash` is fine.

Go on to the report only if it printed `exit 0` and the grep shows `Compiling evolutionary
v… (<branch checkout>)` and nothing else. A failed run, a panic such as `Baseline '…' must
exist`, or a run that did not compile the branch produces no table: fix the cause and redo
both runs.

`--baseline` stops at a benchmark the base does not have. If the PR adds one, use
`--baseline-lenient` instead and say in the report that the new case has no before.

## 5. Report

```bash
cd <branch checkout> && python3 -I .claude/skills/bench-report/compare.py <base> --target-dir <target> --log <target>/criterion/<head>.log
```

It prints a table of every benchmark compared against `<base>`, sorted by id, with
criterion's verdict for each from the log, which includes its p-value test. The line under
the table says where the verdicts came from; it must say "criterion's, from its log". It
leaves out, and counts on stderr, any benchmark with no `<base>` baseline or whose change is
not the change between its numbers. `--json` prints the same rows as JSON, `--filter
<prefix>` keeps some, and `--noise-threshold` must match the one criterion used.

If the control case says improved or regressed, the machine was disturbed: redo both runs
(step 3 and 4). If it still does after two redos, stop: write in the PR "could not get a quiet
machine" and the control's rows, instead of a table.

Paste into the PR description:

```markdown
## Benchmarks

Base `<base>`, head `<head>`. <CPU model>, `<uname -m>`, `RAYON_NUM_THREADS=4`, criterion
settings of `benches/common/mod.rs` <or the flags passed>. <The lockfile differed, if it did.>
Ran `cargo bench --bench crossover --bench mutation -- '<filter>'`, with
`mutation/bin_swap/300x300` as the control. Skipped: <the other groups and cases, and why>.

<the table, with the line under it>
```

The CPU model: `sysctl -n machdep.cpu.brand_string` on macOS, `grep -m1 'model name'
/proc/cpuinfo` on Linux.

## 6. Honesty

- A change inside the noise threshold, or without a significant verdict, is "no change", never
  an improvement, whatever its sign.
- Never compare against a baseline from another machine or another day. Redo both runs.
- If the machine was on battery, asleep or not quiet before or after either run, redo it
  rather than report it.
- A failing golden test means the seeded results changed. For the `generation` and
  `checkpoint` benches the "after" is then a different workload, a seeded run following
  another trajectory: say so above the table and give no verdicts for them. The operator
  benches work on a fixed input population and keep their verdicts.
- Say what was skipped, and why.
- A regression stays in the report. Explain it or fix it, do not drop the row.

## 7. Cleanup

```bash
cd <branch checkout> && git worktree remove --force <tmp>/base
rm -rf <tmp>
```

`--force` because of its `target/` and the copied bench code. `git worktree list` no longer
shows it.
