---
name: bench-report
description: Produce a before-and-after criterion benchmark report for a PR of this crate and paste it into the PR. Use for any change under src/ that touches evolution.rs, a selection, crossover or mutation operator, fitness evaluation or the RNG (rng.rs), and whenever asked to benchmark, compare performance, or do a before-and-after.
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
| `src/crossover/` | `--bench crossover`, narrowed to the cases changed: `-- crossover/pmx/` |
| `src/mutation/` | `--bench mutation`, narrowed the same way |
| `src/selection/` | `--bench selection`, narrowed the same way |
| `src/fitness.rs`, fitness evaluation, `src/rng.rs` | `--bench generation` |
| `src/checkpoint.rs` | `--features serde --bench checkpoint` |
| `src/evolution.rs`, or a release | everything: `--all-features` |

For several cases, quote a regex: `-- 'crossover/(pmx|cycle)/'`. A full run takes about 7
minutes, so a full report about 15. One group takes one to two minutes per run. Write down the
groups you skip: the report says which.

## 2. Preconditions

- `git status --porcelain` prints nothing. Commit or set aside your changes first.
- Close other heavy processes: builds, test runs, other agents' cargo, video. Right before
  each run, `uptime` shows a 1-minute load average well below the number of cores. During a
  run the bench itself adds about `RAYON_NUM_THREADS`.
- The machine is awake and on mains power. On macOS, `pmset -g batt | head -1` says
  `AC Power`, and each `cargo bench` runs under `caffeinate -i`. On Linux, check
  `/sys/class/power_supply/*/online` and run under `systemd-inhibit`.
- Both runs on the same machine, in the same session, one after the other.
- The same environment for both runs: `RAYON_NUM_THREADS=4 CARGO_INCREMENTAL=0` on each
  `cargo bench`, as below. Keep 4 unless the PR says why.
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
```

For a PR stacked on another branch, use `origin/<that branch>` instead of `origin/main`.
`<target>` is `<branch checkout>/target`, or `$CARGO_TARGET_DIR` if that is set.

Empty criterion's results, so nothing from another run or day can end up in the report:

```bash
rm -rf <target>/criterion
```

## 3. Baseline

The baseline runs in a throwaway worktree of the base, with its own build, and with
`CRITERION_HOME` set so criterion saves it in the branch's `<target>/criterion`, where the
branch run reads it. Do not share `CARGO_TARGET_DIR` between the two checkouts instead: cargo
then finds the base's binaries up to date in the branch, does not rebuild, and the comparison
measures the base twice.

```bash
mktemp -d                                       # <tmp>
git worktree add --detach <tmp>/base <base>
```

The base must contain the suite: `<tmp>/base/benches/common/mod.rs` exists. If it does not,
the base is older than the suite. Replace its `benches/` with the branch's
(`rm -rf <tmp>/base/benches && cp -R <branch checkout>/benches <tmp>/base/`), and in
`<tmp>/base/Cargo.toml` replace the `[lib]` and `[[bench]]` sections and the `criterion`
dev-dependency with the branch's, keeping the rest. `cargo bench --all-features --no-run`
there checks that it builds; 0.5.0 does. A bench that does not compile against the base's
API cannot be compared: drop it and list it as skipped.

Run the chosen groups there, saving the baseline under `<base>`:

```bash
cd <tmp>/base && CRITERION_HOME=<target>/criterion RAYON_NUM_THREADS=4 CARGO_INCREMENTAL=0 caffeinate -i cargo bench --bench crossover -- 'crossover/(pmx|real_uniform)/' --save-baseline <base>
```

The first build there compiles every dependency, about 15 seconds. Give the command a timeout
above the run time (10 minutes for one group), or run it in the background and wait for it.

## 4. Comparison

In the branch checkout, the same groups and the same environment, against the baseline,
keeping the log:

```bash
cd <branch checkout> && RAYON_NUM_THREADS=4 CARGO_INCREMENTAL=0 caffeinate -i cargo bench --bench crossover -- 'crossover/(pmx|real_uniform)/' --baseline <base> 2>&1 | tee <target>/criterion/<head>.log
```

`--baseline` stops at a benchmark the base does not have. If the PR adds one, use
`--baseline-lenient` instead and say in the report that the new case has no before.

## 5. Report

```bash
cd <branch checkout> && python3 -I .claude/skills/bench-report/compare.py <base>
```

It prints a table of every benchmark compared against `<base>`, sorted by id, and leaves out
(counting them on stderr) any whose comparison is older than its numbers. `--json` prints the
same rows as JSON, `--filter <prefix>` keeps some, `--target-dir` reads another target dir,
and `--noise-threshold` must match the one criterion used.

The script cannot see criterion's p-value. Check the log: if criterion said `No change in
performance detected.` for a benchmark the table calls improved or regressed, change that
verdict to `no change`.

Paste into the PR description:

```markdown
## Benchmarks

Base `<base>`, head `<head>`. <CPU model>, `<uname -m>`, `RAYON_NUM_THREADS=4`, criterion
settings of `benches/common/mod.rs` <or the flags passed>.
Ran `cargo bench --bench crossover -- 'crossover/(pmx|real_uniform)/'`. Skipped: <the other
groups and cases, and why>.

<the table, with the line under it>
```

The CPU model: `sysctl -n machdep.cpu.brand_string` on macOS, `grep -m1 'model name'
/proc/cpuinfo` on Linux.

## 6. Honesty

- A change inside the noise threshold, or without a significant verdict, is "no change", never
  an improvement, whatever its sign.
- Never compare against a baseline from another machine or another day. Redo both runs.
- If the machine was loaded, asleep or on battery during either run, redo both rather than
  report them.
- A case the change cannot affect that comes out improved or regressed, or a time far from
  the other run's for no reason, means a disturbed run: redo both. If it persists, report it
  as noise next to the table.
- Say what was skipped, and why.
- A regression stays in the report. Explain it or fix it, do not drop the row.

## 7. Cleanup

```bash
cd <branch checkout> && git worktree remove --force <tmp>/base
rm -rf <tmp>
```

`--force` because of its `target/` and of a copied `benches/`. `git worktree list` no longer
shows it.
