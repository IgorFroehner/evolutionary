#!/usr/bin/env python3
"""Print criterion's comparison against a saved baseline as a Markdown table.

After `cargo bench ... -- --baseline <name> 2>&1 | tee <log>`, run from the repository root:

    python3 -I .claude/skills/bench-report/compare.py <name> --log <log>
    python3 -I .claude/skills/bench-report/compare.py <name> --filter crossover/pmx --json

It reads what criterion 0.8 writes under <target-dir>/criterion for every benchmark:

    new/benchmark.json      the id, `full_id`
    new/estimates.json      this run; `mean` is the time per iteration in ns
    <name>/estimates.json   the baseline, same format
    change/estimates.json   the relative change of the mean, with its 95% confidence interval

Before and after are the means, so after / before - 1 is the change. Criterion's console shows
the slope instead, so its middle `time:` value differs a little.

The verdict is the one criterion printed for the id in <log>, which includes its p-value test.
Without it, the verdict is criterion's rule on what it stores, without the p-value: "improved"
if the whole interval is below -threshold, "regressed" if above +threshold, else "no change".

A benchmark is left out when it has no <name> baseline, or when its change is not the change
between these numbers: written before them, or computed against another baseline.
"""

import argparse
import json
import math
import os
import re
import sys

LOG_VERDICTS = {
    "Performance has improved.": "improved",
    "Performance has regressed.": "regressed",
    "No change in performance detected.": "no change",
    "Change within noise threshold.": "no change",
}
CRITERION_DIRS = {"new", "change", "report"}


def load(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def log_verdicts(path, ids):
    """The verdict criterion printed for each id in its console output."""
    verdicts, current = {}, None
    with open(path, encoding="utf-8", errors="replace") as file:
        for line in file:
            line = re.sub(r"\x1b\[[0-9;]*m", "", line).rstrip()
            words = line.split()
            if words and not line[0].isspace():
                # Every column-0 line ends the previous benchmark's block, so the verdict of a
                # benchmark that is not a row cannot land on the row before it.
                current = words[0] if words[0] in ids else None
            elif current and line.strip() in LOG_VERDICTS:
                verdicts[current] = LOG_VERDICTS[line.strip()]
    return verdicts


def rows(criterion_dir, baseline, filters, threshold):
    found, no_baseline, stale, baselines = [], 0, 0, set()
    for root, _dirs, files in os.walk(criterion_dir):
        if os.path.basename(root) != "new" or "benchmark.json" not in files:
            continue
        bench_id = load(os.path.join(root, "benchmark.json"))["full_id"]
        if filters and not any(bench_id.startswith(f) for f in filters):
            continue
        bench = os.path.dirname(root)
        baselines.update(set(os.listdir(bench)) - CRITERION_DIRS)
        after = os.path.join(root, "estimates.json")
        before = os.path.join(bench, baseline, "estimates.json")
        change_path = os.path.join(bench, "change", "estimates.json")
        if not os.path.isfile(before):
            no_baseline += 1
            continue
        written = os.path.getmtime(change_path) if os.path.isfile(change_path) else 0
        after_ns = load(after)["mean"]["point_estimate"]
        before_ns = load(before)["mean"]["point_estimate"]
        change = load(change_path)["mean"] if written else None
        fresh = written >= max(os.path.getmtime(after), os.path.getmtime(before))
        ratio = after_ns / before_ns - 1
        if not (change and fresh and math.isclose(change["point_estimate"], ratio, abs_tol=1e-9)):
            stale += 1
            continue
        lower = change["confidence_interval"]["lower_bound"]
        upper = change["confidence_interval"]["upper_bound"]
        verdict = "improved" if upper < -threshold else "regressed" if lower > threshold else None
        found.append(
            dict(id=bench_id, before_ns=before_ns, after_ns=after_ns)
            | dict(change=change["point_estimate"], change_lower=lower, change_upper=upper)
            | dict(verdict=verdict or "no change", verdict_from="interval")
        )
    return sorted(found, key=by_id), no_baseline, stale, sorted(baselines)


def by_id(row):
    """The id, with its numbers compared as numbers: 300x300 before 2000x500."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", row["id"])]


def duration(ns):
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.2f} {unit}"
    return f"{ns:.2f} ns"


def percent(ratio):
    return f"{ratio * 100:+.2f}%"


def markdown(found, threshold):
    lines = ["| Benchmark | Before | After | Change | Verdict |", "|---|---:|---:|---:|---|"]
    for row in found:
        interval = f"[{percent(row['change_lower'])}, {percent(row['change_upper'])}]"
        lines.append(
            f"| `{row['id']}` | {duration(row['before_ns'])} | {duration(row['after_ns'])} "
            f"| {percent(row['change'])} {interval} | {row['verdict']} |"
        )
    from_log = sum(row["verdict_from"] == "log" for row in found)
    if from_log == len(found):
        verdicts = "criterion's, from its log"
    elif from_log:
        verdicts = f"criterion's from its log, {len(found) - from_log} by the interval rule"
    else:
        verdicts = "by the interval rule, without criterion's p-value"
    lines.append("")
    lines.append(
        "Mean time per iteration, not criterion's slope. Change: point estimate [95% "
        f"confidence interval]. Noise threshold {threshold * 100:g}%. Verdicts: {verdicts}."
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("baseline", help="the name given to --save-baseline and --baseline")
    parser.add_argument(
        "--target-dir",
        default=os.environ.get("CARGO_TARGET_DIR", "target"),
        help="cargo's target dir (default: $CARGO_TARGET_DIR, else target)",
    )
    parser.add_argument("--log", help="criterion's console output of the --baseline run")
    parser.add_argument(
        "--filter", action="append", default=[], help="only ids that start with this; repeatable"
    )
    parser.add_argument("--noise-threshold", type=float, default=0.03, help="default 0.03")
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    args = parser.parse_args()

    criterion_dir = os.path.join(args.target_dir, "criterion")
    found, no_baseline, stale, baselines = rows(
        criterion_dir, args.baseline, args.filter, args.noise_threshold
    )
    if args.log:
        verdicts = log_verdicts(args.log, {row["id"] for row in found})
        for row in found:
            if row["id"] in verdicts:
                row["verdict"], row["verdict_from"] = verdicts[row["id"]], "log"
    left_out = ((no_baseline, f"no '{args.baseline}' baseline"), (stale, "no current change"))
    for count, why in left_out:
        if count:
            print(f"left out {count} benchmarks with {why}", file=sys.stderr)
    if not found:
        if args.baseline not in baselines:
            sys.exit(
                f"error: no baseline '{args.baseline}' in {criterion_dir}; "
                f"baselines there: {', '.join(baselines) or 'none'}"
            )
        sys.exit(
            f"error: no current comparison against '{args.baseline}' in {criterion_dir}. "
            f"Run `cargo bench ... -- --baseline {args.baseline}` first."
        )
    if args.json:
        print(json.dumps(found, indent=2, ensure_ascii=False))
    else:
        print(markdown(found, args.noise_threshold))


if __name__ == "__main__":
    main()
