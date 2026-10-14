#!/usr/bin/env python3
"""Print criterion's comparison against a saved baseline as a Markdown table.

After `cargo bench ... -- --baseline <name>`, run from the repository root:

    python3 -I .claude/skills/bench-report/compare.py <name>
    python3 -I .claude/skills/bench-report/compare.py <name> --filter crossover/pmx --json

It reads what criterion 0.8 writes under <target-dir>/criterion for every benchmark:

    new/benchmark.json      the id, `full_id`
    new/estimates.json      this run; `mean` is the time per iteration in ns
    <name>/estimates.json   the baseline, same format
    change/estimates.json   the relative change of the mean, with its 95% confidence interval

Before and after are the means, so after / before - 1 is the change. The verdict is criterion's
rule on what it stores: "improved" if the whole interval is below -threshold, "regressed" if it
is above +threshold, "no change" otherwise. Criterion also requires p < 0.05, which it does not
store, so when its console said "No change in performance detected" that is the verdict.

A benchmark is left out when its change was written before its run or its baseline, that is,
when the comparison is older than the numbers it would be shown with.
"""

import argparse
import json
import os
import re
import sys


def load(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def rows(criterion_dir, baseline, filters, threshold):
    found, skipped = [], 0
    for root, _dirs, files in os.walk(criterion_dir):
        if os.path.basename(root) != "new" or "benchmark.json" not in files:
            continue
        bench_id = load(os.path.join(root, "benchmark.json"))["full_id"]
        if filters and not any(bench_id.startswith(f) for f in filters):
            continue
        bench = os.path.dirname(root)
        after = os.path.join(root, "estimates.json")
        before = os.path.join(bench, baseline, "estimates.json")
        change_path = os.path.join(bench, "change", "estimates.json")
        if not all(os.path.isfile(p) for p in (after, before, change_path)):
            skipped += 1
            continue
        written = os.path.getmtime(change_path)
        if written < os.path.getmtime(after) or written < os.path.getmtime(before):
            skipped += 1
            continue
        change = load(change_path)["mean"]
        lower = change["confidence_interval"]["lower_bound"]
        upper = change["confidence_interval"]["upper_bound"]
        if upper < -threshold:
            verdict = "improved"
        elif lower > threshold:
            verdict = "regressed"
        else:
            verdict = "no change"
        found.append(
            {
                "id": bench_id,
                "before_ns": load(before)["mean"]["point_estimate"],
                "after_ns": load(after)["mean"]["point_estimate"],
                "change": change["point_estimate"],
                "change_lower": lower,
                "change_upper": upper,
                "verdict": verdict,
            }
        )
    return sorted(found, key=by_id), skipped


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
    lines = [
        "| Benchmark | Before | After | Change | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for row in found:
        interval = f"[{percent(row['change_lower'])}, {percent(row['change_upper'])}]"
        lines.append(
            f"| `{row['id']}` | {duration(row['before_ns'])} | {duration(row['after_ns'])} "
            f"| {percent(row['change'])} {interval} | {row['verdict']} |"
        )
    lines.append("")
    lines.append(
        "Mean time per iteration. Change: point estimate [95% confidence interval]. "
        f"Noise threshold {threshold * 100:g}%."
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("baseline", help="the name given to --save-baseline and --baseline")
    parser.add_argument("--target-dir", default="target", help="cargo's target dir (target)")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="only ids that start with this; repeat for more than one",
    )
    parser.add_argument("--noise-threshold", type=float, default=0.03, help="default 0.03")
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    args = parser.parse_args()

    criterion_dir = os.path.join(args.target_dir, "criterion")
    found, skipped = rows(criterion_dir, args.baseline, args.filter, args.noise_threshold)
    if skipped:
        print(f"skipped {skipped} benchmarks without a current comparison", file=sys.stderr)
    if not found:
        matching = f" matching {', '.join(args.filter)}" if args.filter else ""
        sys.exit(
            f"error: no comparison against baseline '{args.baseline}' in {criterion_dir}"
            f"{matching}. Run `cargo bench ... -- --baseline {args.baseline}` first."
        )
    if args.json:
        print(json.dumps(found, indent=2, ensure_ascii=False))
    else:
        print(markdown(found, args.noise_threshold))


if __name__ == "__main__":
    main()
