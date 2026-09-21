import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def _allocate(total_n, cell_sizes):
    pop = sum(cell_sizes.values())
    if pop <= total_n:
        return dict(cell_sizes)
    keys = sorted(cell_sizes)
    raw = {k: total_n * cell_sizes[k] / pop for k in keys}
    alloc = {k: min(int(raw[k]), cell_sizes[k]) for k in keys}
    remainder = total_n - sum(alloc.values())
    order = sorted(keys, key=lambda k: (raw[k] - int(raw[k]), k), reverse=True)
    idx = 0
    guard = 0
    max_guard = (remainder + len(order) + 1) * (len(order) + 1)
    while remainder > 0 and guard < max_guard:
        k = order[idx % len(order)]
        if alloc[k] < cell_sizes[k]:
            alloc[k] += 1
            remainder -= 1
        idx += 1
        guard += 1
    return alloc


def sample_variants(
    classification_json,
    suites,
    per_axis_n,
    seed,
    exclude_names=None,
    keep_null_difficulty=False,
):
    exclude_names = set(exclude_names or ())
    with open(classification_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed)
    per_axis_cells = defaultdict(lambda: defaultdict(list))
    dropped_null = Counter()
    kept_null = Counter()
    dropped_excluded = Counter()
    axis_pool = Counter()
    for suite in suites:
        if suite not in data:
            raise KeyError(f"suite {suite!r} not in {classification_json}")
        for task_id, rec in enumerate(data[suite]):
            axis = rec.get("category")
            difficulty = rec.get("difficulty_level")
            if difficulty is None:
                if not keep_null_difficulty:
                    dropped_null[axis] += 1
                    continue
                difficulty = -1
                kept_null[axis] += 1
            if rec.get("name") in exclude_names:
                dropped_excluded[axis] += 1
                continue
            per_axis_cells[axis][suite, difficulty].append(task_id)
            axis_pool[axis] += 1
    manifest = []
    sidecar = []
    per_axis_sampled = Counter()
    per_axis_alloc = {}
    for axis in sorted(per_axis_cells):
        cells = per_axis_cells[axis]
        cell_sizes = {c: len(v) for c, v in cells.items()}
        alloc = _allocate(per_axis_n, cell_sizes)
        per_axis_alloc[axis] = alloc
        for cell in sorted(cells):
            n_draw = alloc.get(cell, 0)
            if n_draw <= 0:
                continue
            pool = sorted(cells[cell])
            picked = rng.sample(pool, n_draw) if n_draw < len(pool) else list(pool)
            suite, difficulty = cell
            for task_id in picked:
                manifest.append((suite, task_id))
                sidecar.append((suite, task_id, axis, difficulty))
                per_axis_sampled[axis] += 1
    manifest.sort()
    sidecar.sort()
    report = {
        "seed": seed,
        "per_axis_n_target": per_axis_n,
        "axis_pool": dict(axis_pool),
        "per_axis_sampled": dict(per_axis_sampled),
        "dropped_null": dict(dropped_null),
        "kept_null": dict(kept_null),
        "dropped_excluded": dict(dropped_excluded),
        "n_exclude_names": len(exclude_names),
        "total_sampled": len(manifest),
    }
    return (manifest, sidecar, report)


def _load_exclude_names(path):
    if path is None:
        return set()
    names = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and (not s.startswith("#")):
                names.add(s)
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Select LIBERO-Plus variants by perturbation axis."
    )
    parser.add_argument(
        "--classification-json",
        required=True,
        help="Path to LIBERO-Plus task_classification.json.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Manifest output path ({suite},{task_id} per line).",
    )
    parser.add_argument(
        "--sidecar",
        default=None,
        help="Optional audit TSV: suite\\ttask_id\\taxis\\tdifficulty.",
    )
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument(
        "--per-axis-n",
        type=int,
        default=400,
        help="Variants to sample per axis (default 400).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260707,
        help="Sampling seed; fixed so arms (1)/(2)/(5) share one variant set.",
    )
    parser.add_argument(
        "--exclude-names",
        default=None,
        help="Optional file of variant names to exclude, one per line.",
    )
    parser.add_argument(
        "--keep-null-difficulty",
        action="store_true",
        help="Keep difficulty_level=null variants (sentinel cell -1) instead of dropping them. Required for the official-10030 full-census manifest.",
    )
    args = parser.parse_args()
    exclude_names = _load_exclude_names(args.exclude_names)
    manifest, sidecar, report = sample_variants(
        classification_json=args.classification_json,
        suites=args.suites,
        per_axis_n=args.per_axis_n,
        seed=args.seed,
        exclude_names=exclude_names,
        keep_null_difficulty=args.keep_null_difficulty,
    )
    out = Path(os.path.expanduser(os.path.expandvars(args.output)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for suite, task_id in manifest:
            f.write(f"{suite},{task_id}\n")
    if args.sidecar:
        side = Path(os.path.expanduser(os.path.expandvars(args.sidecar)))
        side.parent.mkdir(parents=True, exist_ok=True)
        with side.open("w", encoding="utf-8") as f:
            f.write("suite\ttask_id\taxis\tdifficulty\n")
            for suite, task_id, axis, difficulty in sidecar:
                f.write(f"{suite}\t{task_id}\t{axis}\t{difficulty}\n")
    print("=== OOD variant sampling ===")
    print(f"classification: {args.classification_json}")
    print(f"seed={report['seed']}  per_axis_n={report['per_axis_n_target']}")
    print(f"suites: {args.suites}")
    print(f"exclude-names: {report['n_exclude_names']} name(s)")
    print("axis                     pool  sampled  dropped_null  dropped_excluded")
    for axis in sorted(report["axis_pool"]):
        dn = report["dropped_null"].get(axis, 0)
        de = report["dropped_excluded"].get(axis, 0)
        print(
            f"{axis:24s} {report['axis_pool'][axis]:5d} {report['per_axis_sampled'].get(axis, 0):8d} {dn:13d} {de:16d}"
        )
    print(f"TOTAL sampled: {report['total_sampled']}  -> {out}")
    if args.sidecar:
        print(f"sidecar: {args.sidecar}")


if __name__ == "__main__":
    main()
