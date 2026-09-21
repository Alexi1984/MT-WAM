import os
import sys
import json
import argparse
from collections import defaultdict
import pandas as pd
import math


def format_time(seconds):
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds:02d}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:02d}m{remaining_seconds:02d}s"
    else:
        hours = seconds // 3600
        remaining = seconds % 3600
        minutes = remaining // 60
        remaining_seconds = remaining % 60
        return f"{hours:02d}h{minutes:02d}m{remaining_seconds:02d}s"


SUITE_EXPECTED_TASKS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}


def audit_suite_completeness(
    seen_task_ids_by_suite, trials_by_task, expected_tasks=None
):
    expected_tasks = expected_tasks or SUITE_EXPECTED_TASKS
    warnings = []
    for suite in sorted(seen_task_ids_by_suite):
        seen = set(seen_task_ids_by_suite[suite])
        exp = expected_tasks.get(suite)
        if exp is not None and len(seen) < exp:
            missing = sorted(set(range(exp)) - seen)
            warnings.append(
                (
                    suite,
                    f"only {len(seen)}/{exp} task JSONs present; missing task ids {missing}",
                )
            )
        counts = [
            trials_by_task[suite, t] for t in seen if (suite, t) in trials_by_task
        ]
        if counts:
            modal = max(set(counts), key=counts.count)
            short = sorted(
                (t for t in seen if trials_by_task.get((suite, t), modal) < modal)
            )
            if short:
                warnings.append(
                    (
                        suite,
                        f"tasks {short} ran fewer than {modal} trials (modal) — partial task results",
                    )
                )
    return warnings


def summarize_results(
    output_dir,
    strict=False,
    classification_json=None,
    manifest=None,
    classification_sidecar=None,
):
    axis_of = None
    if classification_json is not None:
        with open(classification_json, "r", encoding="utf-8") as f:
            _cls = json.load(f)
        axis_of = {s: [rec.get("category") for rec in _cls[s]] for s in _cls}
    if classification_sidecar is not None:
        if classification_json is not None:
            raise ValueError("Provide one classification source.")
        import csv

        axis_of = {}
        with open(classification_sidecar, newline="", encoding="utf-8") as f:
            seen = set()
            for row in csv.DictReader(f, delimiter="\t"):
                suite, task_id = (row["suite"], int(row["task_id"]))
                if task_id < 0 or (suite, task_id) in seen:
                    raise ValueError(
                        "Classification sidecar has invalid or duplicate task IDs."
                    )
                seen.add((suite, task_id))
                values = axis_of.setdefault(suite, [])
                values.extend([None] * max(0, task_id + 1 - len(values)))
                values[task_id] = row["axis"]
    ood_mode = axis_of is not None
    axis_stats = defaultdict(
        lambda: {"total_trials": 0, "total_successes": 0, "n_variants": 0}
    )
    manifest_expected = None
    if manifest is not None:
        manifest_expected = set()
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                suite_name, tid = line.split(",")
                manifest_expected.add((suite_name, int(tid)))
    suite_stats = defaultdict(
        lambda: {
            "total_tasks": 0,
            "total_trials": 0,
            "total_successes": 0,
            "total_time": 0,
            "max_time": 0,
        }
    )
    task_results = {}
    seen_task_ids_by_suite = defaultdict(set)
    trials_by_task = {}
    corrupt_files = []
    for suite in [
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_90",
    ]:
        suite_dir = os.path.join(output_dir, suite)
        if not os.path.exists(suite_dir):
            continue
        for filename in os.listdir(suite_dir):
            if not filename.startswith("gpu") or not filename.endswith("_results.json"):
                continue
            try:
                with open(os.path.join(suite_dir, filename), "r") as f:
                    result = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                corrupt_files.append((suite, filename, str(e)))
                continue
            parts = filename.split("_")
            task_id = int(parts[1].replace("task", ""))
            task_key = f"{suite}_{task_id}"
            if task_key in task_results:
                raise ValueError(
                    f"Duplicate result for {suite}, task {task_id}. Keep one complete result per task."
                )
            count = result.get("total_episodes")
            successes = result.get("successes")
            duration = result.get("duration")
            if (
                type(count) is not int
                or count <= 0
                or type(successes) is not int
                or (not 0 <= successes <= count)
            ):
                raise ValueError(f"Invalid episode counts in {suite}/{filename}.")
            if (
                not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise ValueError(f"Invalid duration in {suite}/{filename}.")
            if (
                result.get("task_suite", suite) != suite
                or result.get("task_id", task_id) != task_id
            ):
                raise ValueError(f"Result identity does not match {suite}/{filename}.")
            seen_task_ids_by_suite[suite].add(task_id)
            trials_by_task[suite, task_id] = result["total_episodes"]
            if ood_mode:
                suite_axes = axis_of.get(suite)
                axis = (
                    suite_axes[task_id]
                    if suite_axes is not None and 0 <= task_id < len(suite_axes)
                    else None
                )
                if axis is not None:
                    a = axis_stats[axis]
                    a["total_trials"] += result["total_episodes"]
                    a["total_successes"] += result["successes"]
                    a["n_variants"] += 1
            stats = suite_stats[suite]
            stats["total_tasks"] += 1
            stats["total_trials"] += result["total_episodes"]
            stats["total_successes"] += result["successes"]
            stats["total_time"] += result["duration"]
            stats["max_time"] = max(stats["max_time"], result["duration"])
            task_result = {
                "success_rate": result["successes"] / result["total_episodes"] * 100,
                "duration": result["duration"],
                "total_episodes": result["total_episodes"],
                "successes": result["successes"],
                "task_description": result["task_description"]
                if "task_description" in result
                else "",
            }
            task_results[task_key] = task_result
    if ood_mode and manifest_expected is not None:
        seen_pairs = {
            (s, t) for s, tids in seen_task_ids_by_suite.items() for t in tids
        }
        missing_pairs = sorted(manifest_expected - seen_pairs)
        extra_pairs = sorted(seen_pairs - manifest_expected)
        completeness_warnings = []
        if missing_pairs:
            preview = ", ".join((f"{s},{t}" for s, t in missing_pairs[:20]))
            completeness_warnings.append(
                (
                    "__manifest__",
                    f"{len(missing_pairs)}/{len(manifest_expected)} manifest variants have no result JSON. First missing: {preview}"
                    + (" ..." if len(missing_pairs) > 20 else ""),
                )
            )
        if extra_pairs:
            completeness_warnings.append(
                (
                    "__manifest__",
                    f"{len(extra_pairs)} result JSON(s) NOT in the manifest (unexpected task results): {extra_pairs[:10]}",
                )
            )
    else:
        completeness_warnings = (
            []
            if ood_mode
            else audit_suite_completeness(seen_task_ids_by_suite, trials_by_task)
        )
    if corrupt_files:
        for suite, filename, err in corrupt_files:
            completeness_warnings.append(
                (suite, f"CORRUPT result JSON skipped: {filename} ({err})")
            )
    if completeness_warnings:
        print("\n" + "!" * 72)
        print(
            "!! INCOMPLETE EVALUATION — success rates below are over a SHRUNK denominator"
        )
        for suite, msg in completeness_warnings:
            print(f"!!   [{suite}] {msg}")
        print("!! Complete the missing tasks to include them in the summary.")
        print("!" * 72)
    print("\n=== Evaluation Results Summary ===")
    print("\nStatistics for each task suite:")
    total_success_rate = 0
    total_time = 0
    total_suites = 0
    df_data = {
        "Task Suite": [],
        "Success Rate (%)": [],
        "Average Time (s)": [],
        "Max Time (s)": [],
    }
    for suite, stats in suite_stats.items():
        if stats["total_trials"] > 0:
            success_rate = stats["total_successes"] / stats["total_trials"] * 100
            avg_time = stats["total_time"] / stats["total_tasks"]
            max_time = stats["max_time"]
            print(f"\n{suite}:")
            print(f"- Tasks completed: {stats['total_tasks']}")
            print(f"- Total attempts: {stats['total_trials']}")
            print(f"- Successful attempts: {stats['total_successes']}")
            print(f"- Success rate: {success_rate:.2f}%")
            print(f"- Total time: {format_time(stats['total_time'])}")
            print(f"- Average time per task: {format_time(avg_time)}")
            print(f"- Longest task time: {format_time(max_time)}")
            df_data["Task Suite"].append(suite)
            df_data["Success Rate (%)"].append(f"{success_rate:.2f}")
            df_data["Average Time (s)"].append(f"{avg_time:.2f}")
            df_data["Max Time (s)"].append(f"{max_time:.2f}")
            total_success_rate += success_rate
            total_time += stats["total_time"]
            total_suites += 1
    if total_suites > 0:
        print("\nOverall statistics:")
        avg_success_rate = total_success_rate / total_suites
        avg_task_time = total_time / sum(
            (s["total_tasks"] for s in suite_stats.values())
        )
        max_task_time = max((s["max_time"] for s in suite_stats.values()))
        print(f"- Average success rate: {avg_success_rate:.2f}%")
        print(f"- Total time: {format_time(total_time)}")
        print(f"- Average time per task: {format_time(avg_task_time)}")
        print(f"- Longest task time: {format_time(max_task_time)}")
        df_data["Task Suite"].append("Overall")
        df_data["Success Rate (%)"].append(f"{avg_success_rate:.2f}")
        df_data["Average Time (s)"].append(f"{avg_task_time:.2f}")
        df_data["Max Time (s)"].append(f"{max_task_time:.2f}")
    axis_summary = {}
    if ood_mode and axis_stats:
        print("\n=== OOD Per-Axis Success Rates ===")
        pooled_succ = 0
        pooled_trials = 0
        for axis in sorted(axis_stats):
            a = axis_stats[axis]
            if a["total_trials"] == 0:
                continue
            sr = a["total_successes"] / a["total_trials"] * 100
            axis_summary[axis] = {
                "success_rate": sr,
                "n_variants": a["n_variants"],
                "total_trials": a["total_trials"],
                "total_successes": a["total_successes"],
            }
            pooled_succ += a["total_successes"]
            pooled_trials += a["total_trials"]
            print(
                f"- {axis}: {sr:.2f}%  (variants={a['n_variants']}, trials={a['total_trials']})"
            )
        if pooled_trials > 0:
            total_sr = pooled_succ / pooled_trials * 100
            axis_summary["__total_variant_weighted__"] = {
                "success_rate": total_sr,
                "total_trials": pooled_trials,
                "total_successes": pooled_succ,
            }
            print(
                f"- TOTAL (variant-weighted): {total_sr:.2f}%  (trials={pooled_trials})"
            )
    df = pd.DataFrame(df_data)
    ckpt_path = os.environ.get("MTWAM_CKPT", "")
    title = os.path.basename(ckpt_path) if ckpt_path else "Results"
    df = df.set_index("Task Suite").T
    with open(os.path.join(output_dir, "summary.csv"), "w") as f:
        f.write(f"{title}\n")
        df.to_csv(f)
    task_success_data = {"Task": [], "Description": [], "Success Rate (%)": []}
    suite_tasks = defaultdict(list)
    for task in task_results:
        suite = task.split("_")[0] + "_" + task.split("_")[1]
        suite_tasks[suite].append(task)
    for suite in suite_tasks:
        suite_tasks[suite].sort(key=lambda x: int(x.split("_")[-1]))
    for suite in sorted(suite_tasks.keys()):
        for task in suite_tasks[suite]:
            result = task_results[task]
            task_success_data["Task"].append(task)
            task_success_data["Description"].append(
                result["task_description"] if "task_description" in result else ""
            )
            task_success_data["Success Rate (%)"].append(
                f"{result['success_rate']:.2f}"
            )
    suite_stats_output = {}
    for suite, stats in suite_stats.items():
        suite_stats_output[suite] = {
            "total_tasks": stats["total_tasks"],
            "total_trials": stats["total_trials"],
            "total_successes": stats["total_successes"],
            "total_time": stats["total_time"],
            "max_time": stats["max_time"],
        }
    task_success_df = pd.DataFrame(task_success_data)
    task_success_df.to_csv(
        os.path.join(output_dir, "task_success_rates.csv"), index=False
    )
    summary_file = os.path.join(output_dir, "summary.json")
    overall_stats = {
        "average_success_rate": total_success_rate / total_suites
        if total_suites > 0
        else 0,
        "total_time": total_time,
        "average_task_time": total_time
        / sum((s["total_tasks"] for s in suite_stats.values()))
        if suite_stats
        else 0,
    }
    with open(summary_file, "w") as f:
        json.dump(
            {
                "run_id": os.path.basename(output_dir),
                "ckpt": os.environ.get("MTWAM_CKPT", ""),
                "config": os.environ.get("MTWAM_CONFIG", ""),
                "suite_stats": suite_stats_output,
                "task_results": task_results,
                "axis_summary": axis_summary,
                "overall": overall_stats,
                "completeness_warnings": [
                    {"suite": suite, "message": msg}
                    for suite, msg in completeness_warnings
                ],
            },
            f,
            indent=4,
        )
    print(f"\n=== Run Information ===")
    print(f"Run ID: {os.path.basename(output_dir)}")
    print(f"Results directory: {output_dir}")
    print(f"Summary file: {summary_file}")
    print(f"Summary CSV: {os.path.join(output_dir, 'summary.csv')}")
    print(
        f"Task success rates CSV: {os.path.join(output_dir, 'task_success_rates.csv')}"
    )
    print("\n=== Task Success Rates ===")
    print(task_success_df.to_string(index=False))
    print("\n=== Results Table ===")
    print(df.to_string(index=False))
    if strict and completeness_warnings:
        print(
            "\n[--strict] Exiting non-zero due to incomplete evaluation.",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory containing evaluation results",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any suite is incomplete (missing tasks / short trials)",
    )
    parser.add_argument(
        "--classification-json",
        type=str,
        default=None,
        help="LIBERO-Plus task_classification.json; enables per-axis OOD aggregation (and disables the IID 10-task-per-suite completeness audit).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Eval manifest ({suite},{task_id} lines). OOD mode only: audits completeness against the manifest (missing/extra variants flagged) instead of skipping the audit.",
    )
    parser.add_argument(
        "--classification-sidecar",
        type=str,
        default=None,
        help="TSV with suite, task_id and axis columns.",
    )
    args = parser.parse_args()
    summarize_results(
        args.output_dir,
        strict=args.strict,
        classification_json=args.classification_json,
        manifest=args.manifest,
        classification_sidecar=args.classification_sidecar,
    )


if __name__ == "__main__":
    main()
