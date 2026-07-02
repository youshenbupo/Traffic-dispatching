"""Select one teacher per scenario using held-out, censoring-aware metrics."""
import argparse
import glob
import json
import os
import re
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--audit_dir", default="results/teacher_audit")
    parser.add_argument("--output_root", default="logs/selected_teachers")
    parser.add_argument("--completion_fraction", type=float, default=0.95)
    args = parser.parse_args()

    pattern = os.path.join(
        args.audit_dir, f"{args.scenario}_s*_model.json"
    )
    candidates = []
    regex = re.compile(
        rf"{re.escape(args.scenario)}_s(\d+)_(best_model|final_model)\.json$"
    )
    for path in sorted(glob.glob(pattern)):
        match = regex.search(os.path.basename(path))
        if not match:
            continue
        with open(path, encoding="utf-8") as stream:
            metrics = json.load(stream)
        candidates.append({
            "seed": int(match.group(1)),
            "checkpoint": match.group(2),
            "audit_path": path,
            "completion_rate": metrics["completion_rate"],
            "time_spent_per_departed_vehicle": metrics[
                "time_spent_per_departed_vehicle"
            ],
            "total_time_spent": metrics["total_time_spent"],
        })
    if not candidates:
        raise RuntimeError(f"No audit candidates found for {args.scenario}")

    best_completion = max(c["completion_rate"] for c in candidates)
    threshold = args.completion_fraction * best_completion
    eligible = [
        candidate for candidate in candidates
        if candidate["completion_rate"] >= threshold
    ]
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate["time_spent_per_departed_vehicle"],
            -candidate["completion_rate"],
        ),
    )

    if args.scenario == "hangzhou":
        experiment = f"mappo_hangzhou4x4_continue_s{selected['seed']}"
    elif args.scenario == "cologne3":
        experiment = f"mappo_cologne3_continue_s{selected['seed']}"
    else:
        raise ValueError(f"Unknown scenario {args.scenario}")
    source = os.path.join(
        "logs", experiment, selected["checkpoint"], "mappo.pth"
    )
    output_dir = os.path.join(args.output_root, args.scenario)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(source, os.path.join(output_dir, "mappo.pth"))
    report = {
        "selection_rule": (
            "completion_rate >= 0.95 * best, then minimize "
            "time_spent_per_departed_vehicle"
        ),
        "completion_threshold": threshold,
        "selected": selected,
        "source_checkpoint": source,
        "eligible_candidates": eligible,
        "all_candidates": candidates,
    }
    with open(
        os.path.join(output_dir, "selection.json"), "w", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report["selected"], indent=2))


if __name__ == "__main__":
    main()
