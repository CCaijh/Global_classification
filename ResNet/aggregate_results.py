"""Summarize one verified run per model/seed as macro-F1 (%) mean +/- sample SD."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

from configs import train_configs
from experiment_utils import sha256_file


SEEDS = (42, 2026, 3407)
MODELS = ("resnet50", "resnet101")
COMPARISON_KEYS = ("batch_size", "epochs", "lr", "max_lr", "momentum", "weight_decay",
                   "pct_start", "early_stop_patience", "early_stop_min_delta", "num_classes",
                   "architecture", "preprocessing", "optimizer", "scheduler", "checkpoint_selection")


def summarize(paths):
    grouped = {}
    warnings = []
    for path in paths:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "complete":
            raise ValueError(f"Incomplete result: {path}")
        exp = data["experiment"]
        key = (exp["model"], exp["seed"])
        if key[0] not in MODELS or key[1] not in SEEDS:
            continue
        if key in grouped:
            raise ValueError(f"Multiple runs for {key}: {grouped[key][0].parent} and {path.parent}. "
                             "Use --runs to explicitly choose one run per model/seed.")
        for split in ("test1", "test2"):
            metrics = data[split]
            identity = metrics["experiment"]
            for field in ("run_id", "model", "seed", "checkpoint_sha256", "state_dict_sha256"):
                if identity[field] != exp[field]:
                    raise ValueError(f"Test metadata mismatch ({field}): {path}")
            score = metrics["macro_f1"]
            if not 0 <= score <= 1:
                raise ValueError(f"Invalid macro-F1: {path}")
            prediction_path = path.parent / split / "predictions.csv"
            if sha256_file(prediction_path) != metrics["predictions_sha256"]:
                raise ValueError(f"Prediction file was changed: {prediction_path}")
        grouped[key] = (path, data)
    rows = []
    details = []
    for model in MODELS:
        missing = [seed for seed in SEEDS if (model, seed) not in grouped]
        if missing:
            raise ValueError(f"Missing complete {model} runs for seeds {missing}")
        reference = grouped[model, SEEDS[0]][1]["experiment"]
        for seed in SEEDS:
            path, data = grouped[model, seed]
            exp = data["experiment"]
            for field in COMPARISON_KEYS:
                if exp["config"][field] != reference["config"][field]:
                    raise ValueError(f"Training settings differ for {model}: {field} ({path})")
            for field in ("dataset_sizes", "dataset_index_sha256", "class_to_idx"):
                if exp[field] != reference[field]:
                    raise ValueError(f"Dataset information differs ({field}) for {model}: {path}")
            for source in ("train_benchmark.py", "experiment_utils.py", "model_resnet_448.py"):
                if exp["source_sha256"][source] != reference["source_sha256"][source]:
                    raise ValueError(f"Implementation differs ({source}) for {model}: {path}")
            details.append({"model": model, "seed": seed,
                            "test1_macro_f1_percent": 100 * data["test1"]["macro_f1"],
                            "test2_macro_f1_percent": 100 * data["test2"]["macro_f1"],
                            "run_directory": str(path.parent)})
        row = {"model": model, "training_seeds": ", ".join(map(str, SEEDS))}
        for split in ("test1", "test2"):
            scores = [100 * grouped[model, seed][1][split]["macro_f1"] for seed in SEEDS]
            mean, sd = statistics.mean(scores), statistics.stdev(scores)
            row[f"{split}_mean_percent"] = mean
            row[f"{split}_sample_sd_percent"] = sd
            row[f"{split}_mean_sd"] = f"{mean:.2f} ± {sd:.2f}"
            for i, seed in enumerate(SEEDS):
                for other in SEEDS[i + 1:]:
                    left, right = grouped[model, seed][1], grouped[model, other][1]
                    if left[split]["predictions_sha256"] == right[split]["predictions_sha256"]:
                        warnings.append(f"{model} {split}: seeds {seed}/{other} have identical predictions; verify the runs.")
        for i, seed in enumerate(SEEDS):
            for other in SEEDS[i + 1:]:
                left, right = grouped[model, seed][1], grouped[model, other][1]
                if left["experiment"]["state_dict_sha256"] == right["experiment"]["state_dict_sha256"]:
                    warnings.append(f"{model}: seeds {seed}/{other} have identical tensor values; verify the runs.")
        rows.append(row)
    # Models should also use the same configured protocol.
    first = grouped[MODELS[0], SEEDS[0]][1]["experiment"]
    second = grouped[MODELS[1], SEEDS[0]][1]["experiment"]
    for field in COMPARISON_KEYS:
        if first["config"][field] != second["config"][field]:
            raise ValueError(f"ResNet-50/101 settings differ: {field}")
    for field in ("dataset_sizes", "dataset_index_sha256", "class_to_idx"):
        if first[field] != second[field]:
            raise ValueError(f"ResNet-50/101 dataset information differs: {field}")
    for source in ("train_benchmark.py", "experiment_utils.py", "model_resnet_448.py"):
        if first["source_sha256"][source] != second["source_sha256"][source]:
            raise ValueError(f"ResNet-50/101 implementation differs: {source}")
    return rows, details, warnings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=train_configs["output_path"])
    parser.add_argument("--runs", nargs="+", help="Explicitly choose six run directories")
    parser.add_argument("--output", help="Summary CSV path (default: results root/benchmark_macro_f1_summary.csv)")
    args = parser.parse_args(argv)
    root = Path(args.results_root).expanduser().resolve()
    paths = ([Path(run) / "final_test_summary.json" for run in args.runs] if args.runs
             else sorted(root.glob("*/final_test_summary.json")))
    try:
        rows, details, warnings = summarize(paths)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Cannot summarize: {exc}\n")
    output = Path(args.output).resolve() if args.output else root / "benchmark_macro_f1_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    for destination, contents in [(output, rows), (output.with_name(output.stem + "_per_seed.csv"), details)]:
        with destination.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(contents[0]))
            writer.writeheader()
            writer.writerows(contents)
    print("Model       Test1 macro-F1 (%)    Test2 macro-F1 (%)")
    for row in rows:
        print(f"{row['model']:11} {row['test1_mean_sd']:22} {row['test2_mean_sd']}")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
