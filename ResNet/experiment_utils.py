"""Framework-independent metrics, run isolation and result verification."""

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import uuid


def write_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temp.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict):
    """Hash tensor values, independent of checkpoint metadata and torch.save."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)])
        digest.update(header.encode("utf-8") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def dataset_index_sha256(dataset):
    """Fingerprint relative filenames and labels (not image file contents)."""
    digest = hashlib.sha256()
    root = Path(dataset.root)
    for filename, label in dataset.samples:
        relative = Path(filename).relative_to(root).as_posix()
        digest.update(json.dumps([relative, label], ensure_ascii=False).encode("utf-8") + b"\n")
    return digest.hexdigest()


def create_run_dir(output_root, model, seed):
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"{model}_seed{seed}_{stamp}_{uuid.uuid4().hex[:12]}"
        run_dir = root / run_id
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError("Unable to allocate a unique experiment directory")


def metrics_from_confusion(matrix, classes):
    count = len(classes)
    if count == 0 or len(matrix) != count or any(len(row) != count for row in matrix):
        raise ValueError("Confusion matrix dimensions must match the class list")
    if any(not isinstance(v, int) or v < 0 for row in matrix for v in row):
        raise ValueError("Confusion matrix entries must be nonnegative integers")
    supports = [sum(row) for row in matrix]
    predicted_totals = [sum(row[i] for row in matrix) for i in range(count)]
    total = sum(supports)
    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    records = {}
    for i, name in enumerate(classes):
        tp = matrix[i][i]
        precision = tp / predicted_totals[i] if predicted_totals[i] else 0.0
        recall = tp / supports[i] if supports[i] else 0.0
        denominator = supports[i] + predicted_totals[i]
        f1 = 2 * tp / denominator if denominator else 0.0
        records[name] = {
            "class_id": i, "correct": tp, "total": supports[i],
            "predicted_total": predicted_totals[i],
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": recall,  # Compatibility with the original export.
        }
    macro_recall = sum(v["recall"] for v in records.values()) / count
    macro_f1 = sum(v["f1"] for v in records.values()) / count
    return {
        "total_images": total,
        "overall_accuracy": sum(matrix[i][i] for i in range(count)) / total,
        "macro_accuracy": macro_recall,
        "macro_recall": macro_recall,
        "macro_precision": sum(v["precision"] for v in records.values()) / count,
        "macro_f1": macro_f1,
        "macro_f1_percent": 100 * macro_f1,
        "class_metrics": records,
        "class_accuracy": {
            name: {key: values[key] for key in ("accuracy", "correct", "total")}
            for name, values in records.items()
        },
    }


def save_metric_tables(directory, matrix, classes, metrics):
    directory = Path(directory)
    with (directory / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *classes])
        for name, row in zip(classes, matrix):
            writer.writerow([name, *row])
    with (directory / "class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["class_id", "class_name", "precision", "recall", "f1", "correct", "total", "predicted_total"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for name, values in metrics["class_metrics"].items():
            writer.writerow({"class_name": name, **values})
    with (directory / "class_accuracy.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "correct", "total", "accuracy"])
        for name, values in metrics["class_metrics"].items():
            writer.writerow([values["class_id"], name, values["correct"], values["total"], values["accuracy"]])


class BestModelTracker:
    """Save every strict accuracy improvement; early-stop delta is separate."""

    def __init__(self, patience=0, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_accuracy = -math.inf
        self.early_stop_anchor = -math.inf
        self.bad_epochs = 0

    def update(self, accuracy):
        if not math.isfinite(accuracy):
            raise ValueError("Non-finite validation accuracy")
        improved = accuracy > self.best_accuracy
        if improved:
            self.best_accuracy = accuracy
        if accuracy > self.early_stop_anchor + self.min_delta:
            self.early_stop_anchor = accuracy
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        stop = self.patience > 0 and self.bad_epochs >= self.patience
        return improved, stop
