"""
Print everything worth knowing about the saved results in one report:
checkpoint metadata, test metrics, training history, error breakdowns and a
decision-threshold sweep over the saved test predictions.

Usage (from the project root):

    python evaluation/show_results.py
    python evaluation/show_results.py --out results/results_report.txt
    python evaluation/show_results.py --no-checkpoints      # skip the slow part
    python evaluation/show_results.py --root "D:/other/copy/of/project"
"""

import argparse
import csv
import datetime
import gc
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

BREAKDOWN_FILES = [
    "error_by_modality.csv",
    "error_by_duration.csv",
    "error_by_fake_period_count.csv",
]


class Report:

    def __init__(self):
        self.lines = []

    def add(self, text=""):
        print(text)
        self.lines.append(text)

    def section(self, title):
        self.add()
        self.add("=" * 80)
        self.add(title)
        self.add("=" * 80)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def table(report, header, rows):
    if not rows:
        report.add("(none)")
        return

    cells = [[fmt(c) for c in row] for row in [header] + rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]

    for index, row in enumerate(cells):
        report.add("  ".join(c.ljust(w) for c, w in zip(row, widths)))
        if index == 0:
            report.add("  ".join("-" * w for w in widths))


def load_checkpoint(path):
    import torch

    # mmap avoids reading 2+ GB of weights just to look at the metadata.
    attempts = [
        {"mmap": True, "weights_only": True},
        {"mmap": True, "weights_only": False},
        {"weights_only": False},
    ]

    last_error = None

    for kwargs in attempts:
        try:
            return torch.load(path, map_location="cpu", **kwargs)
        except Exception as error:
            last_error = error

    raise last_error


def show_checkpoints(report, results_dir):
    report.section("CHECKPOINTS")

    paths = sorted(
        list(results_dir.rglob("*.pth"))
        + list(results_dir.rglob("*.pt"))
        + list(results_dir.rglob("*.ckpt")),
        key=lambda p: p.stat().st_mtime,
    )

    if not paths:
        report.add(f"No checkpoints found under {results_dir}")
        return

    try:
        import torch
    except ImportError:
        report.add("torch is not installed - listing files only.")
        torch = None

    for path in paths:
        stat = path.stat()
        modified = datetime.datetime.fromtimestamp(stat.st_mtime)

        report.add()
        report.add(f"[{path.relative_to(results_dir.parent).as_posix()}]")
        report.add(
            f"  size: {stat.st_size / 1024 ** 3:.2f} GB"
            f" | modified: {modified:%Y-%m-%d %H:%M}"
        )

        if torch is None:
            continue

        try:
            checkpoint = load_checkpoint(path)
        except Exception as error:
            report.add(f"  COULD NOT LOAD: {type(error).__name__}: {error}")
            continue

        if not isinstance(checkpoint, dict):
            report.add(f"  unexpected checkpoint type: {type(checkpoint)}")
            continue

        state = checkpoint.get("model_state_dict")

        if state is None and all(
            torch.is_tensor(v) for v in checkpoint.values()
        ):
            state = checkpoint
            report.add("  (bare state_dict, no training metadata)")

        for key, value in checkpoint.items():
            if key.endswith("state_dict") or torch.is_tensor(value):
                continue
            if isinstance(value, (dict, list)) and len(str(value)) > 300:
                value = f"<{type(value).__name__} with {len(value)} items>"
            report.add(f"  {key}: {fmt(value)}")

        if state is not None:
            groups = {}
            for name, tensor in state.items():
                group = name.split(".")[0]
                groups[group] = groups.get(group, 0) + tensor.numel()

            total = sum(groups.values())
            report.add(f"  parameters: {total / 1e6:.1f}M")
            for group, count in groups.items():
                report.add(f"    {group}: {count / 1e6:.1f}M")

        optimizer = checkpoint.get("optimizer_state_dict")

        if optimizer is not None:
            rates = [g.get("lr") for g in optimizer.get("param_groups", [])]
            report.add(f"  optimizer lr per group: {rates}")
        else:
            report.add("  optimizer state: not saved (cannot resume exactly)")

        for key in ("scheduler_state_dict", "scaler_state_dict"):
            if key in checkpoint:
                report.add(f"  {key}: saved")

        del checkpoint, state
        gc.collect()


def show_test_metrics(report, results_dir):
    report.section("TEST METRICS")

    rows = []

    for path in sorted(results_dir.rglob("*metrics*.json")):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            continue

        rows.append([
            path.stem,
            data.get("test_samples", ""),
            data.get("accuracy", ""),
            data.get("precision", ""),
            data.get("recall", ""),
            data.get("f1_score", ""),
            data.get("roc_auc", ""),
            data.get("false_positive", ""),
            data.get("false_negative", ""),
            data.get("test_loss", ""),
        ])

    table(
        report,
        ["run", "n", "acc", "prec", "recall", "f1", "auc", "FP", "FN", "loss"],
        rows,
    )


def show_training_history(report, results_dir):
    report.section("TRAINING HISTORY")

    paths = sorted(results_dir.rglob("*history*.json"))

    if not paths:
        report.add("(none)")

    for path in paths:
        with open(path, encoding="utf-8") as handle:
            history = json.load(handle)

        report.add()
        report.add(f"[{path.relative_to(results_dir.parent).as_posix()}]")

        rows = [
            [
                epoch.get("epoch", ""),
                epoch.get("train_samples", ""),
                epoch.get("validation_samples", ""),
                epoch.get("train_loss", ""),
                epoch.get("train_accuracy", ""),
                epoch.get("val_loss", ""),
                epoch.get("val_accuracy", ""),
                epoch.get("val_f1", ""),
                epoch.get("val_auc", ""),
                epoch.get("training_time_seconds", 0.0) / 60,
            ]
            for epoch in history
        ]

        table(
            report,
            [
                "epoch", "train_n", "val_n", "train_loss", "train_acc",
                "val_loss", "val_acc", "val_f1", "val_auc", "train_min",
            ],
            rows,
        )


def show_error_breakdowns(report, results_dir):
    report.section("ERROR BREAKDOWNS")

    found = False

    for directory in sorted(results_dir.glob("error_analysis*")):
        for name in BREAKDOWN_FILES:
            path = directory / name

            if not path.is_file():
                continue

            found = True

            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

            report.add()
            report.add(f"[{directory.name}/{name}]")

            body = [
                [
                    float(c) if "." in c and c.replace(".", "").isdigit() else c
                    for c in row
                ]
                for row in rows[1:]
            ]

            table(report, rows[0], body)

    if not found:
        report.add("(none)")


def show_threshold_sweeps(report, results_dir):
    report.section("THRESHOLD SWEEP (deepfake if probability >= threshold)")

    # Error-analysis folders hold filtered subsets, not full prediction sets.
    paths = sorted(
        p for p in results_dir.rglob("*predictions*.csv")
        if not p.parent.name.startswith("error_analysis")
    )

    if not paths:
        report.add("(no prediction CSVs found - they live in results/test/)")

    for path in paths:
        labels = []
        probabilities = []

        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if "deepfake_probability" not in row:
                    break
                labels.append(int(row["true_label"]))
                probabilities.append(float(row["deepfake_probability"]))

        if not labels:
            continue

        report.add()
        report.add(f"[{path.relative_to(results_dir.parent).as_posix()}]")

        rows = []

        for threshold in THRESHOLDS:
            tp = fp = tn = fn = 0

            for label, probability in zip(labels, probabilities):
                predicted = probability >= threshold
                if predicted and label:
                    tp += 1
                elif predicted:
                    fp += 1
                elif label:
                    fn += 1
                else:
                    tn += 1

            real_recall = tn / (tn + fp) if tn + fp else 0.0
            fake_recall = tp / (tp + fn) if tp + fn else 0.0

            rows.append([
                threshold,
                (tp + tn) / len(labels),
                (real_recall + fake_recall) / 2,
                fp,
                fn,
            ])

        table(report, ["threshold", "acc", "balanced_acc", "FP", "FN"], rows)


def show_run_logs(report, results_dir, tail):
    report.section("RUN LOGS")

    paths = sorted(results_dir.rglob("*.log"))

    if not paths:
        report.add("(none)")

    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()

        report.add()
        report.add(
            f"[{path.relative_to(results_dir.parent).as_posix()}]"
            f" last {min(tail, len(lines))} of {len(lines)} lines"
        )

        for line in lines[-tail:]:
            report.add("  " + line[:300])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="project root containing results/ (default: this repo)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="also write the report to this text file",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="skip opening checkpoint files",
    )
    parser.add_argument(
        "--log-tail",
        type=int,
        default=15,
        help="lines to show from the end of each run log",
    )
    args = parser.parse_args()

    results_dir = args.root.resolve() / "results"

    if not results_dir.is_dir():
        sys.exit(f"No results directory at {results_dir}")

    report = Report()
    report.add(f"Results report for {results_dir}")
    report.add(f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}")

    if not args.no_checkpoints:
        show_checkpoints(report, results_dir)

    show_test_metrics(report, results_dir)
    show_training_history(report, results_dir)
    show_error_breakdowns(report, results_dir)
    show_threshold_sweeps(report, results_dir)
    show_run_logs(report, results_dir, args.log_tail)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(report.lines) + "\n", encoding="utf-8")
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
