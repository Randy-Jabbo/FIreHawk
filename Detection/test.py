"""Evaluate a trained checkpoint on held-out thermal frames, without retraining."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from model import binary_metrics, choose_device, load_checkpoint
from preprocess import ThermalDataset, read_manifest, script_path


# EDIT THESE SETTINGS, then run: python test.py
# Relative paths start in the folder containing these scripts.
SETTINGS = {
    "checkpoint": "runs/firehawk/best.pt",
    "manifest": "data/prepared/test.csv",
    "output_dir": "runs/evaluation",
    "batch_size": 32,
    "workers": 0,
    "threads": 4,
    "device": "auto",
    "threshold": None,       # None = use the trained checkpoint's threshold.
}


def main():
    settings = SimpleNamespace(**SETTINGS)
    settings.checkpoint = script_path(settings.checkpoint)
    settings.manifest = script_path(settings.manifest)
    settings.output_dir = script_path(settings.output_dir)
    if settings.batch_size < 1 or settings.workers < 0 or settings.threads < 1:
        raise ValueError("Invalid batch size, workers, or threads.")
    torch.set_num_threads(settings.threads)
    device = choose_device(settings.device)
    model, checkpoint = load_checkpoint(settings.checkpoint, device)
    threshold = checkpoint["threshold"] if settings.threshold is None else settings.threshold
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1, exclusive.")
    rows = read_manifest(settings.manifest)
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Evaluation needs both fire and no-fire examples.")
    if len({row['file'] for row in rows}) != len(rows):
        raise ValueError("Evaluation manifest contains duplicate images.")
    dataset = ThermalDataset(rows, checkpoint["preprocessing"])
    loader = DataLoader(dataset, batch_size=settings.batch_size, num_workers=settings.workers)
    labels, scores = [], []
    with torch.inference_mode():
        for images, targets in loader:
            scores.extend(model(images.to(device)).sigmoid().cpu().tolist())
            labels.extend(targets.tolist())
    metrics = binary_metrics(labels, scores, threshold)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (settings.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "true_label", "fire_score", "predicted_label"])
        for row, score in zip(rows, scores):
            writer.writerow([str(row["file"]), row["label"], score, int(score >= threshold)])
    print(json.dumps(metrics, indent=2))
    print("Confusion matrix: rows=actual [no_fire, fire], columns=predicted [no_fire, fire]")
    print([[metrics["true_negative"], metrics["false_positive"]],
           [metrics["false_negative"], metrics["true_positive"]]])
    print(f"Saved metrics.json and predictions.csv in {settings.output_dir}")


if __name__ == "__main__":
    main()
