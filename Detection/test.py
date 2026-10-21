"""Evaluate a trained checkpoint on held-out thermal frames, without retraining."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import binary_metrics, choose_device, load_checkpoint
from preprocess import ThermalDataset, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/firehawk/best.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("data/prepared/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override only with a threshold chosen using validation data")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.threads < 1:
        parser.error("Invalid batch size, workers, or threads.")
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    threshold = checkpoint["threshold"] if args.threshold is None else args.threshold
    if not 0 < threshold < 1:
        parser.error("Threshold must be between 0 and 1, exclusive.")
    rows = read_manifest(args.manifest)
    if {row["label"] for row in rows} != {0, 1}:
        parser.error("Evaluation needs both fire and no-fire examples.")
    if len({row['file'] for row in rows}) != len(rows):
        parser.error("Evaluation manifest contains duplicate images.")
    dataset = ThermalDataset(rows, checkpoint["preprocessing"])
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    labels, scores = [], []
    with torch.inference_mode():
        for images, targets in loader:
            scores.extend(model(images.to(device)).sigmoid().cpu().tolist())
            labels.extend(targets.tolist())
    metrics = binary_metrics(labels, scores, threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "true_label", "fire_score", "predicted_label"])
        for row, score in zip(rows, scores):
            writer.writerow([str(row["file"]), row["label"], score, int(score >= threshold)])
    print(json.dumps(metrics, indent=2))
    print("Confusion matrix: rows=actual [no_fire, fire], columns=predicted [no_fire, fire]")
    print([[metrics["true_negative"], metrics["false_positive"]],
           [metrics["false_negative"], metrics["true_positive"]]])
    print(f"Saved metrics.json and predictions.csv in {args.output_dir}")


if __name__ == "__main__":
    main()
