"""Train a thermal-only fire/no-fire model; reserve test.csv for test.py."""

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import ThermalFireClassifier, binary_metrics, choose_device, load_checkpoint, save_checkpoint
from preprocess import (DEFAULT_CONFIG, ThermalDataset, read_manifest,
                        validate_manifest_splits)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, sample_count, labels, scores = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; inspect images and learning rate.")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * targets.numel()
            sample_count += targets.numel()
            labels.extend(targets.detach().cpu().tolist())
            scores.extend(logits.detach().sigmoid().cpu().tolist())
    return total_loss / sample_count, labels, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/firehawk"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0, help="Increase to 2-4 on a training computer")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Default 0.5, or inherited from --init-checkpoint; choose using validation only")
    parser.add_argument("--no-pretrained", action="store_true", help="Skip ImageNet download; usually worse")
    parser.add_argument("--init-checkpoint", type=Path, help="Fine-tune a previous checkpoint on NAJA images")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 2 or args.image_size < 32
            or args.lr <= 0 or args.patience < 1 or args.workers < 0 or args.threads < 1):
        parser.error("Use positive settings, batch size >= 2, image size >= 32, workers >= 0.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    manifests = {name: read_manifest(args.data_dir / f"{name}.csv")
                 for name in ("train", "val", "test")}
    validate_manifest_splits(manifests)
    # Only filenames/labels are checked for test overlap. No test images or scores
    # are used for model selection, class weighting, or threshold selection.
    if args.init_checkpoint:
        model, previous = load_checkpoint(args.init_checkpoint, device)
        config = previous["preprocessing"]
        threshold = previous["threshold"] if args.threshold is None else args.threshold
    else:
        model = ThermalFireClassifier(pretrained=not args.no_pretrained).to(device)
        config = {**DEFAULT_CONFIG, "image_size": args.image_size}
        threshold = 0.5 if args.threshold is None else args.threshold
    if not 0 < threshold < 1:
        parser.error("--threshold must be between 0 and 1, exclusive.")
    if len(manifests["train"]) < 2:
        parser.error("At least two training examples are required.")
    datasets = {name: ThermalDataset(manifests[name], config, augment=(name == "train"))
                for name in ("train", "val")}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {name: DataLoader(dataset, batch_size=args.batch_size,
                               shuffle=(name == "train"), num_workers=args.workers,
                               worker_init_fn=seed_worker, generator=generator,
                               pin_memory=(device.type == "cuda"),
                               # BatchNorm cannot train on a final singleton at small sizes.
                               drop_last=(name == "train" and len(dataset) % args.batch_size == 1))
               for name, dataset in datasets.items()}
    positives = sum(row["label"] for row in manifests["train"])
    negatives = len(manifests["train"]) - positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    (args.output_dir / "training_config.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Device: {device}; input: {config['image_size']} square; threshold: {threshold}", flush=True)
    print(f"Training classes: {negatives} no_fire, {positives} fire", flush=True)
    best_score, stale = -1.0, 0
    fields = ["epoch", "train_loss", "val_loss", "val_balanced_accuracy", "val_fire_recall", "val_fire_precision", "seconds"]
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        history = csv.DictWriter(handle, fieldnames=fields)
        history.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_loss, _, _ = run_epoch(model, loaders["train"], criterion, device, optimizer)
            val_loss, labels, scores = run_epoch(model, loaders["val"], criterion, device)
            metrics = binary_metrics(labels, scores, threshold)
            score = metrics["balanced_accuracy"]
            scheduler.step(score)
            seconds = time.perf_counter() - started
            history.writerow(dict(zip(fields, [epoch, train_loss, val_loss, score,
                                               metrics["fire_recall"], metrics["fire_precision"], seconds])))
            handle.flush()
            print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"val_balanced_accuracy={score:.3f}, fire_recall={metrics['fire_recall']:.3f} "
                  f"({seconds:.1f}s)", flush=True)
            if score > best_score:
                best_score, stale = score, 0
                save_checkpoint(args.output_dir / "best.pt", model, config, threshold,
                                {"epoch": epoch, "validation_metrics": metrics,
                                 "seed": args.seed, "training_source": str(args.data_dir.resolve())})
                (args.output_dir / "best_validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
            else:
                stale += 1
            if stale >= args.patience:
                print("Early stopping: validation balanced accuracy stopped improving.", flush=True)
                break
    print(f"Saved best model: {args.output_dir / 'best.pt'}")
    print("Next: evaluate once with test.py, then validate on separate NAJA recordings.")


if __name__ == "__main__":
    main()
