"""Train a thermal-only fire/no-fire model; reserve test.csv for test.py."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import ThermalFireClassifier, binary_metrics, choose_device, load_checkpoint, save_checkpoint
from preprocess import (DEFAULT_CONFIG, ThermalDataset, read_manifest,
                        validate_manifest_splits, script_path)


# EDIT THESE SETTINGS, then run: python train.py
# Relative paths start in the folder containing these scripts.
SETTINGS = {
    "data_dir": "data/prepared",
    "output_dir": "runs/firehawk",
    "epochs": 25,
    "batch_size": 32,
    "image_size": 224,
    "lr": 0.0001,
    "patience": 5,
    "workers": 0,            # Increase to 2-4 on a training computer if useful.
    "threads": 4,
    "device": "auto",        # "auto", "cpu", "cuda", or "mps".
    "seed": 42,
    "threshold": None,       # None = 0.5, or inherit when fine-tuning.
    "pretrained": True,      # Download/use ImageNet starting weights.
    "init_checkpoint": None, # Set "runs/firehawk/best.pt" to fine-tune on NAJA data.
}


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
    settings = SimpleNamespace(**SETTINGS)
    settings.data_dir = script_path(settings.data_dir)
    settings.output_dir = script_path(settings.output_dir)
    settings.init_checkpoint = script_path(settings.init_checkpoint)
    if (settings.epochs < 1 or settings.batch_size < 2 or settings.image_size < 32
            or settings.lr <= 0 or settings.patience < 1 or settings.workers < 0 or settings.threads < 1):
        raise ValueError("Use positive settings, batch size >= 2, image size >= 32, workers >= 0.")
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.set_num_threads(settings.threads)
    device = choose_device(settings.device)
    manifests = {name: read_manifest(settings.data_dir / f"{name}.csv")
                 for name in ("train", "val", "test")}
    validate_manifest_splits(manifests)
    # Only filenames/labels are checked for test overlap. No test images or scores
    # are used for model selection, class weighting, or threshold selection.
    if settings.init_checkpoint:
        model, previous = load_checkpoint(settings.init_checkpoint, device)
        config = previous["preprocessing"]
        threshold = previous["threshold"] if settings.threshold is None else settings.threshold
    else:
        model = ThermalFireClassifier(pretrained=settings.pretrained).to(device)
        config = {**DEFAULT_CONFIG, "image_size": settings.image_size}
        threshold = 0.5 if settings.threshold is None else settings.threshold
    if not 0 < threshold < 1:
        raise ValueError("SETTINGS['threshold'] must be between 0 and 1, exclusive.")
    if len(manifests["train"]) < 2:
        raise ValueError("At least two training examples are required.")
    datasets = {name: ThermalDataset(manifests[name], config, augment=(name == "train"))
                for name in ("train", "val")}
    generator = torch.Generator().manual_seed(settings.seed)
    loaders = {name: DataLoader(dataset, batch_size=settings.batch_size,
                               shuffle=(name == "train"), num_workers=settings.workers,
                               worker_init_fn=seed_worker, generator=generator,
                               pin_memory=(device.type == "cuda"),
                               # BatchNorm cannot train on a final singleton at small sizes.
                               drop_last=(name == "train" and len(dataset) % settings.batch_size == 1))
               for name, dataset in datasets.items()}
    positives = sum(row["label"] for row in manifests["train"])
    negatives = len(manifests["train"]) - positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    saved_settings = {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(settings).items()}
    (settings.output_dir / "training_config.json").write_text(json.dumps(saved_settings, indent=2) + "\n")
    print(f"Device: {device}; input: {config['image_size']} square; threshold: {threshold}", flush=True)
    print(f"Training classes: {negatives} no_fire, {positives} fire", flush=True)
    best_score, stale = -1.0, 0
    fields = ["epoch", "train_loss", "val_loss", "val_balanced_accuracy", "val_fire_recall", "val_fire_precision", "seconds"]
    with (settings.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        history = csv.DictWriter(handle, fieldnames=fields)
        history.writeheader()
        for epoch in range(1, settings.epochs + 1):
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
                save_checkpoint(settings.output_dir / "best.pt", model, config, threshold,
                                {"epoch": epoch, "validation_metrics": metrics,
                                 "seed": settings.seed, "training_source": str(settings.data_dir.resolve())})
                (settings.output_dir / "best_validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
            else:
                stale += 1
            if stale >= settings.patience:
                print("Early stopping: validation balanced accuracy stopped improving.", flush=True)
                break
    print(f"Saved best model: {settings.output_dir / 'best.pt'}")
    print("Next: evaluate once with test.py, then validate on separate NAJA recordings.")


if __name__ == "__main__":
    main()
