"""Small thermal-frame classifier and shared checkpoint/metric helpers."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

FORMAT_VERSION = 1
CLASS_NAMES = {0: "no_fire", 1: "fire"}


class ThermalFireClassifier(nn.Module):
    """One thermal image in; one fire logit out (apply sigmoid for a score).

    Three channels describe the *same* rendered thermal frame, including its
    palette. No visible-light camera image is used.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.network = mobilenet_v3_small(weights=weights)
        features = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(features, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images).squeeze(1)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
    return device


def binary_metrics(labels, scores, threshold: float = 0.5) -> dict:
    """Fire is positive; a false negative means a missed fire frame."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != scores.shape or labels.size == 0:
        raise ValueError("Metrics need matching, nonempty one-dimensional arrays.")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Invalid labels or non-finite model scores.")
    predicted = scores >= threshold
    tp = int(((labels == 1) & predicted).sum())
    tn = int(((labels == 0) & ~predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "samples": int(labels.size),
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / labels.size),
        "balanced_accuracy": (recall + specificity) / 2,
        "fire_precision": precision,
        "fire_recall": recall,
        "fire_f1": 2 * precision * recall / (precision + recall)
        if precision + recall else 0.0,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def save_checkpoint(path, model, preprocessing, threshold, metadata=None):
    """Save plain metadata and tensors, without pickling a model object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "architecture": "mobilenet_v3_small_thermal_binary",
        "class_names": CLASS_NAMES,
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "preprocessing": preprocessing,
        "threshold": float(threshold),
        "metadata": metadata or {},
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format; use a checkpoint from train.py.")
    if checkpoint.get("architecture") != "mobilenet_v3_small_thermal_binary":
        raise ValueError("This checkpoint has a different architecture.")
    if checkpoint.get("class_names") != CLASS_NAMES:
        raise ValueError("Checkpoint fire/no-fire mapping does not match this program.")
    model = ThermalFireClassifier(pretrained=False)  # Never downloads weights here.
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


if __name__ == "__main__":
    model = ThermalFireClassifier().eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224))
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input: [batch, 3, 224, 224]; output logits: {tuple(output.shape)}")
    print("This is an architecture check, not a trained fire detector.")
