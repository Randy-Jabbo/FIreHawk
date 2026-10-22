"""Build labeled manifests and apply identical training/inference preprocessing."""

import bisect
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
import random
import re

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

# EDIT THESE SETTINGS, then run: python preprocess.py
# Relative paths start in the folder containing these scripts.
SCRIPT_DIR = Path(__file__).resolve().parent
SETTINGS = {
    "data_dir": "254p Thermal Images",  # Your original FLAME 2 thermal folder.
    "labels_csv": None,                # For your own labeled images, set a CSV path.
                                      # When set, labels_csv takes priority over data_dir.
    "output_dir": "data/prepared",
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "gap_frames": 150,                 # Guard frames on EACH side of split boundaries.
    "verify_images": True,
}


def script_path(value):
    """Resolve an editable file/folder setting relative to this script folder."""
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
DEFAULT_CONFIG = {
    "image_size": 224,
    "resize": "letterbox",
    "color_mode": "rendered_thermal_rgb",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "padding_rgb": [124, 116, 104],
}


def flame2_label(frame_id: int) -> int:
    """Pair-level label for the original globally numbered 254p FLAME 2 pairs."""
    if not 1 <= frame_id <= 53451:
        raise ValueError(f"FLAME 2 frame ID {frame_id} is outside 1..53451.")
    return int(frame_id >= 13701)


def scan_flame2(folder: Path) -> list[dict]:
    rows = []
    seen = set()

    if not folder.is_dir():
        raise ValueError(f"Thermal image directory does not exist: {folder}")

    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Original filename: 254p Thermal Frame (1).jpg
        match = re.fullmatch(
            r"254p\s+Thermal\s+Frame\s*\((\d+)\)",
            path.stem,
            flags=re.IGNORECASE,
        )

        if match:
            frame_id = int(match.group(1))
        else:
            # Also allow names such as 1.jpg or frame_1.jpg.
            numbers = re.findall(r"\d+", path.stem)

            if len(numbers) != 1:
                raise ValueError(
                    f"Cannot identify the frame ID in {path.name!r}. "
                    "Keep the original FLAME 2 filenames."
                )

            frame_id = int(numbers[0])

        if frame_id in seen:
            raise ValueError(
                f"Duplicate frame ID {frame_id}. "
                "Select only the thermal image folder."
            )

        seen.add(frame_id)

        rows.append({
            "file": path.resolve(),
            "label": flame2_label(frame_id),
            "frame_id": frame_id,
        })

    if not rows:
        raise ValueError(f"No supported images found in {folder}.")

    return sorted(rows, key=lambda row: row["frame_id"])


def chronological_split(rows, val_fraction=0.15, test_fraction=0.15, gap_frames=150):
    """Split contiguous sections within each class, then guard EVERY boundary.

    This also guards the no-fire/fire label boundary if its split changes.
    It reduces adjacent-frame leakage; it cannot create independent burn events.
    """
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1
            and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than 1.")
    if gap_frames < 0:
        raise ValueError("SETTINGS['gap_frames'] must be nonnegative.")
    assigned = []
    for label in (0, 1):
        group = sorted((r for r in rows if r["label"] == label),
                       key=lambda r: r["frame_id"])
        n_val, n_test = max(1, int(len(group) * val_fraction)), max(1, int(len(group) * test_fraction))
        n_train = len(group) - n_val - n_test
        if n_train < 1:
            raise ValueError("Need at least three images of EACH class to create splits.")
        for index, row in enumerate(group):
            split = "train" if index < n_train else "val" if index < n_train + n_val else "test"
            assigned.append({**row, "split": split})
    assigned.sort(key=lambda row: row["frame_id"])
    boundaries = [(a["frame_id"] + b["frame_id"]) / 2
                  for a, b in zip(assigned, assigned[1:]) if a["split"] != b["split"]]
    kept = []
    for row in assigned:
        index = bisect.bisect_left(boundaries, row["frame_id"])
        neighbors = boundaries[max(0, index - 1):index + 1]
        if any(abs(row["frame_id"] - boundary) <= gap_frames for boundary in neighbors):
            continue
        kept.append(row)
    validate_rows(kept)
    return kept, len(rows) - len(kept)


def read_manual_labels(path: Path) -> list[dict]:
    """Own-camera CSV: file,label,split. Assign whole recording sessions together."""
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"file", "label", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Manual CSV must contain file,label,split columns.")
        for source in reader:
            file = Path(source["file"]).expanduser()
            if not file.is_absolute():
                file = path.parent / file
            label_text = source["label"].strip().lower()
            mapping = {"0": 0, "no_fire": 0, "1": 1, "fire": 1}
            if label_text not in mapping:
                raise ValueError(f"Invalid label {label_text!r}; use 0/1 or no_fire/fire.")
            rows.append({"file": file.resolve(), "label": mapping[label_text],
                         "split": source["split"].strip().lower(), "frame_id": ""})
    validate_rows(rows)
    return rows


def validate_rows(rows):
    seen = set()
    for row in rows:
        if row["split"] not in SPLITS:
            raise ValueError(f"Unknown split {row['split']!r}; use train,val,test.")
        if row["file"] in seen:
            raise ValueError(f"An image appears more than once: {row['file']}")
        if not row["file"].is_file():
            raise ValueError(f"Missing image: {row['file']}")
        seen.add(row["file"])
    for split in SPLITS:
        if {r["label"] for r in rows if r["split"] == split} != {0, 1}:
            raise ValueError(
                f"{split} needs both fire and no-fire images after applying the guard gap. "
                "Use more data or a smaller SETTINGS['gap_frames'] for a small development subset."
            )


def read_manifest(path: Path) -> list[dict]:
    path = Path(path).resolve()
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"file", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain file,label columns.")
        for source in reader:
            file = Path(source["file"])
            file = file if file.is_absolute() else path.parent / file
            label = int(source["label"])
            if label not in (0, 1):
                raise ValueError(f"Invalid label in {path}: {label}")
            if not file.is_file():
                raise ValueError(f"Missing image: {file}")
            rows.append({"file": file.resolve(), "label": label,
                         "frame_id": source.get("frame_id", "")})
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def validate_manifest_splits(manifests):
    """Protect training/evaluation even if prepared CSVs are later edited."""
    seen = set()
    for name, rows in manifests.items():
        if {row["label"] for row in rows} != {0, 1}:
            raise ValueError(f"{name} must contain both classes.")
        for row in rows:
            if row["file"] in seen:
                raise ValueError(f"Duplicate image across/in splits: {row['file']}")
            seen.add(row["file"])


def preprocess_image(image: Image.Image, config: dict, augment=False) -> torch.Tensor:
    """Preserve the full frame; pad to a square rather than cropping away fire."""
    if config.get("resize") != "letterbox" or config.get("color_mode") != "rendered_thermal_rgb":
        raise ValueError("Unsupported checkpoint preprocessing configuration.")
    size = int(config["image_size"])
    if size < 32:
        raise ValueError("Image size must be at least 32 pixels.")
    if image.mode not in {"RGB", "RGBA", "L", "LA", "P", "1"}:
        raise ValueError("Use 8-bit rendered thermal images, not raw/radiometric arrays.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    if augment and random.random() < 0.5:
        image = ImageOps.mirror(image)
    image = ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR,
                         color=tuple(config["padding_rgb"]), centering=(0.5, 0.5))
    array = np.array(image, dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(config["mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(config["std"], dtype=torch.float32)[:, None, None]
    return ((tensor - mean) / std).contiguous()


def preprocess_bgr(frame: np.ndarray, config: dict) -> torch.Tensor:
    """OpenCV supplies BGR; PIL/dataset images supply RGB. Convert explicitly."""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected an 8-bit, 3-channel rendered thermal capture frame.")
    return preprocess_image(Image.fromarray(frame[:, :, ::-1].copy()), config)


class ThermalDataset(Dataset):
    def __init__(self, rows, config, augment=False):
        self.rows, self.config, self.augment = rows, config, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            with Image.open(row["file"]) as image:
                tensor = preprocess_image(image, self.config, self.augment)
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read thermal image {row['file']}: {error}") from error
        return tensor, torch.tensor(row["label"], dtype=torch.float32)


def main():
    settings = SimpleNamespace(**SETTINGS)
    settings.labels_csv = script_path(settings.labels_csv)
    settings.data_dir = None if settings.labels_csv is not None else script_path(settings.data_dir)
    settings.output_dir = script_path(settings.output_dir)
    if settings.labels_csv is None and settings.data_dir is None:
        raise ValueError("Set SETTINGS['data_dir'] or SETTINGS['labels_csv'].")
    if settings.data_dir:
        original = scan_flame2(settings.data_dir)
        rows, discarded = chronological_split(original, settings.val_fraction,
                                               settings.test_fraction, settings.gap_frames)
        source_name = "FLAME2_original_254p_global_frame_ids"
    else:
        rows = read_manual_labels(settings.labels_csv.resolve())
        discarded, source_name = 0, "manual_session_splits"
    if settings.verify_images:
        for row in rows:
            with Image.open(row["file"]) as image:
                image.load()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"source": source_name, "discarded_boundary_frames": discarded,
               "gap_frames_each_side": settings.gap_frames if settings.data_dir else None,
               "class_names": {"0": "no_fire", "1": "fire"}, "splits": {}}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        path = settings.output_dir / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["file", "label", "frame_id"])
            writer.writeheader()
            for row in selected:
                writer.writerow({"file": os.path.relpath(row["file"], path.parent.resolve()),
                                 "label": row["label"], "frame_id": row["frame_id"]})
        summary["splits"][split] = {
            "no_fire": sum(row["label"] == 0 for row in selected),
            "fire": sum(row["label"] == 1 for row in selected),
        }
    (settings.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Prepared CSV manifests. Images were not copied or permanently resized.")


if __name__ == "__main__":
    main()
