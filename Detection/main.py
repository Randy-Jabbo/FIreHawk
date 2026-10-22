"""Classify a thermal image, recorded thermal video, or USB thermal capture feed."""

import json
from pathlib import Path
from types import SimpleNamespace
import queue
import threading
import time

import cv2
from PIL import Image
import torch

from model import choose_device, load_checkpoint
from preprocess import preprocess_bgr, preprocess_image, script_path


# EDIT THESE SETTINGS, then run: python main.py
# Relative paths start in the folder containing these scripts.
SETTINGS = {
    "source": "image",       # Choose "image", "video", or "camera".
    "checkpoint": "runs/firehawk/best.pt",
    "image": "254p Thermal Images/13701.jpg", # Change to an actual thermal filename.
    "video": "thermal_recording.mp4",
    "camera": 0,             # USB capture index, or "/dev/video0" on Linux.
    "device": "cpu",
    "threads": 4,
    "threshold": None,       # None = use the trained checkpoint's threshold.
    "display": False,       # True needs desktop OpenCV rather than headless OpenCV.
    "max_frames": 0,         # 0 = until video ends or you press Ctrl+C.
    "rate": 5.0,             # Target camera predictions per second.
    "roi": None,            # Optional video/camera crop: (x, y, width, height).
}


class LatestCamera:
    """Continuously drain capture into a ONE-frame queue to avoid inference backlog."""

    def __init__(self, capture):
        self.capture = capture
        self.frames = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._read, daemon=True)

    def _read(self):
        try:
            while not self.stopped.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    self.error = "Camera capture failed. No fire/no-fire decision is available."
                    return
                item = (frame, time.monotonic())
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                self.frames.put_nowait(item)
        except Exception as error:
            self.error = f"Camera capture failed: {error}"

    def start(self):
        self.thread.start()
        return self

    def read(self):
        if self.error:
            raise RuntimeError(self.error)
        try:
            item = self.frames.get(timeout=5)
        except queue.Empty as error:
            raise RuntimeError("No camera frames received within 5 seconds.") from error
        if self.error:
            raise RuntimeError(self.error)
        return item

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)


def crop_roi(frame, roi):
    if roi is None:
        return frame
    x, y, width, height = roi
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
        raise ValueError("ROI (x, y, width, height) must lie inside the captured frame.")
    return frame[y:y + height, x:x + width]


def main():
    settings = SimpleNamespace(**SETTINGS)
    if settings.source not in {"image", "video", "camera"}:
        raise ValueError("Set SETTINGS['source'] to 'image', 'video', or 'camera'.")
    settings.checkpoint = script_path(settings.checkpoint)
    settings.image = script_path(settings.image) if settings.source == "image" else None
    settings.video = script_path(settings.video) if settings.source == "video" else None
    settings.camera = str(settings.camera) if settings.source == "camera" and settings.camera is not None else None
    if settings.source == "image" and settings.image is None:
        raise ValueError("Set SETTINGS['image'] to a thermal image path.")
    if settings.source == "video" and settings.video is None:
        raise ValueError("Set SETTINGS['video'] to a thermal video path.")
    if settings.source == "camera" and settings.camera is None:
        raise ValueError("Set SETTINGS['camera'] to a USB capture index or device path.")
    if settings.threads < 1 or settings.rate <= 0 or settings.max_frames < 0:
        raise ValueError("Use threads >= 1, rate > 0, max-frames >= 0.")
    if settings.image and settings.roi:
        raise ValueError("In image mode, save the thermal crop first; roi is for video/camera capture.")
    torch.set_num_threads(settings.threads)
    device = choose_device(settings.device)
    model, checkpoint = load_checkpoint(settings.checkpoint, device)
    config = checkpoint["preprocessing"]
    threshold = checkpoint["threshold"] if settings.threshold is None else settings.threshold
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1, exclusive.")

    def classify(tensor, frame_number):
        started = time.perf_counter()
        with torch.inference_mode():
            score = model(tensor.unsqueeze(0).to(device)).sigmoid().item()
        if not 0.0 <= score <= 1.0:
            raise RuntimeError("Model produced an invalid score; no decision is available.")
        detected = score >= threshold
        return {"frame": frame_number, "fire_detected": detected,
                "label": "fire" if detected else "no_fire", "fire_score": round(score, 6),
                "threshold": threshold,
                "model_ms": round((time.perf_counter() - started) * 1000, 2)}

    if settings.image:
        with Image.open(settings.image) as image:
            result = classify(preprocess_image(image, config), 1)
        print(json.dumps(result), flush=True)
        return

    capture = None
    latest = None
    try:
        if settings.camera is not None:
            camera = int(settings.camera) if settings.camera.isdecimal() else settings.camera
            capture = cv2.VideoCapture(camera)
        else:
            if not settings.video.is_file():
                raise ValueError(f"Video does not exist: {settings.video}")
            capture = cv2.VideoCapture(str(settings.video))
        if not capture.isOpened():
            raise RuntimeError("Cannot open capture. Check USB adapter, video device, and permissions.")
        if settings.camera is not None:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Backend may ignore this setting.
            latest = LatestCamera(capture).start()
        count = 0
        while settings.max_frames == 0 or count < settings.max_frames:
            iteration_started = time.monotonic()
            if latest:
                frame, dequeued_at = latest.read()
            else:
                ok, frame = capture.read()
                if not ok:
                    if count == 0:
                        raise RuntimeError("Video contains no decodable frames.")
                    break
            frame = crop_roi(frame, settings.roi)
            count += 1
            result = classify(preprocess_bgr(frame, config), count)
            if latest:
                # Software read-to-result delay, not sensor-to-result latency.
                result["read_to_result_ms"] = round((time.monotonic() - dequeued_at) * 1000, 2)
            print(json.dumps(result), flush=True)
            if settings.display:
                preview = frame.copy()
                color = (0, 0, 255) if result["fire_detected"] else (0, 200, 0)
                text = f"{result['label'].upper()}  score={result['fire_score']:.3f}"
                cv2.putText(preview, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.imshow("Thermal fire classifier", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if latest:
                delay = 1 / settings.rate - (time.monotonic() - iteration_started)
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        if latest:
            latest.close()
        # A blocked driver read must finish before another thread releases capture.
        if capture is not None and (latest is None or not latest.thread.is_alive()):
            capture.release()
        if settings.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
