"""Classify a thermal image, recorded thermal video, or USB thermal capture feed."""

import argparse
import json
from pathlib import Path
import queue
import threading
import time

import cv2
from PIL import Image
import torch

from model import choose_device, load_checkpoint
from preprocess import preprocess_bgr, preprocess_image


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
        raise ValueError("--roi x y width height must lie inside the captured frame.")
    return frame[y:y + height, x:x + width]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/firehawk/best.pt"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path)
    source.add_argument("--camera", help="USB capture index, e.g. 0, or Linux /dev/video0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--display", action="store_true", help="Open a preview; press q to stop")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means until EOF or Ctrl+C")
    parser.add_argument("--rate", type=float, default=5.0,
                        help="Target camera inference rate; actual rate depends on hardware")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="Crop to the thermal view if capture includes borders/overlays")
    args = parser.parse_args()
    if args.threads < 1 or args.rate <= 0 or args.max_frames < 0:
        parser.error("Use threads >= 1, rate > 0, max-frames >= 0.")
    if args.image and args.roi:
        parser.error("For --image, save the thermal crop first; --roi is for video/camera capture.")
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["preprocessing"]
    threshold = checkpoint["threshold"] if args.threshold is None else args.threshold
    if not 0 < threshold < 1:
        parser.error("Threshold must be between 0 and 1, exclusive.")

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

    if args.image:
        with Image.open(args.image) as image:
            result = classify(preprocess_image(image, config), 1)
        print(json.dumps(result), flush=True)
        return

    capture = None
    latest = None
    try:
        if args.camera is not None:
            camera = int(args.camera) if args.camera.isdecimal() else args.camera
            capture = cv2.VideoCapture(camera)
        else:
            if not args.video.is_file():
                raise ValueError(f"Video does not exist: {args.video}")
            capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise RuntimeError("Cannot open capture. Check USB adapter, video device, and permissions.")
        if args.camera is not None:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Backend may ignore this setting.
            latest = LatestCamera(capture).start()
        count = 0
        while args.max_frames == 0 or count < args.max_frames:
            iteration_started = time.monotonic()
            if latest:
                frame, dequeued_at = latest.read()
            else:
                ok, frame = capture.read()
                if not ok:
                    if count == 0:
                        raise RuntimeError("Video contains no decodable frames.")
                    break
            frame = crop_roi(frame, args.roi)
            count += 1
            result = classify(preprocess_bgr(frame, config), count)
            if latest:
                # Software read-to-result delay, not sensor-to-result latency.
                result["read_to_result_ms"] = round((time.monotonic() - dequeued_at) * 1000, 2)
            print(json.dumps(result), flush=True)
            if args.display:
                preview = frame.copy()
                color = (0, 0, 255) if result["fire_detected"] else (0, 200, 0)
                text = f"{result['label'].upper()}  score={result['fire_score']:.3f}"
                cv2.putText(preview, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.imshow("Thermal fire classifier", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if latest:
                delay = 1 / args.rate - (time.monotonic() - iteration_started)
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
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
