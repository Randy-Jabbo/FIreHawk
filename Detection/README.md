# FireHawk: one thermal frame -> fire / no fire

This is a training and inference starter project for the FLAME 2 dataset and a
NAJA TC256 thermal video feed. It uses MobileNetV3-Small with one binary output:
`0 = no_fire`, `1 = fire`. Train on a laptop or GPU computer; run `main.py` on the
Raspberry Pi 5. No RGB camera input, bounding boxes, or temperature data is used.

The five scripts are:

| File | Purpose |
| --- | --- |
| `preprocess.py` | Get labels from original FLAME 2 frame IDs, create data splits, and share image preprocessing. |
| `model.py` | Define the classifier and load/save its weights. |
| `train.py` | Train, validate, and save the best model. |
| `test.py` | Measure performance on held-out images. |
| `main.py` | Predict from one thermal image, a thermal video, or USB video capture. |

## Edit settings and run

Each runnable script has a `SETTINGS` dictionary near the top. Change its values,
save the file, then run the script with no command-line arguments. Relative paths
start in the folder containing these scripts, even if your terminal is elsewhere.
You can also use absolute paths such as `"/Users/yourname/Downloads/thermal"`.

For your layout, keep `254p Thermal Images` beside `preprocess.py` inside
`Detection`. Run these in order:

```bash
python preprocess.py
python train.py
python test.py
python main.py
```

The first three prepare data, train, and evaluate. Before running `main.py`, set
its `"source"` to `"image"`, `"video"`, or `"camera"`, and edit the corresponding
path/device. The default is an image named `254p Thermal Images/13701.jpg`; change
that value if your actual filename is different. `model.py` defines the network
and normally does not need editing.

## 1. Install

Use Python 3.11 or 3.12. These scripts were checked with Python 3.12. On the Pi,
use a 64-bit OS and a matching ARM64 PyTorch/torchvision wheel pair. The training
computer and Pi can have different CPU architectures; copy the checkpoint and
scripts, not the virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. For a CPU-only Linux
training computer, installing the CPU wheels first avoids downloading CUDA:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

`opencv-python-headless` supports image/video capture without a desktop window.
For `"display": True` on a desktop, replace it rather than installing both variants:

```bash
python -m pip uninstall -y opencv-python-headless
python -m pip install "opencv-python>=4.8,<5"
```

## 2. Prepare the right FLAME 2 images

Get the official **254p Frame Pairs** archive from
[FLAME 2 on IEEE DataPort](https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset).
Extract it and locate **254p Thermal Images**. Point the script at that folder
only. Do not include the RGB folder. Keep the original global frame numbers.

The dataset authors' [labeling code](https://github.com/xiwenc1/Flame_2_dataset/blob/main/dataset.py)
uses these IDs. Collapsing its smoke categories gives:

| Original frame ID | Our label |
| --- | --- |
| 1 through 13700 | `0` / `no_fire` |
| 13701 through 53451 | `1` / `fire` |

Those numbers apply to the prepared, globally numbered sequence. They do **not**
label arbitrary frames extracted from a raw video or your own camera. The
filename must have exactly one numeric group, such as `13701.jpg` or
`frame_13701.jpg`. Unknown and duplicate IDs are rejected. Files must actually
be thermal images; a filename alone cannot establish which sensor produced it.

The original fire label describes an RGB/thermal **pair**: fire could be observed
in either view. It is not a separate annotation proving that fire is visible in
every positive thermal frame. Audit thermal examples and validate on your own
thermal-only labels before drawing conclusions about camera performance.

```bash
python preprocess.py
```

This writes `train.csv`, `val.csv`, `test.csv`, and `dataset_summary.json`.
Images remain in place, so keep their folder or rerun preprocessing after moving
them. Each CSV links an image path to its label. The model sees the image pixels,
not the filename or frame ID.

The default split is chronological within each class: approximately 70% training,
15% validation, and 15% testing, before removing boundary frames. It discards
150 frame IDs on **each side** of every split boundary, including a boundary
between the two label ranges. This reduces adjacent-frame leakage. FLAME 2 still
contains related footage from one burn, so these splits do not demonstrate
performance on a new location, new fire, or NAJA camera. They also do not provide
a smoke-only negative class.

For a tiny development subset, a smaller `SETTINGS["gap_frames"]` may be necessary to keep
both classes in every split. Use the default gap for the full dataset.

## 3. Train

```bash
python train.py
```

The first run downloads ImageNet starting weights. Setting `"pretrained": False` avoids that
download, but training from scratch usually needs more data. The script uses
class-weighted binary cross-entropy, horizontal flips, and early stopping. It
selects `best.pt` by validation balanced accuracy; it never evaluates test images
during training. Reduce `SETTINGS["batch_size"]` if GPU memory runs out.

Outputs: `best.pt`, `history.csv`, `training_config.json`, and
`best_validation.json`. The checkpoint includes the image preprocessing and
decision threshold. It contains inference weights, not optimizer state for an
exact interrupted-training resume.

The default model input is 224 x 224. Both training images and camera frames are
resized to fit that square with padding, preserving the entire frame and its
aspect ratio. Do **not** stretch every source image to 256 x 192 just because
the TC256 sensor has that resolution. Resizing cannot reproduce its field of
view, detail, palette, or automatic contrast behavior.

The network uses three image channels to retain a rendered thermal palette.
These are three channels of **one thermal image**, not a second RGB sensor.
Use the closest available NAJA palette to your training images. Turning a
pseudocolor image gray does not reconstruct the original temperature values.

## 4. Evaluate on the held-out set

```bash
python test.py
```

Read `fire_recall` (how many fire frames were caught), `fire_precision` (how many
fire predictions were correct), and the confusion matrix. A false negative is a
missed fire. The output includes `metrics.json` and per-image `predictions.csv`.

The default decision threshold is 0.5. If choosing another value, choose it using
validation data first, then evaluate once on the test set. A sigmoid score is
not a calibrated probability. Lowering the threshold generally trades more
false alarms for fewer missed fires; verify that tradeoff with your footage.

## 5. Run a prediction

On the Pi, copy this folder and your trained `best.pt`, install dependencies,
and start with a saved thermal image:

```bash
python main.py
```

The JSON output includes `fire_detected`, `label`, `fire_score`, and `model_ms`.
`model_ms` measures the model call, not complete capture-to-result latency.

For a recorded thermal video, edit these values in `main.py` and run it again:

```python
"source": "video",
"video": "thermal_recording.mp4",
```

For the TC256, this implementation expects its **thermal video output** to reach
the Pi through a compatible **CVBS-to-USB capture adapter** that Linux exposes
as a video device. The NAJA camera is not assumed to be a native USB webcam or
CSI camera. Check the exact module pinout, output selection, PAL compatibility,
adapter driver, and power requirements before connecting it.

Set these values in `main.py`, then run `python main.py`:

```python
"source": "camera",
"camera": 0,
"rate": 5.0,
```

The camera setting can also be `"/dev/video0"` on Linux. Capture dimensions can differ from the sensor's
256 x 192 dimensions because analog capture produces a video raster. Confirm
that the feed contains the thermal image alone, with the intended palette. If
the adapter feed contains borders or another view, set `"roi": (X, Y, W, H)` to select
the thermal area. Remove embedded overlays at the source when possible; a crop
cannot remove an overlay covering useful scene pixels. Save the same thermal
crop when collecting images for fine-tuning.

The camera reader keeps the latest software frame instead of queuing old frames
behind inference. `"rate": 5.0` is a target, not a promised Pi frame rate. Camera
errors produce an error, not a `no_fire` result. Ctrl+C stops capture. Set
`"display": True` only with desktop OpenCV installed. No physical camera was available
for the automated checks.

## 6. Make it work with your actual NAJA camera

Training on FLAME 2 alone does not establish accuracy on a TC256. Before trusting
the output, record and label independent NAJA sessions: visible fires of different
sizes/distances, no-fire scenes, and confusing objects such as sun-heated surfaces,
people, vehicles, and exhaust. The target here is visible **fire**, not every hot
object. A hidden or smoldering hotspot is not validated by these labels.

Create your own CSV. Paths are relative to this CSV. Put whole recording sessions
in one split, with both classes represented in each split. Do not scatter adjacent
frames from one recording across train/val/test.

```csv
file,label,split
naja/session_a/frame001.jpg,no_fire,train
naja/session_b/frame001.jpg,fire,train
naja/session_c/frame001.jpg,no_fire,val
naja/session_d/frame001.jpg,fire,val
naja/session_e/frame001.jpg,no_fire,test
naja/session_f/frame001.jpg,fire,test
```

This is a format example, not enough images to train a useful model. Once you
have a representative labeled collection:

Change the settings in each script as follows:

| Script | Settings to change |
| --- | --- |
| `preprocess.py` | `"labels_csv": "naja_labels.csv"`, `"output_dir": "data/naja"`. The CSV takes priority over `data_dir`. |
| `train.py` | `"data_dir": "data/naja"`, `"init_checkpoint": "runs/firehawk/best.pt"`, `"output_dir": "runs/naja"`, `"lr": 0.00001`. |
| `test.py` | `"checkpoint": "runs/naja/best.pt"`, `"manifest": "data/naja/test.csv"`, `"output_dir": "runs/naja_evaluation"`. |
| `main.py` | `"checkpoint": "runs/naja/best.pt"`, `"source": "camera"`, `"camera": 0`. |

Save each file and run `preprocess.py`, `train.py`, `test.py`, and `main.py` in
that order, with no command-line arguments.

Fine-tuning inherits the original checkpoint's input size and preprocessing.
Check performance on the held-out NAJA sessions at your required distances,
lighting conditions, and Pi processing rate. This package supplies working code;
it does not include a trained checkpoint or a measured TC256 detection accuracy.

## Code verification

Checked with Python 3.12, PyTorch 2.14.1 (CPU), torchvision 0.29.1, OpenCV 4.14.0,
NumPy 2.3.5, and Pillow 12.3.0. Checks covered label boundaries, guarded splitting
over all 53,451 original IDs, matching PIL/OpenCV preprocessing, checkpoint reloads,
capture failure handling, and keeping the latest camera frame. Generated image
fixtures exercised preprocessing, training from scratch, ImageNet initialization
at the default 224 input size, evaluation, image/video predictions, and fine-tuning.
The editable-settings version was also checked by running preprocessing, training,
evaluation, and image/video predictions without command-line arguments, from a
different working directory. A simulated capture checked camera-source selection.
No FLAME 2 accuracy benchmark, physical NAJA test, or Pi performance test was run.

## Sources

- [FLAME 2 authors' repository and label mapping](https://github.com/xiwenc1/Flame_2_dataset)
- [Official FLAME 2 dataset](https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset)
- [Dataset authors' description of pair-level labels](https://open.clemson.edu/all_data/3173/)
- [Torchvision MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
- [OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [NAJA dual camera manufacturer](https://www.axisflying.com/products/eco-naja-dual-camera-model-256-384-640-fpv-diy-cameras-for-day-and-night)
