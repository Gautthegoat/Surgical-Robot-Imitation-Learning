# Surgical Robot Imitation Learning

Needle pick-and-hand-off on a dual-arm surgical robot using imitation learning. This repo contains a simple training engine (ClassicIL), a dataset pipeline, and an ACT-style policy for predicting future joint trajectories from video frames.

## Demo

<video controls src="Archive/videos/needle_handoff_demo.mp4" width="720"></video>

[Watch the hand-off demo (MP4)](Archive/videos/needle_handoff_demo.mp4)

## Highlights

- Classic imitation learning engine with train/resume/visualize/export modes.
- ACT-style model predicting short action sequences from images (+ optional style embedding).
- Dataset tools to extract frames from video and align robot logs.
- Automatic plots for loss curves and action distributions, archived per run.

## Quickstart

1) Environment (Python 3.10+ recommended):

```
pip install torch torchvision
pip install numpy pandas pyyaml matplotlib albumentations opencv-python av colorama
```

2) Configure data paths in `source/configs/act_il.yaml` (`train_data_path`, `val_data_path`).

3) Train / Visualize / Export:

```
python main.py train --engine ClassicIL --config act_il
python main.py visualize --archive_model ClassicIL_ACTModel_Vanilla_20240907_194141
python main.py export --archive_model ClassicIL_ACTModel_Vanilla_20240907_194141
```

## Data (expected)

- Each dataset root contains one or more `...demos/` folders with `demo_*` subfolders.
- Each `demo_*` has an `index.csv` referencing saved frames (`frame_*.npy`) and associated robot logs.
- Frames are stored as RGB numpy arrays; logs include joint angles used as training targets.
- Basic transforms/normalization and optional augmentations are configured in `act_il.yaml`.

## Results (example)

<img src="Archive/ClassicIL_ACTModel_Vanilla_20240907_194141/plots/train_loss.png" width="420"> <img src="Archive/ClassicIL_ACTModel_Vanilla_20240907_194141/plots/val_loss.png" width="420">

<img src="Archive/ClassicIL_ACTModel_Vanilla_20240907_194141/plots/train_distr/joint_1.png" width="420"> <img src="Archive/ClassicIL_ACTModel_Vanilla_20240907_194141/plots/val_distr/joint_1.png" width="420">

## Repo Tips

- All training artifacts are stored under `Archive/<ENGINE>_<MODEL>_<TAG>_<TIMESTAMP>/` (config, logs, plots, checkpoints).
- Engine entrypoint: `main.py` with modes `train|resume|visualize|export`.
- Key files: `source/engines/`, `source/models/`, `source/datasets/`, `source/configs/act_il.yaml`.

---

Note: Update the demo video path if your file name under `Archive/videos/` differs from `needle_handoff_demo.mp4`.
