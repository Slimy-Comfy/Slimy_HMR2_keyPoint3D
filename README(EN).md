# Slimy_HMR2_keyPoint3D

A 3D keypoint estimation node for ComfyUI, based on HMR2.0.  
**No WSL2 or Ubuntu required. Runs natively on Windows ComfyUI Portable.**

The hmr2 library is bundled locally within this node and does not affect other nodes.

---

## Background

This node was created as a companion to **VNCCSPoseStudio**, a ComfyUI node for pose-based workflows.  
The goal was to extend OpenPose-style 2D keypoint data with a **Z-axis (depth)** — something HMR2.0 is capable of estimating from a single image.

While a 4DHumans-based ComfyUI node already existed, getting it running on Windows was a significant barrier:

- **Detectron2**, used for person detection in the original 4DHumans, is notoriously difficult to build on Windows.
- The original codebase assumed a Linux + conda environment, with cache paths hardcoded to `~/.cache/`.
- Running it inside ComfyUI Portable (embedded Python, no WSL2) was not straightforward.

This node solves those issues by:

- Replacing Detectron2 with **YOLOv8** for person detection.
- Redirecting all model storage to `ComfyUI/models/hmr2/` instead of the user's home cache.
- Making everything work out of the box on **Windows ComfyUI Portable**.

---

## Installation

### 1. Place the node

Put this folder in `ComfyUI/custom_nodes/`.

### 2. Install dependencies

Run the following in PowerShell (adjust paths as needed):

```powershell
D:\ComfyUI_windows_portable\python_embeded\python.exe -m pip install -r D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\Slimy_HMR2_keyPoint3D\requirements.txt
```

### 3. First run (automatic download)

When you run the node for the first time, approximately 2.5GB of model data will be downloaded automatically and placed under `ComfyUI/models/hmr2/`:

```
ComfyUI/models/hmr2/
├── data/
│   ├── smpl/               ← Place SMPL_NEUTRAL.pkl here (see Step 4)
│   ├── smpl_mean_params.npz
│   └── SMPL_to_J19.pkl
└── logs/train/multiruns/hmr2/0/
    ├── checkpoints/
    │   └── epoch=35-step=1000000.ckpt
    └── model_config.yaml
```

After the download completes, the node will return an error because `SMPL_NEUTRAL.pkl` is missing. Proceed to Step 4.

### 4. Place the SMPL model (required, manual)

The SMPL model requires a separate download due to its academic license.

1. Go to https://smpl.is.tue.mpg.de/ and register
2. Download `SMPL for Python`
3. Extract `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` from the zip
4. Rename it to `SMPL_NEUTRAL.pkl`
5. Place it at:

```
ComfyUI/models/hmr2/data/smpl/SMPL_NEUTRAL.pkl
```

6. Restart ComfyUI and run the node again.

---

## Usage

| Input | Description |
|-------|-------------|
| `image` | Input image for inference |
| `person_index` | `0` = all people, `1` = first person, `1,2` = multiple |
| `output_filename` | Filename prefix for the output JSON |

| Output | Description |
|--------|-------------|
| `pose_image` | Input image with skeleton overlay |
| `skeleton_only` | Skeleton on black background |
| `pose_json` | 3D keypoint data (JSON) |

Use the **💾 Save JSON** button in the node to download the JSON directly from the browser.

---

## Notes

- The first run takes several minutes for model download and loading.
- NVIDIA GPU required.
- Internet connection required on first run.

---

## Credits

This node bundles a modified version of [4D-Humans (HMR2.0)](https://github.com/shubham-goel/4D-Humans) by Shubham Goel et al., licensed under the MIT License.
