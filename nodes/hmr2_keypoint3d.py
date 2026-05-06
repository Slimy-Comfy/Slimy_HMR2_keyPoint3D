"""
Slimy_HMR2_keyPoint3D — HMR2.0 3D pose estimator for ComfyUI (Windows Native)
WSL2不要・Windows embedded Pythonで直接動作
"""

import base64
import io
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage, ImageDraw


# HOME環境変数がない場合に自動設定（Windows対応）
if not os.environ.get("HOME"):
    os.environ["HOME"] = os.environ.get("USERPROFILE", str(Path.home()))


# OpenPose 25ポイント順（smpl_to_openpose マッピング準拠）
JOINT_NAMES = [
    "nose",             # 0
    "neck",             # 1
    "right_shoulder",   # 2
    "right_elbow",      # 3
    "right_wrist",      # 4
    "left_shoulder",    # 5
    "left_elbow",       # 6
    "left_wrist",       # 7
    "pelvis",           # 8
    "right_hip",        # 9
    "right_knee",       # 10
    "right_ankle",      # 11
    "left_hip",         # 12
    "left_knee",        # 13
    "left_ankle",       # 14
    "right_eye",        # 15
    "left_eye",         # 16
    "right_ear",        # 17
    "left_ear",         # 18
    "left_big_toe",     # 19
    "left_small_toe",   # 20
    "left_heel",        # 21
    "right_big_toe",    # 22
    "right_small_toe",  # 23
    "right_heel",       # 24
]


# ── 骨格描画 ──────────────────────────────────────────────────────────────────

_C = {
    'body':  (204, 204, 204),
    'left':  ( 68, 136, 255),
    'right': (255,  68,  68),
}

_LEFT_NAMES  = {"left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_knee", "left_ankle",
                "left_eye", "left_ear", "left_big_toe", "left_small_toe", "left_heel"}
_RIGHT_NAMES = {"right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_knee", "right_ankle",
                "right_eye", "right_ear", "right_big_toe", "right_small_toe", "right_heel"}

_CONNS = [
    ("nose",           "right_eye",      'right'),
    ("nose",           "left_eye",       'left'),
    ("right_eye",      "right_ear",      'right'),
    ("left_eye",       "left_ear",       'left'),
    ("nose",           "neck",           'body'),
    ("neck",           "right_shoulder", 'right'),
    ("neck",           "left_shoulder",  'left'),
    ("right_shoulder", "right_elbow",    'right'),
    ("right_elbow",    "right_wrist",    'right'),
    ("left_shoulder",  "left_elbow",     'left'),
    ("left_elbow",     "left_wrist",     'left'),
    ("neck",           "pelvis",         'body'),
    ("right_shoulder", "right_hip",      'right'),
    ("left_shoulder",  "left_hip",       'left'),
    ("pelvis",         "right_hip",      'right'),
    ("pelvis",         "left_hip",       'left'),
    ("right_hip",      "right_knee",     'right'),
    ("right_knee",     "right_ankle",    'right'),
    ("left_hip",       "left_knee",      'left'),
    ("left_knee",      "left_ankle",     'left'),
]


def _get_pts(person: dict, W: int, H: int) -> dict:
    pts = {}
    kp2d = person.get("keypoints_2d_norm", {})
    for name, (nx, ny) in kp2d.items():
        pts[name] = (nx * W, ny * H)
    return pts


def _draw_skeleton(canvas: PILImage.Image, person: dict, W: int, H: int):
    draw = ImageDraw.Draw(canvas)
    lw   = max(2, min(W, H) // 200)
    jr   = max(3, min(W, H) // 150)
    pts  = _get_pts(person, W, H)

    for a, b, ck in _CONNS:
        if a in pts and b in pts:
            ax, ay = pts[a]
            bx, by = pts[b]
            if 0 <= ax < W and 0 <= ay < H and 0 <= bx < W and 0 <= by < H:
                draw.line([pts[a], pts[b]], fill=_C[ck], width=lw)

    for name, pt in pts.items():
        x, y = pt
        if not (0 <= x < W and 0 <= y < H):
            continue
        color = _C['left'] if name in _LEFT_NAMES else \
                _C['right'] if name in _RIGHT_NAMES else _C['body']
        draw.ellipse([x - jr, y - jr, x + jr, y + jr], fill=color)


def _to_tensor(img: PILImage.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _run_inference(img_np: np.ndarray, max_people: int) -> dict:
    """WindowsネイティブでHMR2推論を実行"""
    import sys
    import cv2
    from ultralytics import YOLO

    # ノードローカルのhmr2を優先してsys.pathに追加
    _node_dir = str(Path(__file__).parent.parent)
    if _node_dir not in sys.path:
        sys.path.insert(0, _node_dir)

    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT, download_models
    from hmr2.utils import recursive_to
    from hmr2.datasets.vitdet_dataset import ViTDetDataset

    # チェックポイント未DLなら自動ダウンロード
    from pathlib import Path as _Path
    if not _Path(DEFAULT_CHECKPOINT).exists():
        download_models()

    H, W = img_np.shape[:2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルロード（キャッシュ済みなら高速）
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device)
    model.eval()

    # BGR変換（ComfyUIはRGB）
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # YOLO人物検出
    yolo = YOLO("yolov8n.pt")
    results = yolo(img_bgr, classes=[0], verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append([x1, y1, x2, y2])
    boxes = boxes[:max_people]

    if not boxes:
        return {"error": "no person detected"}

    boxes_np = np.array(boxes)
    dataset = ViTDetDataset(model_cfg, img_bgr, boxes_np)
    ds_centers = dataset.center
    ds_sizes   = [float(dataset[idx]["box_size"]) for idx in range(len(dataset))]

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    people    = []
    person_id = 0
    for batch in dataloader:
        batch_size_cur = batch["img"].shape[0]
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        pred_joints    = out["pred_keypoints_3d"].cpu().numpy()
        pred_joints_2d = out["pred_keypoints_2d"].cpu().numpy()

        for i in range(batch_size_cur):
            joints3d      = pred_joints[i]
            joints2d_norm = pred_joints_2d[i]

            cx, cy   = float(ds_centers[person_id][0]), float(ds_centers[person_id][1])
            box_size = ds_sizes[person_id]
            crop_left = cx - box_size / 2.0
            crop_top  = cy - box_size / 2.0

            joints2d_norm01 = {}
            for j, name in enumerate(JOINT_NAMES):
                if j >= len(joints2d_norm):
                    break
                nx, ny = float(joints2d_norm[j][0]), float(joints2d_norm[j][1])
                px = crop_left + box_size * (nx + 0.5)
                py = crop_top  + box_size * (ny + 0.5)
                joints2d_norm01[name] = [round(px / W, 6), round(py / H, 6)]

            kp3d = {JOINT_NAMES[j]: [float(v) for v in joints3d[j]]
                    for j in range(min(25, len(joints3d)))}

            bbox = boxes[person_id] if person_id < len(boxes) else None
            bbox_norm = [bbox[0]/W, bbox[1]/H, bbox[2]/W, bbox[3]/H] if bbox else None

            people.append({
                "person_id":         person_id,
                "bbox_norm":         bbox_norm,
                "keypoints_3d":      kp3d,
                "keypoints_2d_norm": joints2d_norm01,
            })
            person_id += 1

    return {
        "version":           "hmr2_3d_v1",
        "model":             "hmr2_native",
        "source_image_size": [W, H],
        "coordinate_space":  "smpl_root_relative",
        "joint_names":       JOINT_NAMES,
        "people":            people,
    }


class VNCCS_HMR2KeyPoint3D:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":           ("IMAGE",),
                "person_index":    ("STRING", {"default": "0"}),
                "output_filename": ("STRING", {"default": "hmr2_keypoint3d"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES  = ("pose_image", "skeleton_only", "pose_json")
    FUNCTION      = "estimate"
    CATEGORY      = "Slimy/Pose"
    OUTPUT_NODE   = True

    def estimate(self, image, person_index, output_filename):

        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        H, W   = img_np.shape[:2]

        indices = [int(i.strip()) for i in person_index.split(",") if i.strip().isdigit()]
        if all(i == 0 for i in indices):
            max_people = 10
        else:
            max_people = max(i for i in indices if i > 0)

        try:
            import folder_paths
            out_dir = Path(folder_paths.get_output_directory()) / "pose3d"
        except Exception:
            out_dir = Path(tempfile.gettempdir()) / "pose3d"

        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            data = _run_inference(img_np, max_people)
        except Exception as e:
            return self._err(image, f"推論エラー: {e}")

        if "error" in data:
            return self._err(image, data["error"])

        pil_orig     = PILImage.fromarray(img_np).convert("RGB")
        pil_skeleton = PILImage.new("RGB", (W, H), (0, 0, 0))

        people = data.get("people", [])

        if not all(i == 0 for i in indices):
            selected = [people[i - 1] for i in indices if i > 0 and i <= len(people)]
            people = selected

        for person in people:
            _draw_skeleton(pil_orig,     person, W, H)
            _draw_skeleton(pil_skeleton, person, W, H)

        out_data = {**data, "people": people}
        json_str = json.dumps(out_data, ensure_ascii=False, indent=2)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path  = out_dir / f"{output_filename}_{ts}.json"
        out_path.write_text(json_str, encoding="utf-8")

        n = len(people)
        print(f"[Slimy_HMR2_keyPoint3D] {n} 人検出。保存先 → {out_path}")

        # サムネイル生成（長辺256px、アスペクト比維持）
        thumb = pil_orig.copy()
        thumb.thumbnail((256, 256), PILImage.LANCZOS)
        buf = io.BytesIO()
        thumb.convert("RGB").save(buf, format="JPEG", quality=85)
        thumb_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "ui":     {"text": [json_str], "thumbnail_b64": [thumb_b64]},
            "result": (_to_tensor(pil_orig), _to_tensor(pil_skeleton), json_str),
        }

    @staticmethod
    def _err(image, msg: str):
        print(f"[Slimy_HMR2_keyPoint3D] ERROR: {msg}")
        err_json = json.dumps({"error": msg, "people": []}, ensure_ascii=False)
        B, H, W, C = image.shape
        black = torch.zeros(1, H, W, C)
        return {"ui": {"text": [err_json]}, "result": (image, black, err_json)}


NODE_CLASS_MAPPINGS = {
    "VNCCS_HMR2KeyPoint3D": VNCCS_HMR2KeyPoint3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_HMR2KeyPoint3D": "Slimy_HMR2_keyPoint3D",
}
