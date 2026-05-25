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


# ── OpenPose Body25 公式カラー（poseParametersRender.hpp準拠） ──────────────
# インデックスはJOINT_NAMESの順番に対応
_JOINT_COLORS = [
    (255,   0,  85),   # 0  nose
    (255,   0,   0),   # 1  neck
    (255,  85,   0),   # 2  right_shoulder
    (255, 170,   0),   # 3  right_elbow
    (255, 255,   0),   # 4  right_wrist
    (170, 255,   0),   # 5  left_shoulder
    ( 85, 255,   0),   # 6  left_elbow
    (  0, 255,   0),   # 7  left_wrist
    (255,   0,   0),   # 8  pelvis
    (  0, 255,  85),   # 9  right_hip
    (  0, 255, 170),   # 10 right_knee
    (  0, 255, 255),   # 11 right_ankle
    (  0, 170, 255),   # 12 left_hip
    (  0,  85, 255),   # 13 left_knee
    (  0,   0, 255),   # 14 left_ankle
    (255,   0, 170),   # 15 right_eye
    (170,   0, 255),   # 16 left_eye
    (255,   0, 255),   # 17 right_ear
    ( 85,   0, 255),   # 18 left_ear
    (  0,   0, 255),   # 19 left_big_toe
    (  0,   0, 255),   # 20 left_small_toe
    (  0,   0, 255),   # 21 left_heel
    (  0, 255, 255),   # 22 right_big_toe
    (  0, 255, 255),   # 23 right_small_toe
    (  0, 255, 255),   # 24 right_heel
]

_JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}

# 接続ペア（Body25準拠）: (joint_a, joint_b)
_CONNS = [
    ("pelvis",         "neck"),
    ("neck",           "right_shoulder"),
    ("neck",           "left_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("pelvis",         "right_hip"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("pelvis",         "left_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("neck",           "nose"),
    ("nose",           "right_eye"),
    ("right_eye",      "right_ear"),
    ("nose",           "left_eye"),
    ("left_eye",       "left_ear"),
    ("left_ankle",     "left_big_toe"),
    ("left_big_toe",   "left_small_toe"),
    ("left_ankle",     "left_heel"),
    ("right_ankle",    "right_big_toe"),
    ("right_big_toe",  "right_small_toe"),
    ("right_ankle",    "right_heel"),
]


def _get_pts(person: dict, W: int, H: int) -> dict:
    pts = {}
    kp2d = person.get("keypoints_2d_norm", {})
    for name, (nx, ny) in kp2d.items():
        pts[name] = (nx * W, ny * H)
    return pts


def _lerp_color(c1, c2, t):
    """2色をt(0.0-1.0)で線形補間"""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _draw_skeleton(canvas: PILImage.Image, person: dict, W: int, H: int):
    draw = ImageDraw.Draw(canvas)
    lw   = max(2, min(W, H) // 150)
    jr   = max(4, min(W, H) // 100)
    pts  = _get_pts(person, W, H)

    # 接続線をグラデーションで描画（両端の関節色を線形補間）
    segments = 10  # グラデーションの分割数
    for a, b in _CONNS:
        if a not in pts or b not in pts:
            continue
        ax, ay = pts[a]
        bx, by = pts[b]
        if not (0 <= ax < W and 0 <= ay < H and 0 <= bx < W and 0 <= by < H):
            continue
        ca = _JOINT_COLORS[_JOINT_INDEX[a]]
        cb = _JOINT_COLORS[_JOINT_INDEX[b]]
        for s in range(segments):
            t0 = s / segments
            t1 = (s + 1) / segments
            x0 = ax + (bx - ax) * t0
            y0 = ay + (by - ay) * t0
            x1 = ax + (bx - ax) * t1
            y1 = ay + (by - ay) * t1
            color = _lerp_color(ca, cb, (t0 + t1) / 2)
            draw.line([(x0, y0), (x1, y1)], fill=color, width=lw)

    # 関節点を描画
    for name, pt in pts.items():
        x, y = pt
        if not (0 <= x < W and 0 <= y < H):
            continue
        idx = _JOINT_INDEX.get(name)
        if idx is None:
            continue
        color = _JOINT_COLORS[idx]
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

    # YOLO人物検出（iou=0.45で重複bbox排除を強化）
    yolo = YOLO("yolov8n.pt")
    results = yolo(img_bgr, classes=[0], verbose=False, iou=0.45)
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

        # SMPLパラメータ取得（body_pose: (batch,23,3,3), global_orient: (batch,1,3,3)）
        pred_smpl   = out["pred_smpl_params"]
        body_poses  = pred_smpl["body_pose"].cpu().numpy()       # (batch, 23, 3, 3)
        global_orients = pred_smpl["global_orient"].cpu().numpy() # (batch,  1, 3, 3)

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

            # body_pose: 23関節の回転行列をリストに変換 [[row0,row1,row2], ...]
            body_pose_list = body_poses[i].tolist()   # (23, 3, 3)
            global_orient_list = global_orients[i].tolist()  # (1, 3, 3)

            people.append({
                "person_id":         person_id,
                "bbox_norm":         bbox_norm,
                "keypoints_3d":      kp3d,
                "keypoints_2d_norm": joints2d_norm01,
                "smpl_params": {
                    "global_orient": global_orient_list,  # (1, 3, 3) 骨盤ワールド回転
                    "body_pose":     body_pose_list,       # (23, 3, 3) 各関節ローカル回転行列
                },
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
                "person_index":    ("STRING", {"default": "1"}),
                "output_filename": ("STRING", {"default": "hmr2_keypoint3d"}),
                "output_format":   (["JSON", "PNG (with JSON)"], {"default": "JSON"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES  = ("pose_image", "skeleton_only", "pose_json")
    FUNCTION      = "estimate"
    CATEGORY      = "Slimy/Pose"
    OUTPUT_NODE   = True

    def estimate(self, image, person_index, output_filename, output_format="JSON"):

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

        if output_format == "JSON":
            out_path = out_dir / f"{output_filename}_{ts}.json"
            out_path.write_text(json_str, encoding="utf-8")
            print(f"[Slimy_HMR2_keyPoint3D] {len(people)} 人検出。JSON保存先 → {out_path}")
            return {
                "ui":     {"text": [json_str], "thumbnail_b64": [], "output_format": ["JSON"]},
                "result": (_to_tensor(pil_orig), _to_tensor(pil_skeleton), json_str),
            }
        else:
            from PIL.PngImagePlugin import PngInfo
            thumb = pil_orig.copy()
            thumb.thumbnail((256, 256), PILImage.LANCZOS)
            meta = PngInfo()
            meta.add_text("hmr2_pose_json", json_str)
            buf = io.BytesIO()
            thumb.convert("RGB").save(buf, format="PNG", pnginfo=meta)
            thumb_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            out_path = out_dir / f"{output_filename}_{ts}.png"
            out_path.write_bytes(base64.b64decode(thumb_b64))
            print(f"[Slimy_HMR2_keyPoint3D] {len(people)} 人検出。PNG保存先 → {out_path}")
            return {
                "ui":     {"text": [json_str], "thumbnail_b64": [thumb_b64], "output_format": ["PNG"]},
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
