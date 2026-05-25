"""
Slimy_HMR2_keyPoint3D — HMR2.0 3D KeyPoint estimator for ComfyUI (Windows Native)
"""

import os
import inspect
from pathlib import Path

# HOME環境変数がない場合に自動設定（Windows対応）
if not os.environ.get("HOME"):
    os.environ["HOME"] = os.environ.get("USERPROFILE", str(Path.home()))

# Python 3.13 + NumPy 2.x 対応: chumpy のインポート前にパッチを当てる
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

try:
    import numpy as np
    if not hasattr(np, 'int'):
        np.bool = np.bool_
        np.int = np.int_
        np.float = np.float64
        np.complex = np.complex128
        np.object = object
        np.str = np.str_
        np.unicode = np.str_
except Exception:
    pass

# HMR2モデルのキャッシュディレクトリをComfyUIのmodels/hmr2に向ける
# hmr2/configs/__init__.py のインポート前に設定する必要がある
if not os.environ.get("HMR2_CACHE_DIR"):
    try:
        import folder_paths
        _models_dir = Path(folder_paths.models_dir) / "hmr2"
    except Exception:
        # folder_pathsが使えない場合はノード内のdataフォルダにフォールバック
        _models_dir = Path(__file__).parent / "data"
    os.environ["HMR2_CACHE_DIR"] = str(_models_dir)

from .nodes.hmr2_keypoint3d import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
