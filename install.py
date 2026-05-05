"""
Slimy_HMR2_keyPoint3D — install.py
SMPLモデルの配置確認のみ行います。
パッケージは requirements.txt で管理しています。
"""

import os
from pathlib import Path


def log(msg):
    print(f"[Slimy_HMR2_keyPoint3D] {msg}")


def get_smpl_path() -> Path:
    """ComfyUIのmodels/hmr2/data/smpl/を優先し、なければフォールバック"""
    # folder_pathsが使える場合はComfyUIのmodelsディレクトリを使用
    try:
        import folder_paths
        return Path(folder_paths.models_dir) / "hmr2" / "data" / "smpl" / "SMPL_NEUTRAL.pkl"
    except Exception:
        pass
    # フォールバック: ノード内のdataフォルダ
    return Path(__file__).parent / "data" / "smpl" / "SMPL_NEUTRAL.pkl"


def check_smpl():
    smpl_path = get_smpl_path()

    if smpl_path.exists():
        log(f"SMPLモデル確認済み: {smpl_path}")
        return True

    log("=" * 60)
    log("SMPLモデルが見つかりません。")
    log("以下からダウンロードし、配置してください：")
    log("  URL: https://smpl.is.tue.mpg.de/ （要登録）")
    log(f"  配置先: {smpl_path}")
    log("ファイル名は SMPL_NEUTRAL.pkl にしてください。")
    log("=" * 60)
    return False


if not os.environ.get("HOME"):
    os.environ["HOME"] = os.environ.get("USERPROFILE", str(Path.home()))

check_smpl()
