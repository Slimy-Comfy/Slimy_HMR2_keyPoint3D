from .smpl_wrapper import SMPL
from .hmr2 import HMR2
from .discriminator import Discriminator

from ..utils.download import cache_url
from ..configs import CACHE_DIR_4DHUMANS


def download_models(folder=CACHE_DIR_4DHUMANS):
    """Download checkpoints and files for running inference.
    """
    import os
    import shutil
    from pathlib import Path
    os.makedirs(folder, exist_ok=True)
    download_files = {
        "hmr2_data.tar.gz"      : ["https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz", folder],
    }

    for file_name, url in download_files.items():
        output_path = os.path.join(url[1], file_name)
        if not os.path.exists(output_path):
            print("Downloading file: " + file_name)
            output = cache_url(url[0], output_path)
            assert os.path.exists(output_path), f"{output} does not exist"

            if file_name.endswith(".tar.gz"):
                print("Extracting file: " + file_name)
                os.system("tar -xvf " + output_path + " -C " + url[1])
                os.remove(output_path)
                print("Deleted: " + output_path)

    # smpl_mean_params.npz がfolder直下に展開された場合、data/へ移動する
    _src = Path(folder) / "smpl_mean_params.npz"
    _dst = Path(folder) / "data" / "smpl_mean_params.npz"
    if _src.exists() and not _dst.exists():
        _dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(_src), str(_dst))
        print(f"Moved smpl_mean_params.npz -> {_dst}")

def check_smpl_exists():
    import os
    candidates = [
        f'{CACHE_DIR_4DHUMANS}/data/smpl/SMPL_NEUTRAL.pkl',
        f'data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
    ]
    candidates_exist = [os.path.exists(c) for c in candidates]
    if not any(candidates_exist):
        raise FileNotFoundError(f"SMPL model not found. Please download it from https://smplify.is.tue.mpg.de/ and place it at {candidates[1]}")

    if (not candidates_exist[0]) and candidates_exist[1]:
        convert_pkl(candidates[1], candidates[0])

    return True

def convert_pkl(old_pkl, new_pkl):
    """
    Convert a Python 2 pickle to Python 3
    """
    import dill
    import pickle

    dill._dill._reverse_typemap["ObjectType"] = object

    with open(old_pkl, "rb") as f:
        loaded = pickle.load(f, encoding="latin1")

    with open(new_pkl, "wb") as outfile:
        pickle.dump(loaded, outfile)

DEFAULT_CHECKPOINT=f'{CACHE_DIR_4DHUMANS}/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
def load_hmr2(checkpoint_path=DEFAULT_CHECKPOINT):
    from pathlib import Path
    from ..configs import get_config
    # model_config.yaml はノード内に同梱済みのものを使用
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    # Ensure SMPL model exists
    check_smpl_exists()

    model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, weights_only=False)
    return model, model_cfg
