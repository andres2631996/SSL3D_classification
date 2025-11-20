import os, sys
import numpy as np
import argparse
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


def load_yaml(path: str):
    """
    Load a YAML file and return its content as a Python object.
    Uses safe_load to avoid executing arbitrary code inside YAML.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


sys.path.insert(
    0, "/media/E132-Projekte/Projects/2025_MartinezMora_SSLBrain/SSL3D_classification"
)
# sys.path.insert(0, "SSL3D_classification")

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    save_json,
)
from datasets.preprocess_3D_data.default_resampling import (
    resample_data_or_seg_to_spacing,
    resample_data_or_seg_to_shape,
)
from datasets.preprocess_3D_data.crop_to_mask import (
    crop_center_with_padding_np,
    get_mask_center,
)
from datasets.preprocess_3D_data.normalization import (
    ZScoreNormalization,
)
from models.resenc import ResEncoder_Classifier


def tta_augmentations(x: torch.Tensor):
    """
    Generate all flips (8) and all 90° rotations (24) in 3D.
    Returns a list of functions to apply to the input tensor.
    """
    augmentations = []

    # All flip combinations (8)
    flip_dims = [
        [],
        [2],
        [3],
        [4],
        [2, 3],
        [2, 4],
        [3, 4],
        [2, 3, 4],
    ]
    for dims in flip_dims:
        augmentations.append(lambda x, d=dims: torch.flip(x, dims=d) if d else x)

    # Rotations: 90° steps in (D,H), (D,W), (H,W)
    planes = [(2, 3), (2, 4), (3, 4)]
    for dim1, dim2 in planes:
        for k in range(4):
            augmentations.append(
                lambda x, d1=dim1, d2=dim2, k=k: torch.rot90(x, k, (d1, d2)) if k else x
            )

    return augmentations


def preprocessing(file: os.PathLike, cfg: dict, device) -> np.ndarray:
    """
    Complete preprocessing

    Params
    ------
    file : input file
    cfg : configuration
    device : computational device

    Returns
    -------
    data : preprocessed file

    """
    # Load preprocessing configuration
    target_spacing = cfg["target_spacing"]
    target_shape = cfg["target_shape"]

    # Derive array and spacing properties of T1
    image = sitk.ReadImage(file)
    img = sitk.GetArrayFromImage(image)[np.newaxis, ...]
    original_spacing = image.GetSpacing()[::-1]

    # Resize image
    data_1mm = resample_data_or_seg_to_spacing(img, original_spacing, target_spacing)

    new_spacing = np.array(
        [i / j * k for i, j, k in zip(target_spacing, target_shape, data_1mm.shape[1:])]
    )
    resized_img = resample_data_or_seg_to_shape(
        data_1mm, target_shape, target_spacing, new_spacing
    )

    # Normalization
    normalizer = ZScoreNormalization()
    data = normalizer.run(resized_img)

    # Extend data one more channel
    data = np.expand_dims(data, 0)

    # Build torch tensor
    data = torch.tensor(data, device=device)

    return data


def model_setup(model):
    # Load model configuration
    cfg_file = join(os.path.dirname(os.path.abspath(__file__)), "model_cfg.json")
    assert os.path.exists(cfg_file), f"Configuration file '{cfg_file}' does not exist"
    cfg = load_json(cfg_file)

    # Initialize model
    model = ResEncoder_Classifier(**cfg)

    return model


def compute_pred(data: np.ndarray, file: os.PathLike, model_cfg: dict, device) -> float:
    """
    Compute prediction from checkpoint files

    Params
    ------
    data : image to predict
    file : checkpoint file
    model_cfg : model configuration
    device : computational device

    Returns
    -------
    pred : output prediction

    """
    # Iterate through files
    preds = []

    # Load a clean version of the model for each checkpoint

    # Initialize model
    model = ResEncoder_Classifier(**model_cfg["model"])

    state_dict = torch.load(file, weights_only=False)["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Compute prediction
    with torch.no_grad():
        # Apply TTA: flips

        aug_fns = tta_augmentations(data)
        aug_preds = []
        for fn in aug_fns:
            aug_data = fn(data)
            pred = model(aug_data).squeeze().detach().cpu().numpy()
            aug_preds.append(pred)

        # Average TTA predictions
        aug_pred = np.vstack(aug_preds)
        pred = np.mean(aug_pred, 0)
        pred = 1 / (1 + np.exp(-pred))

    return pred


def main():
    parser = argparse.ArgumentParser(description="Run inference on aneurysm folder")

    # Input and output paths using modality names from task config
    parser.add_argument(
        "--i", type=str, required=True, help="Input folder (NIfTI format)"
    )
    parser.add_argument("--c", type=str, required=True, help="Configuration file")
    parser.add_argument(
        "--model_cfg", type=str, required=True, help="Model configuration file"
    )
    parser.add_argument("--cpt", type=str, required=True, help="Checkpoint file")
    parser.add_argument(
        "--o", type=str, required=True, help="Output path for prediction"
    )

    # Parse arguments
    args = parser.parse_args()

    infolder = args.i
    cfg_file = args.c
    model_cfg_file = args.model_cfg
    cpt_file = args.cpt
    outfile = args.o

    assert os.path.exists(infolder), f"Input folder '{infolder}' does not exist"
    assert os.path.exists(cfg_file), f"Configuration file '{cfg_file}' does not exist"
    assert os.path.exists(
        model_cfg_file
    ), f"Model configuration file '{model_cfg_file}' does not exist"
    assert os.path.exists(cpt_file), f"Checkpoint file '{cpt_file}' does not exist"
    assert os.path.exists(
        os.path.dirname(outfile)
    ), f"Output parent folder '{os.path.dirname(outfile)}' does not exist"

    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load configuration file
    cfg = load_json(cfg_file)
    model_cfg = load_yaml(model_cfg_file)

    # Set up files
    files = sorted(os.listdir(infolder))
    results = {}
    for file in files:
        if file.endswith("_0000.nii.gz"):
            cid = file.replace("_0000.nii.gz", "")
            # Preprocess files
            data = preprocessing(
                file=os.path.join(infolder, file), cfg=cfg, device=device
            )

            # Prediction
            pred = compute_pred(
                data=data, file=cpt_file, model_cfg=model_cfg, device=device
            )
            results[cid] = float(pred[-1])

            print(cid, results[cid])

    # Write final prediction
    save_json(results, outfile)


if __name__ == "__main__":
    main()
