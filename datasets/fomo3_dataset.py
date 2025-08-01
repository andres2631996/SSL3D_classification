import torch
import blosc2
from batchgenerators.utilities.file_and_folder_operations import load_json
import os, sys
from torch.utils.data import Dataset
import numpy as np
from .base_datamodule import BaseDataModule


class fomo3Dataset(Dataset):
    def __init__(
        self,
        ids: list,
        data_path: os.PathLike,
        label_file: os.PathLike,
        split: str = "train",
        transform: list = None,
    ):
        self.ids = ids
        self.labels = load_json(label_file)
        self.split = split
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        t1_path = os.path.join(self.data_path, f"{self.ids[idx]}_0000.b2nd")
        t2_path = os.path.join(self.data_path, f"{self.ids[idx]}_0001.b2nd")
        t1_img, t2_img = blosc2.open(t1_path)[:], blosc2.open(t2_path)[:]
        # Concatenate both images
        img = np.concatenate([t1_img, t2_img], axis=0)
        # Apply transforms, if any
        if self.transform is not None:
            img = self.transform(**{"image": img})

        # Return label
        label = self.labels[self.ids[idx]]
        return img, label
