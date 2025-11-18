import torch
import blosc2
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
import os, sys
from torch.utils.data import Dataset
import numpy as np
from .base_datamodule import BaseDataModule

import matplotlib.pyplot as plt


class aneurysmDataset(Dataset):
    def __init__(
        self,
        data_path: os.PathLike,
        ids: list,
        split: str = "train",
        transform: list = None,
    ):
        self.split = split
        self.data_path = data_path
        self.transform = transform

        label_file = os.path.join(data_path, "labels.json")
        self.labels = load_json(label_file)
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, f"{self.ids[idx]}_0000.b2nd")

        img = blosc2.open(img_path)[:]

        img = torch.tensor(img)

        # Return label
        label = self.labels[self.ids[idx]]

        # Apply transforms, if any
        if self.transform is not None:
            img = self.transform(**{"image": img})["image"]

        return img, label


class AneurysmDataModule(BaseDataModule):
    def __init__(self, **params):
        super(AneurysmDataModule, self).__init__(**params)
        self.params = params

    def split(self):
        split_file = os.path.join(os.path.dirname(self.data_path), "cv_splits.json")
        if not (os.path.exists(split_file)):
            # Create split
            files = sorted(os.listdir(self.data_path))
            ids = np.array(
                [f.replace("_0000.b2nd", "") for f in files if "_0000.b2nd" in f],
                dtype=str,
            )
            s = np.array_split(ids, 5)  # Random split
            splits = []
            for i in range(len(s)):
                val_ids = s[i].tolist()
                train_ids = np.setdiff1d(ids, val_ids).tolist()
                splits.append({"train": train_ids, "val": val_ids})

            save_json(splits, split_file)
            return splits

        splits = load_json(split_file)
        return splits

    def setup(self, stage: str):
        splits = self.split()
        if isinstance(self.params["fold"], str):
            # Run fold_all: train and validate with everything
            train_ids = splits[0]["train"] + splits[0]["val"]
            val_ids = splits[0]["train"] + splits[0]["val"]
        else:
            # Classical cross-validation
            split = splits[self.params["fold"]]
            train_ids, val_ids = split["train"], split["val"]

        self.train_dataset = aneurysmDataset(
            data_path=self.data_path,
            ids=train_ids,
            split="train",
            transform=self.train_transforms,
        )
        self.val_dataset = aneurysmDataset(
            data_path=self.data_path,
            ids=val_ids,
            split="val",
            transform=self.test_transforms,
        )
