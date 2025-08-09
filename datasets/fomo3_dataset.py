import torch
import blosc2
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
import os, sys
from torch.utils.data import Dataset
import numpy as np
from .base_datamodule import BaseDataModule

import matplotlib.pyplot as plt


class fomo3Dataset(Dataset):
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
        t1_path = os.path.join(self.data_path, f"{self.ids[idx]}_0000.b2nd")
        t2_path = os.path.join(self.data_path, f"{self.ids[idx]}_0001.b2nd")
        t1_img = blosc2.open(t1_path)[:]
        # t2_img = blosc2.open(t2_path)[:]

        # Concatenate both images
        # img = torch.tensor(np.concatenate([t1_img, t2_img], axis=0), dtype=torch.float)
        img = torch.tensor(t1_img)

        # Return label
        label = self.labels[self.ids[idx]]

        # Apply transforms, if any
        if self.transform is not None:
            img = self.transform(**{"image": img})["image"]
            """
            plt.figure()
            plt.subplot(321)
            plt.imshow(img[0, img.shape[1] // 2].cpu().detach().numpy(), cmap="gray")
            plt.colorbar()
            plt.subplot(322)
            plt.imshow(img[1, img.shape[1] // 2].cpu().detach().numpy(), cmap="gray")
            plt.colorbar()
            plt.subplot(323)
            plt.imshow(img[0, :, img.shape[2] // 2].cpu().detach().numpy(), cmap="gray")
            plt.colorbar()
            plt.subplot(324)
            plt.imshow(img[1, :, img.shape[2] // 2].cpu().detach().numpy(), cmap="gray")
            plt.colorbar()
            plt.subplot(325)
            plt.imshow(
                img[0, :, :, img.shape[3] // 2].cpu().detach().numpy(), cmap="gray"
            )
            plt.colorbar()
            plt.subplot(326)
            plt.imshow(
                img[1, :, :, img.shape[3] // 2].cpu().detach().numpy(), cmap="gray"
            )
            plt.colorbar()
            plt.suptitle(label)
            plt.show()
            """

        return img, float(label), self.ids[idx]


class Fomo3DataModule(BaseDataModule):
    def __init__(self, **params):
        super(Fomo3DataModule, self).__init__(**params)
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
        split = splits[self.params["fold"]]
        train_ids, val_ids = split["train"], split["val"]
        self.train_dataset = fomo3Dataset(
            data_path=self.data_path,
            ids=train_ids,
            split="train",
            transform=self.train_transforms,
        )
        self.val_dataset = fomo3Dataset(
            data_path=self.data_path,
            ids=val_ids,
            split="val",
            transform=self.test_transforms,
        )
