import torch
import blosc2
from batchgenerators.utilities.file_and_folder_operations import load_json
import os, sys
from torch.utils.data import Dataset
import numpy as np
from .base_datamodule import BaseDataModule

import matplotlib.pyplot as plt


class fomo3Dataset(Dataset):
    def __init__(
        self,
        data_path: os.PathLike,
        split: str = "train",
        transform: list = None,
    ):
        self.split = split
        self.data_path = data_path
        self.transform = transform

        label_file = os.path.join(data_path, "labels.json")
        self.labels = load_json(label_file)
        self.ids = list(self.labels.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        t1_path = os.path.join(self.data_path, f"{self.ids[idx]}_0000.b2nd")
        t2_path = os.path.join(self.data_path, f"{self.ids[idx]}_0001.b2nd")
        t1_img, t2_img = blosc2.open(t1_path)[:], blosc2.open(t2_path)[:]

        # Concatenate both images
        img = torch.tensor(np.concatenate([t1_img, t2_img], axis=0))

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

        return img, label


class Fomo3DataModule(BaseDataModule):
    def __init__(self, **params):
        super(Fomo3DataModule, self).__init__(**params)

    def setup(self, stage: str):
        self.train_dataset = fomo3Dataset(
            data_path=self.data_path,
            split="train",
            transform=self.train_transforms,
        )
        self.val_dataset = fomo3Dataset(
            data_path=self.data_path,
            split="val",
            transform=self.test_transforms,
        )
