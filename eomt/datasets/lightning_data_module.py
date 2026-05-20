# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Optional
import torch
import lightning


#this module serves as a base class for all dataset-specific LightningDataModules, providing common functionality and structure for loading and preprocessing data 
# for training and evaluation. 
# It defines the initialization of the data module, including setting up paths, batch size, number of workers, image size, number of classes, and other relevant parameters. 
# Additionally, it includes static methods for collating training and evaluation batches, which can be customized by subclasses to handle specific data formats and requirements of different datasets.

class LightningDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        path,
        batch_size: int,
        num_workers: int,
        img_size: tuple[int, int],
        num_classes: int,
        check_empty_targets: bool,
        ignore_idx: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()

        self.path = path
        self.check_empty_targets = check_empty_targets
        self.ignore_idx = ignore_idx
        self.img_size = img_size
        self.num_classes = num_classes

        self.dataloader_kwargs = {
            "persistent_workers": False if num_workers == 0 else persistent_workers,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "batch_size": batch_size,
        }

    @staticmethod
    def train_collate(batch):
        imgs, targets = [], []

        for img, target in batch:
            imgs.append(img)
            targets.append(target)

        return torch.stack(imgs), targets

    @staticmethod
    def eval_collate(batch):
        return tuple(zip(*batch))
