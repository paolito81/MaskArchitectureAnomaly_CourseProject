# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms


#This constructor: creates a Cityscapes dataset configuration,, stores training settings, builds the augmentation pipeline.
class CityscapesSemantic(LightningDataModule): #class that inherits from the LightningDataModule base class, which provides common functionality for loading and preprocessing data for training and evaluation.
    def __init__(
        self,
        path,
        #the workers are responsible for loading and preprocessing data in parallel, which can speed up the training process by utilizing multiple CPU cores.
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (1024, 1024),
        num_classes: int = 19,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True,
    ) -> None: #Contructors always return None.
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        #data augmentation transforms data for training and evaluation:
        #color jitter
        #random flip
        #scale gitter
        #crop
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    #the parser converts the target segmentation map into a list of binary masks and corresponding class labels for each instance in the image. It iterates over the unique label IDs in the target segmentation map, creates a binary mask for each class, and collects the corresponding class labels while ignoring classes marked as "ignore_in_eval".
    #we remap the original label IDs to the training IDs defined in the Cityscapes dataset, which are used for training and evaluation. The function returns three lists: masks (binary masks for each class), labels (corresponding class labels), and a list of False values indicating that none of the instances should be ignored during evaluation.
    @staticmethod
    def target_parser(target, **kwargs):
        masks, labels = [], [] #two lists

        #Cityscape give us: One image where each pixel stores a class ID 

        #loop over unique classes
        for label_id in target[0].unique():

            #we are finding the cytiscapes class corresponding to the current label ID by iterating through the predefined classes in the Cityscapes dataset and checking if the class ID matches the current label ID.
            cls = next((cls for cls in Cityscapes.classes if cls.id == label_id), None)

            #some classes in the Cityscapes dataset are marked as "ignore_in_eval", meaning that they should not be considered during evaluation. If the class corresponding to the current label ID is either not found (cls is None) or marked as "ignore_in_eval", we skip processing for that label ID and continue to the next one. 
            if cls is None or cls.ignore_in_eval:
                continue

            #creates a binary mask for the current label ID by comparing the target segmentation map with the label ID. The resulting mask is a binary tensor where pixels belonging to the current class are marked as True (or 1), and all other pixels are marked as False (or 0). This binary mask is then appended to the list of masks, and the corresponding training ID for the class is appended to the list of labels. The training ID is used for training and evaluation purposes, and it may differ from the original label ID defined in the Cityscapes dataset.
            #creation of the Binary Boolean mask
            masks.append(target[0] == label_id)
            labels.append(cls.train_id)

        return masks, labels, [False for _ in range(len(masks))]

    #This function creates: Cityscapes train dataset and Cityscapes validation dataset
    #how Cityscapes files are organized, how to parse them, where they are located
    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        cityscapes_dataset_kwargs = {
            "img_suffix": ".png",
            "target_suffix": ".png",
            "img_stem_suffix": "leftImg8bit",
            "target_stem_suffix": "gtFine_labelIds",
            "zip_path": Path(self.path, "leftImg8bit_trainvaltest.zip"),
            "target_zip_path": Path(self.path, "gtFine_trainvaltest.zip"),
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }

        self.cityscapes_train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path("./leftImg8bit/train"),
            target_folder_path_in_zip=Path("./gtFine/train"),
            **cityscapes_dataset_kwargs,
        )

        self.cityscapes_val_dataset = Dataset(
            img_folder_path_in_zip=Path("./leftImg8bit/val"),
            target_folder_path_in_zip=Path("./gtFine/val"),
            **cityscapes_dataset_kwargs,
        )

        return self


    #the dataloader loads samples like: 
    def train_dataloader(self):
        return DataLoader(
            self.cityscapes_train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cityscapes_val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

#ZIP files
#    ↓
#Dataset.__getitem__(0)
#    ↓
#(img0, target0)

#Dataset.__getitem__(1)
#    ↓
#(img1, target1)

#Dataset.__getitem__(2)
#    ↓
#(img2, target2)

#DataLoader collects them
#    ↓
#collate_fn combines them
#    ↓
#Batch sent to GPU