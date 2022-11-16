from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
# from torchvision.datasets import MNIST
from src.datamodules import chexpert
from torchvision.transforms import transforms
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import cv2


class CheXpertDataModule(LightningDataModule):
    """LightningDataModule for CheXpertDataModule dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        # train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        image_root_path: str = "CheXpert-v1.0-pad224/",
        image_size: int = 224,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms_train =albu.Compose([
        albu.Resize(image_size, image_size),
        albu.OneOf([
            albu.RandomBrightness(limit=.2, p=1), 
            albu.RandomContrast(limit=.2, p=1), 
            albu.RandomGamma(p=1)
        ], p=.3),
        albu.OneOf([
            albu.Blur(blur_limit=3, p=1),
            albu.MedianBlur(blur_limit=3, p=1)
        ], p=.2),
        albu.OneOf([
            albu.GaussNoise(0.002, p=.5),
        ], p=.2),
#         albu.RandomRotate90(p=.5),
        albu.HorizontalFlip(p=.5),
#         albu.VerticalFlip(p=.5),
#         albu.Cutout(num_holes=10, 
#                     max_h_size=int(.1 * size), max_w_size=int(.1 * size), 
#                     p=.25),
        albu.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.01, rotate_limit=10, p=0.4, border_mode = cv2.BORDER_CONSTANT),
        albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ])
        self.transforms_valid = albu.Compose([
        albu.Resize(image_size, image_size),
        albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 5

    # def prepare_data(self):
    #     """Download data if needed.

    #     Do not use it to assign state (self.x = y).
    #     """
    #     chexpert.CheXpertDataset(self.hparams.data_dir, train=True, download=True)
    #     chexpert.CheXpertDataset(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = chexpert.CheXpertDataset(self.hparams.data_dir, self.hparams.image_root_path, transforms=self.transforms_train, mode='train')
            self.data_val = chexpert.CheXpertDataset(self.hparams.data_dir, self.hparams.image_root_path, transforms=self.transforms_valid, mode='valid')
            self.data_test = chexpert.CheXpertDataset(self.hparams.data_dir, self.hparams.image_root_path, transforms=self.transforms_valid, mode='test')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass




if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "chexpert.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
