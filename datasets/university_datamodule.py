import os
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import transforms

from .university_dataset import UniversityDataset, UniversitySampler
from .utils import RandomErasing, ImageNetPolicy


class UniversityDataModule(pl.LightningDataModule):
    name = "university"

    def __init__(
            self,
            # common
            data_dir: str,
            image_size: int = 224,
            thumbnail_scale: int = 2,
            batch_size: int = 4,
            num_workers: int = 4,
            pin_memory: bool = False,
            drop_last: bool = True,
            # characteristic
            sample_num: int = 1,
            pad: int = 0,
            erasing_p: float = 0,
            color_jitter: bool = False,
            DA: bool = False,
            names: Tuple[str] = ('satellite', 'street', 'drone'),
            retrieval: str = 'drone',  # optional 'street'
    ):
        super(UniversityDataModule, self).__init__()
        self.class_names = None
        self.university_test = None
        self.university_val = None
        self.university_train = None
        self.save_hyperparameters()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # self.mean = [0., 0., 0.]
        # self.std = [1., 1., 1.]

    def prepare_data(self) -> None:
        super(UniversityDataModule, self).prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        if stage in (None, 'fit'):
            self.university_train = UniversityDataset(
                os.path.join(hp.data_dir, 'train'),
                drone_street_thumbnail_num=self.hparams.thumbnail_scale ** 2,
                transform_drone_street_thumbnail=self.__get_default_transform_drone_street_thumbnail(),
                transform_drone_street=self.__get_default_transform_drone_street_train(),
                transform_satellite=self.__get_default_transform_satellite(),
                names=hp.names,
            )
            self.class_names = self.university_train.cls_names

        if stage in (None, 'fit', 'validate'):
            self.university_val = UniversityDataset(
                os.path.join(hp.data_dir, 'test'),
                drone_street_thumbnail_num=self.hparams.thumbnail_scale ** 2,
                transform_drone_street_thumbnail=self.__get_default_transform_drone_street_thumbnail(),
                transform_drone_street=self.__get_default_transform_drone_street_test(),  # TODO
                transform_satellite=self.__get_default_transform_satellite(),
                names=hp.names,
            )

        if stage in (None, 'test'):
            self.university_test = UniversityDataset(
                os.path.join(hp.data_dir, 'test'),
                drone_street_thumbnail_num=self.hparams.thumbnail_scale ** 2,
                transform_drone_street_thumbnail=self.__get_default_transform_drone_street_thumbnail(),
                transform_drone_street=self.__get_default_transform_drone_street_test(),
                transform_satellite=self.__get_default_transform_satellite(),
                names=hp.names,
            )

    def train_collate_fn(self, batch):
        """
        # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
        """
        hp = self.hparams
        img_sa, img_st, thumb_st, img_dr, thumb_dr, ids = zip(*batch)
        img_sa = torch.stack(img_sa)
        img_st = torch.stack(img_st)
        img_dr = torch.stack(img_dr)
        ids = torch.tensor(ids, dtype=torch.int64)
        if self.hparams.retrieval == 'None':
            return [img_sa, ids], [img_st, ids], [img_dr, ids]

        elif self.hparams.retrieval == 'street':
            return [img_sa, ids], \
                   [img_st, ids, [torch.stack([_[i] for _ in thumb_st]) for i in range(hp.thumbnail_scale ** 2)]]

        elif self.hparams.retrieval == 'drone':
            return [img_sa, ids], \
                   [img_dr, ids, [torch.stack([_[i] for _ in thumb_dr]) for i in range(hp.thumbnail_scale ** 2)]]

        elif self.hparams.retrieval == 'both':
            return [img_sa, ids], \
                   [img_st, ids, [torch.stack([_[i] for _ in thumb_st]) for i in range(hp.thumbnail_scale ** 2)]], \
                   [img_dr, ids, [torch.stack([_[i] for _ in thumb_dr]) for i in range(hp.thumbnail_scale ** 2)]]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        hp = self.hparams
        dataset = self.university_train
        dataloader = DataLoader(
            dataset,
            batch_size=hp.batch_size,
            sampler=UniversitySampler(
                dataset,
                hp.batch_size,
                hp.sample_num,
            ),
            num_workers=hp.num_workers,
            pin_memory=hp.pin_memory,
            drop_last=hp.drop_last,
            collate_fn=self.train_collate_fn,
        )
        return dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        hp = self.hparams
        dataset = self.university_val
        dataloader = DataLoader(
            dataset,
            batch_size=hp.batch_size,
            sampler=UniversitySampler(
                dataset,
                hp.batch_size,
                hp.sample_num,
            ),
            num_workers=hp.num_workers,
            pin_memory=hp.pin_memory,
            drop_last=hp.drop_last,
            collate_fn=self.train_collate_fn,
        )
        return dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        hp = self.hparams
        dataset = self.university_test
        dataloader = DataLoader(
            dataset,
            batch_size=hp.batch_size,
            sampler=UniversitySampler(
                dataset,
                hp.batch_size,
                hp.sample_num,
            ),
            num_workers=hp.num_workers,
            pin_memory=hp.pin_memory,
            drop_last=hp.drop_last,
            # TODO
        )
        return dataloader

    def __get_default_transform_drone_street_thumbnail(self):
        hp = self.hparams
        transform_val_list = [
            transforms.Resize(
                size=(hp.image_size // hp.thumbnail_scale, hp.image_size // hp.thumbnail_scale),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize(),
        ]
        return transforms.Compose(transform_val_list)

    def __get_default_transform_drone_street_train(self):
        hp = self.hparams
        transform_train_list = [
            transforms.Resize((hp.image_size, hp.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Pad(hp.pad, padding_mode='edge'),
            transforms.RandomCrop((hp.image_size, hp.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize(),
        ]
        if hp.erasing_p > 0:
            transform_train_list.append(RandomErasing(probability=hp.erasing_p, mean=[0.0, 0.0, 0.0]))
        if hp.color_jitter:
            transform_train_list.reverse()
            transform_train_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))
            transform_train_list.reverse()
        if hp.DA:
            transform_train_list.reverse()
            transform_train_list.append(ImageNetPolicy)
            transform_train_list.reverse()

        return transforms.Compose(transform_train_list)

    def __get_default_transform_satellite(self):
        hp = self.hparams
        transform_satellite_list = [
            transforms.Resize((hp.image_size, hp.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Pad(hp.pad, padding_mode='edge'),
            transforms.RandomAffine(90),
            transforms.RandomCrop((hp.image_size, hp.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize(),
        ]
        if hp.color_jitter:
            transform_satellite_list.reverse()
            transform_satellite_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))
            transform_satellite_list.reverse()

        return transforms.Compose(transform_satellite_list)

    def __get_default_transform_drone_street_test(self):
        hp = self.hparams
        transform_val_list = [
            transforms.Resize(size=(hp.image_size, hp.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize(),
        ]
        return transforms.Compose(transform_val_list)

    def normalize(self):
        return transforms.Normalize(self.mean, self.std)

    def denormalize(self):
        return transforms.Normalize(
            mean=[-m / s for m, s in zip(self.mean, self.std)],
            std=[1 / s for s in self.std]
        )

