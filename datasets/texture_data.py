import os

import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

from .modules import Organ, SingleImage
from .aug import base_transforms, random_augmentations


def build_dataloaders(train_args, config):
    T0 = base_transforms(config.im_res)
    T1 = random_augmentations(fill_val=-1.)

    train_ds, val_ds = {}, {}
    #organs = [o for o in os.listdir(config.src_dir)]
    organs = [
    o for o in os.listdir(config.src_dir)
    if os.path.isdir(os.path.join(config.src_dir, o)) and not o.startswith(".")
]
    for class_label, us_organ in enumerate(organs):
        organ_data = Organ(config.src_dir, us_organ)
        train_ds[class_label] = organ_data.train
        val_ds[us_organ] = organ_data.val

    train_ds = UltrasoundTexture(train_ds, T0, T1, config.epoch_size)
    train_dl = DataLoader(
        train_ds,
        batch_size=train_args.batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=False, drop_last=True
    )
    return train_dl, ValidationDataset(val_ds, T0)


class UltrasoundTexture(Dataset):
    def __init__(self, images_by_organs: dict[int, list[SingleImage]], T0: v2.Transform, augmentations: tuple[v2.Transform], epoch_size: int):
        self.T0 = T0
        self.color_aug, self.spatial_aug = augmentations

        self.n_classes = len(images_by_organs.keys())
        self.n_ims_per_class = {k: len(v) for k, v in images_by_organs.items()}

        self.images = images_by_organs
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, _) -> dict[str, torch.Tensor]:
        class_label = torch.randint(self.n_classes, (1,)).item()
        im_num = torch.randint(self.n_ims_per_class[class_label], (1,)).item()
        image_container: SingleImage = self.images[class_label][im_num]
        im = image_container.load(self.T0)
        im1 = self.spatial_aug(im)
        im2 = self.spatial_aug(im)

        return {
            'x1': im1, 'x1_aug1': self.color_aug(im1), 'x1_aug2': self.color_aug(im1),
            'x2': im2, 'x2_aug1': self.color_aug(im2), 'x2_aug2': self.color_aug(im2)
        }


class ValidationDataset:
    def __init__(self, images_by_organs: dict[str, SingleImage], T0: v2.Transform) -> None:
        self.ims = []
        self.dataset_names = list(images_by_organs.keys())
        for ds, val_im in images_by_organs.items():
            im = val_im.load(T0)
            self.ims.append(im)
        self.stacked = torch.vstack(self.ims).unsqueeze(1)
