import os
import random
from dataclasses import dataclass

import torch
from torchvision.io import read_image
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

from .aug import base_transforms, random_augmentations


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# VAL_DATA = {
#     "abdominal": "abdominalUS/2f14d.jpg",
#     "breast": "BrEaST-Lesions_USG-images_and_masks/0acdc.jpg",
#     "cardiac": "CAMUS/0a90e.jpg",
#     "fetal": "FETAL_PLANES_ZENODO/0a946.jpg",
#     "follicle": "OTU_2d/4aa23.jpg",
#     "knee": "JoCoHS/0a6bc.jpg",
#     "lung": "covid19_ultrasound/ba9e1.jpg",
#     "musculoskeletal": "FALLMUD/7ca47.jpg",
#     "neck": "Thyroid Dataset/a8a49.jpg",
#     "vsx": "mouse_cancer/3f83a.jpg",
# }

VAL_DATA = {
    'abdominal': 'abdominal_US/0d0383532d4b.jpg',
    'breast': 'BrEaST-Lesions_USG-images_and_masks/6b286.jpg',
    'cardiac': 'CAMUS/18467d75c94e.jpg',
    'fetal abdominal': 'FASS/0ceb023e7a9e.png',
    'fetal head': 'HC/0dbb856c4637.png',
    'kidney': 'kideny_US/0ce6173d45e8.jpg',
    'knee': 'JoCoHS/0a4827487819.png',
    'liver': 'US_Liver/62a924184044.jpg',
    'lung': 'covid19_ultrasound/3c573d9a185d.png',
    'musculoskeletal': 'FALLMUD/fb85eb471939.png',
    'neck': 'UPBD/7e179ffe131d.png',
    'ovary': 'PCOSGen/0f74326c724b.png',
    'prostate': 'MicroSeg/f59cad7bcac0.png',
    'thyroid': 'Thyroid Dataset/a44258926b19.png',
}




@dataclass
class SingleImage:
    im_path: str

    def load(self, T0: v2.Transform) -> torch.Tensor:
        return T0(read_image(self.im_path))


class Organ:
    def __init__(self, base_path, organ, val_per_organ=8, seed=42) -> None:
        self.train = []
        self.val = []

        organ_dir = os.path.join(base_path, organ)

        all_images = []
        for ds in os.listdir(organ_dir):
            ds_path = os.path.join(organ_dir, ds)
            if not os.path.isdir(ds_path):
                continue

            for f in os.listdir(ds_path):
                im_path = os.path.join(ds_path, f)

                if os.path.isfile(im_path) and os.path.splitext(f.lower())[1] in IMG_EXTS:
                    all_images.append(im_path)

        all_images = sorted(all_images)

        fixed_val_path = None
        if organ in VAL_DATA:
            candidate = os.path.join(base_path, organ, VAL_DATA[organ])
            if os.path.exists(candidate):
                fixed_val_path = candidate

        rng = random.Random(seed)

        remaining = [p for p in all_images if p != fixed_val_path]
        rng.shuffle(remaining)

        selected_val = []

        if fixed_val_path is not None:
            selected_val.append(fixed_val_path)

        selected_val.extend(remaining[: max(0, val_per_organ - len(selected_val))])

        selected_val_set = set(selected_val)

        self.val = [SingleImage(p) for p in selected_val]
        self.train = [SingleImage(p) for p in all_images if p not in selected_val_set]

        if len(self.train) == 0:
            raise RuntimeError(f"No training images found for organ: {organ}")

        if len(self.val) == 0:
            raise RuntimeError(f"No validation images found for organ: {organ}")


def build_dataloaders(train_args, config):
    T0 = base_transforms(config.im_res)
    T1 = random_augmentations(fill_val=-1.0)

    val_per_organ = int(getattr(config, "val_per_organ", 8))
    seed = int(getattr(config, "seed", 42))

    train_ds, val_ds = {}, {}

    organs = [
        o for o in os.listdir(config.src_dir)
        if os.path.isdir(os.path.join(config.src_dir, o)) and not o.startswith(".")
    ]

    organs = sorted(organs)

    for class_label, us_organ in enumerate(organs):
        organ_data = Organ(
            config.src_dir,
            us_organ,
            val_per_organ=val_per_organ,
            seed=seed + class_label,
        )

        train_ds[class_label] = organ_data.train
        val_ds[us_organ] = organ_data.val

    train_ds = UltrasoundTexture(
        train_ds,
        T0,
        T1,
        config.epoch_size,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=train_args.batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=False,
        drop_last=True,
    )

    return train_dl, ValidationDataset(val_ds, T0)


class UltrasoundTexture(Dataset):
    def __init__(
        self,
        images_by_organs: dict[int, list[SingleImage]],
        T0: v2.Transform,
        augmentations: tuple[v2.Transform],
        epoch_size: int,
    ):
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
            "x1": im1,
            "x1_aug1": self.color_aug(im1),
            "x1_aug2": self.color_aug(im1),

            "x2": im2,
            "x2_aug1": self.color_aug(im2),
            "x2_aug2": self.color_aug(im2),
        }


class ValidationDataset:
    def __init__(
        self,
        images_by_organs: dict[str, list[SingleImage]],
        T0: v2.Transform,
    ) -> None:
        self.ims = []
        self.dataset_names = []
        self.organ_indices = {}

        start = 0

        for organ_name, val_images in images_by_organs.items():
            loaded = []

            for val_im in val_images:
                im = val_im.load(T0)
                loaded.append(im)
                self.dataset_names.append(organ_name)

            if loaded:
                self.ims.extend(loaded)
                end = start + len(loaded)
                self.organ_indices[organ_name] = list(range(start, end))
                start = end

        if len(self.ims) == 0:
            raise RuntimeError("No validation images loaded.")

        self.stacked = torch.stack(self.ims, dim=0)