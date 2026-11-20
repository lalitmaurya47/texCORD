import torch
from torchvision.transforms import v2


def base_transforms(im_res) -> v2.Transform:

    im_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((im_res, im_res)),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])

    return im_transforms


def random_augmentations(fill_val: float):

    color_transforms = v2.Compose([
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 1))]),
        ColorJitter(brightness=0.5, contrast=0.25),
        v2.RandomErasing(p=0.5, scale=(0.01, 0.05), ratio=(0.1, 1.), value=fill_val),
    ])

    spatial_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.25),
        v2.RandomVerticalFlip(p=0.25),
        v2.RandomRotation(degrees=15, fill=fill_val),
    ])

    return color_transforms, spatial_transforms


class ColorJitter(torch.nn.Module):
    def __init__(self, brightness, contrast):
        super().__init__()
        self.color_jitter = v2.ColorJitter(
            brightness=brightness,
            contrast=contrast
        )

    def forward(self, x):
        # [-1, 1] -> [0, 1]
        x = (x + 1) * 0.5

        x = self.color_jitter(x)

        # [0, 1] -> [-1, 1]
        x = x * 2 - 1
        return x.clamp(-1, 1)
