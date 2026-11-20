import os
from dataclasses import dataclass

import torch
from torchvision.io import read_image
from torchvision.transforms import v2

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

    def load(self, T0: tuple[v2.Transform]) -> torch.Tensor:
        return T0(read_image(self.im_path))


class Organ:
    def __init__(self, base_path, organ) -> None:
        datasets = [os.path.join(base_path, organ, ds) for ds in os.listdir(os.path.join(base_path, organ))]
        self.train = []
        self.val = SingleImage(os.path.join(base_path, organ, VAL_DATA[organ]))
        for ds in datasets:
            for f in os.listdir(ds):
                im_path = os.path.join(ds, f)
                if im_path != self.val.im_path:
                    self.train.append(SingleImage(im_path))
