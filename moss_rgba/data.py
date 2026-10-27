"""Aligned RGBA image loading; RGB and alpha share every resize and crop."""
from pathlib import Path

import torch
from PIL import Image, ImageOps, AvifImagePlugin
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def init_worker(_):
    torch.set_num_threads(1)
    AvifImagePlugin.DEFAULT_MAX_THREADS = 1


class RGBADataset(Dataset):
    def __init__(self, root, resolution=1024, blend_probability=0.3, training=True):
        self.paths = sorted(p for p in Path(root).rglob('*')
                            if p.suffix.lower() in {'.png', '.webp', '.avif', '.tif', '.tiff'})
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(resolution) if training else transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])
        self.blend_probability = blend_probability

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = ImageOps.exif_transpose(image).convert('RGBA')
            rgba = self.transform(image) * 2 - 1
        if torch.rand(()) < self.blend_probability:
            alpha = (rgba[3:] + 1) * 0.5
            background = torch.randint(0, 3, (3, 1, 1)).float() - 1
            rgba[:3] = rgba[:3] * alpha + background * (1 - alpha)
            rgba[3:] = 1
        return rgba
