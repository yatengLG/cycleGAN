# -*- coding: utf-8 -*-
# @Author  : LG

from PIL import Image
from torchvision.transforms import Resize, RandomCrop, InterpolationMode, RandomHorizontalFlip, ToTensor, Normalize, Compose


class Transforms:
    def __init__(self, load_size, crop_size, is_train=True):
        transforms = []
        resize_size = (load_size)
        transforms.append(Resize(resize_size, interpolation=InterpolationMode.BICUBIC))    # 使用双三次插值，更慢但更准
        if is_train:
            transforms.append(RandomCrop(crop_size))
            transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transforms = Compose(transforms)

    def __call__(self, image: Image.Image):
        image = self.transforms(image)
        return image

