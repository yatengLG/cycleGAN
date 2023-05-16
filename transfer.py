# -*- coding: utf-8 -*-
# @Author  : LG

import torch
from PIL import Image
from data.transform import Transforms
import numpy as np


class Transfer:
    def __init__(self, checkpoint:str, device='cuda:0'):
        self.device = device
        self.model = torch.load(checkpoint)
        self.model.to(self.device)

        self.model.eval()
        self.model.requires_grad_(False)

    def transfer(self, image:str):
        image = Image.open(image).convert('RGB')
        image = Transforms((256, 256), (256, 256), is_train=False)(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        fake: torch.Tensor = self.model(image)
        fake = fake.data
        fake_numpy = fake[0].cpu().float().numpy()
        fake_numpy = (np.transpose(fake_numpy, (1, 2, 0)) + 1) / 2.0 * 255
        return fake_numpy.astype(np.uint8)


if __name__ == '__main__':
    # 导入模型，进行风格迁移.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = Transfer('checkpoints/pretrained/latest_netG_B.pth', device)
    fake = t.transfer('datasets/horse2zebra/testA/n02381460_140.jpg')
    fake_image = Image.fromarray(fake)
    fake_image.show()
