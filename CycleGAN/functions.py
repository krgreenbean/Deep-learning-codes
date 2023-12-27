import os
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn as nn
from PIL import Image
import torch


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def PSNR(ori_img, con_img):
    # 해당 이미지의 최대값 (채널 최대값 - 최솟값)
    max_pixel = 2.0
    mse = nn.MSELoss()(ori_img, con_img)
    psnr = 10 * math.log10(max_pixel**2 / mse)
    return psnr


"""loss functions"""


def recon_loss(pred, gt):
    loss_func = nn.MSELoss()
    loss = loss_func(pred, gt)
    return loss


def gen_loss(fake):
    return torch.mean((fake - 1)**2)


def dis_loss(fake, real):
    return torch.mean(fake**2) + torch.mean((real - 1)**2)

