import os
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn.functional as F


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
    mse = F.mse_loss(ori_img, con_img)
    psnr = 10 * math.log10(max_pixel**2 / mse)
    return psnr
