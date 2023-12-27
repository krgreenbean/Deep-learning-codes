import os
import matplotlib.pyplot as plt
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

