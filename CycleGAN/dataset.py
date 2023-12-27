import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random

from functions import *


class ImageDataset(Dataset):
    def __init__(self, root, transform_, debug_mode, split='train', unaligned=False):
        self.transform = transforms.Compose(transform_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, f"{split}A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{split}B") + "/*.*"))

        if debug_mode:
            self.files_A= self.files_A[:100]
            self.files_B= self.files_B[:100]

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B= Image.open(self.files_B[random.randint(0,len(self.files_B)-1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        return item_A, item_B

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
