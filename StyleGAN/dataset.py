from glob import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from functions import *


class ImageDataset(Dataset):
    def __init__(self, root, transform_, unaligned=False):
        self.transform = transforms.Compose(transform_)
        self.unaligned = unaligned

        self.files = glob(os.path.join(root, '*.png'))

    def __getitem__(self, index):
        image = Image.open(self.files[index % len(self.files)])

        if image.mode != "RGB":
            image = to_rgb(image)

        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.files)

