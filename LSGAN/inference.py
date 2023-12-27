import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import *
from functions import *


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
print("Using Device:", DEVICE)

DATA_PATH = '/hdd1/data/STL10'
resultPATH = '/hdd1/result/LSGAN'
Gmodel_path = f"{resultPATH}/generator.pth"
Dmodel_path = f"{resultPATH}/discriminator.pth"
batch_size = 12
noise_size = 100
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(224),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

testset = torchvision.datasets.STL10(DATA_PATH,split='test', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
print("number of test batches:", len(testloader))

gen = Generator(noise_size).cuda()
gen.load_state_dict(torch.load(Gmodel_path))
gen.eval()
disc = Discriminator().cuda()
disc.load_state_dict(torch.load(Dmodel_path))
disc.eval()
criterion1 = nn.MSELoss

"""test generate 10 images"""
for i in range(10):
    z = torch.randn(batch_size, noise_size).to(DEVICE)
    fake_images = gen(z, batch_size)

    save_image(fake_images, os.path.join(resultPATH, 'fake{}.png'.format(i + 1)), normalize=True)
