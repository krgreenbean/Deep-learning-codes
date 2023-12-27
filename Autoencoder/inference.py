import torch
import torchvision
import torchvision.transforms as transforms
import os
from torchvision.utils import save_image
import torch.nn.functional as F

from model import Net
from functions import PSNR


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
print("Using Device:", DEVICE)

DATA_PATH = '/hdd1/data/STL10'
resultPATH = '/hdd1/result/autoencoder'
model_path = f"{resultPATH}/autoencoder.pth"
batch_size = 32
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(224),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

testset = torchvision.datasets.STL10(DATA_PATH,split='test', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
print("number of test batches:", len(testloader))

model = Net().cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

"""save example results"""
for i in range(10):
    img = testset[i][0].cpu()
    img = img[None, :, :, :]
    z, output = model(img.cuda())

    save_image(img, os.path.join(resultPATH, 'orig{}.png'.format(i + 1)))
    save_image(output, os.path.join(resultPATH, 'recon{}.png'.format(i + 1)))

MSE = 0.0
psnr = 0.0
with torch.no_grad():
    for data in testloader:
        images, _ = data
        z, outputs = model(images.cuda())
        MSE += F.mse_loss(outputs.to(DEVICE), images.to(DEVICE))
        psnr += PSNR(outputs.to(DEVICE), images.to(DEVICE))

    MSE = MSE / len(testloader)
    psnr = psnr / len(testloader)

print('total MSE on test images: %f' % MSE)  # lower better
print('total PSNR on test images: %f' % psnr)  # higher better

