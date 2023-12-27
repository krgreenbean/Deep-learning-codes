from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from model import ViT
from module import *


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
print("Using Device:", DEVICE)

DATA_PATH = '/hdd1/data/STL10'
resultPATH = '/hdd1/result/ViT'
model_path = f"{resultPATH}/ViT.pth"
batch_size = 32
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(128),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

testset = torchvision.datasets.STL10(DATA_PATH,split='test', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
print("number of test batches:", len(testloader))

model = ViT().cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

correct = list(0. for i in range(10))
total = list(0. for i in range(10))
cr = 0
t = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        t += labels.size(0)
        cr += (predicted == labels.cuda()).sum().item()
        c = (predicted == labels.cuda()).squeeze()

        for i in range(batch_size):
            label = labels[i]
            total[label] += 1
            correct[label] += c[i].item()

for i in range(10):
    print('Accuracy of %7s: %2d %%' % ( 
            classes[i], correct[i] / total[i] * 100))
print('total accuracy on test images: %d %%' % (
            100 * cr / t))
