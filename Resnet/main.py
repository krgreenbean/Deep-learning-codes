"""resnet using STL10 dataset.
STL10 dataset: 10 labels, train dataset 5000, test dataset 8000."""

import torch
from torch.utils.data import DataLoader
from torch import optim
import torchvision
import torchvision.transforms as transforms

from functions import *
from module import *
from model import ResNet

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
print("Using Device:", DEVICE)

DATA_PATH = '/hdd1/data/STL10'
batch_size = 32
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(224),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

trainset = torchvision.datasets.STL10(DATA_PATH, split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testset= torchvision.datasets.STL10(DATA_PATH,split='test',download=True, transform=transform)
testloader= torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
print("number of train batches:", len(trainloader), "\nnumber of test batches:", len(testloader))

"""(check dataset)"""
dataiter = iter(trainloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print(''.join(f'{ classes[labels[i]]:7s}' for i in range(batch_size)))


resnet34 = ResNet(resblock, [3, 4, 6, 3]).cuda()
# resnet18= ResNet(resblock, [2,2,2,2]).cuda()
# resnet50= ResNet(BottleNeck, [3,4,6,3]).cuda()
# resnet101= ResNet(BottleNeck, [3, 4, 23, 3]).cuda()
# resnet152= ResNet(BottleNeck, [3, 8, 36, 3]).cuda()


"""Training"""
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
epoch_start = 41
n_EPOCH = 70
resultPATH = mkdir('/hdd1/result/Resnet')

if epoch_start != 0:
    resnet34.load_state_dict(torch.load(f"{resultPATH}/resnet34.pth"))

optimizer = optim.Adam(resnet34.parameters(), lr=learning_rate)

for epoch in range(epoch_start, n_EPOCH):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        resnet34.train(True)
        optimizer.zero_grad()

        outputs = resnet34(images.cuda())
        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print("Epoch [ {}/{} ]  Step [ {}/{} ]  loss : {:.5f}".format
                  (epoch, n_EPOCH, i + 1, len(trainloader), loss.item()))

print('Training Finished')
torch.save(resnet34.state_dict(), f"{resultPATH}/resnet34.pth")
