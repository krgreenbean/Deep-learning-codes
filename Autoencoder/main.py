'''Autoencoder using resnet, STL10 dataset.
STL10 dataset: 10 labels, train dataset 5000, test dataset 8000.'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.functional as F

from model import *
from functions import *

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
print("Using Device:", DEVICE)

DATA_PATH = '/hdd1/data/STL10'
batch_size = 16
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(128),
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


'''models'''
model = Net().to(DEVICE)
mix_model = MixNet().to(DEVICE)


"""Training"""
learning_rate = 0.00001
epoch_start = 20
n_EPOCH = 60
resultPATH = mkdir('/hdd1/result/autoencoder')

if epoch_start != 0:
    model.load_state_dict(torch.load(f"{resultPATH}/autoencoder.pth"))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epoch_start, n_EPOCH):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        model.train(True)
        optimizer.zero_grad()

        z, outputs = model(images.cuda())
        loss = F.mse_loss(outputs.to(DEVICE), images.to(DEVICE))
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print("Epoch [ {}/{} ]  Step [ {}/{} ]  loss : {:.5f}".format
                  (epoch, n_EPOCH, i + 1, len(trainloader), loss.item()))

print('Training Finished')
torch.save(model.state_dict(), f"{resultPATH}/autoencoder.pth")
