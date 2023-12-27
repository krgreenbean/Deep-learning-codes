'''Unet using resnet, STL10 dataset.
STL10 dataset: 10 labels, train dataset 5000, test dataset 8000.'''

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

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

model = UNet().cuda()

"""#### 4. Training"""
criterion = nn.MSELoss
learning_rate = 0.001
epoch_start = 0
n_EPOCH = 10
resultPATH = mkdir('/hdd1/result/unet')

if epoch_start != 0:
    model.load_state_dict(torch.load(f"{resultPATH}/unet.pth"))

optimizer = optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(epoch_start, n_EPOCH):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        model.train(True)
        optimizer.zero_grad()

        outputs = model(inputs.cuda())
        loss = criterion()(outputs.to(DEVICE), inputs.to(DEVICE))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 20 == 0:
            print("Epoch [ {}/{} ]  Step [ {}/{} ]  loss : {:.5f}".format
                  (epoch, n_EPOCH, i + 1, len(trainloader), loss.item()))

print('Training Finished')
torch.save(model.state_dict(), f"{resultPATH}/unet.pth")
