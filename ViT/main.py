import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

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


VIT = ViT().cuda()

"""Training-"""
criterion = nn.CrossEntropyLoss()
learning_rate = 0.00002
epoch_start = 30
n_EPOCH = 100
resultPATH = mkdir('/hdd1/result/ViT')

if epoch_start != 0:
    VIT.load_state_dict(torch.load(f"{resultPATH}/ViT.pth"))

optimizer = torch.optim.Adam(VIT.parameters(), lr=learning_rate)

for epoch in range(epoch_start, n_EPOCH):
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        VIT.train(True)
        optimizer.zero_grad()

        output = VIT(images.cuda())
        loss = criterion(output, labels.cuda())
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print("Epoch [ {}/{} ]  Step [ {}/{} ]  loss : {:.5f}".format(epoch, n_EPOCH, i + 1,
                                                                          len(trainloader), loss.item()))
print('Training Finished')
torch.save(VIT.state_dict(), f"{resultPATH}/ViT.pth")
