# LSGAN using STL10 dataset.
# STL10 dataset: 10 labels, train dataset 5000, test dataset 8000.

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import *
from functions import *

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
print("Using Device:", DEVICE)

DATA_PATH = '/hdd1/data/STL10'
batch_size = 12
noise_size = 100
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(128),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

trainset = torchvision.datasets.STL10(DATA_PATH, split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
testset= torchvision.datasets.STL10(DATA_PATH,split='test',download=True, transform=transform)
testloader= torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
print("number of train batches:", len(trainloader), "\nnumber of test batches:", len(testloader))

"""(check dataset)"""
dataiter = iter(trainloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print(''.join(f'{ classes[labels[i]]:10s}' for i in range(batch_size)))


"""---------3.model 정의---------"""
generator = Generator(noise_size).cuda()
discriminator = Discriminator().cuda()

"""---------4. Training---------"""
criterion = F.mse_loss
learning_rate = 0.00002
epoch_start = 0
n_EPOCH = 50
resultPATH = mkdir('/hdd1/result/LSGAN')

if epoch_start != 0:
    generator.load_state_dict(torch.load(f"{resultPATH}/generator.pth"))
    discriminator.load_state_dict(torch.load(f"{resultPATH}/discriminator.pth"))

g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

for epoch in range(epoch_start, n_EPOCH):
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(DEVICE)
        fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(DEVICE)

        real_images = images.to(DEVICE)

        """train generator"""
        generator.train(True)
        discriminator.train(False)
        g_optimizer.zero_grad()

        z = torch.randn(batch_size, noise_size).to(DEVICE)
        fake_images = generator(z)

        g_loss = criterion(discriminator(fake_images), real_label)
        g_loss.backward()
        g_optimizer.step()

        """train discriminator"""
        generator.train(False)
        discriminator.train(True)
        d_optimizer.zero_grad()

        d_fakeloss = criterion(discriminator(fake_images.detach()), fake_label)
        d_realloss = criterion(discriminator(real_images), real_label)
        d_loss = d_realloss + d_fakeloss

        d_loss.backward()
        d_optimizer.step()

        if (i + 1) % 20 == 0:
            print("Epoch [ {}/{} ]  Step [ {}/{} ]  d_loss : {:.5f}  g_loss : {:.5f}"
                  .format(epoch, n_EPOCH, i + 1, len(trainloader), d_loss.item(), g_loss.item()))

    save_image(fake_images, os.path.join(resultPATH, 'fake{}.png'.format(epoch + 1)), normalize=True)
    save_image(real_images, os.path.join(resultPATH, 'real{}.png'.format(epoch + 1)), normalize=True)

print('Training Finished')

torch.save(generator.state_dict(), f"{resultPATH}/generator_10.pth")
torch.save(discriminator.state_dict(), f"{resultPATH}/discriminator_10.pth")
