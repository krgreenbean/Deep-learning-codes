import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt


from model import ResNet
from module import resblock


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
print("Using Device:", DEVICE)

DATA_PATH = '/hdd1/data/STL10'
resultPATH = '/hdd1/result/CAM'
model_path = f"{resultPATH}/CAM_resnet34.pth"
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


model = ResNet(resblock, [3, 4, 6, 3]).cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

correct = list(0. for i in range(10))
total = list(0. for i in range(10))
cr = 0
t = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs, _ = model(images.cuda())
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


'''inference'''
for i in range(30):
    img = testset[i][0]
    label = testset[i][1]
    batch_img = img[None, :, :, :].cuda()
    output, feature = model(batch_img)

    class_weight = model.fc.weight[int(label-1)].unsqueeze(-1).unsqueeze(-1)
    cam_ = torch.squeeze(feature) * class_weight

    cam = torch.sum(cam_,axis=0)
    cam = cam.detach().cpu().numpy()
    CAM = cv2.resize(cam, dsize=(224, 224))

    orig = img.detach().numpy()

    plt.imshow(np.rollaxis(orig, 0, 3))
    plt.imshow(CAM, alpha=0.4, cmap='rainbow')
    plt.savefig(f"{resultPATH}/{i+1}.png")
    plt.clf()
