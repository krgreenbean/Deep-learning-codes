import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import *
from model import *

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
print("Using Device:", DEVICE)

TRANSFORM = [transforms.ToTensor(),
             transforms.Resize(256),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
batch_size = 6
DATA_PATH = '/hdd1/data/horse2zebra'

"""load data"""
trainloader = torch.utils.data.DataLoader(ImageDataset(f"{DATA_PATH}", split='train',
                                                       transform_=TRANSFORM, debug_mode=False, unaligned=True),
                                          batch_size=batch_size,
                                          drop_last=True,
                                          shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(ImageDataset(f"{DATA_PATH}", split='test',
                                                      transform_=TRANSFORM, debug_mode=False, unaligned=True),
                                         batch_size=batch_size,
                                         drop_last=True,
                                         shuffle=False, num_workers=4)

print("number of train batches:", len(trainloader), "\nnumber of test batches:", len(testloader))


"""---------(check dataset)---------"""
for i, batch in enumerate(trainloader):
    if i < 1:
        horses = batch[0].cpu()
        zebras = batch[1].cpu()
        imshow(torchvision.utils.make_grid(horses))
        imshow(torchvision.utils.make_grid(zebras))
    else:
        break


"""---------Training---------"""
learning_rate = 0.0001
epoch_start = 100
n_EPOCH = 110
modelPATH = mkdir('/hdd1/result/CycleGAN')
resultPATH = mkdir('/hdd1/result/CycleGAN/result')

gen_hz = Generator().cuda()
gen_zh = Generator().cuda()
disc_h = Discriminator().cuda()
disc_z = Discriminator().cuda()

if epoch_start != 0:
    gen_hz.load_state_dict(torch.load(f"{modelPATH}/gen_hz.pth"))
    gen_zh.load_state_dict(torch.load(f"{modelPATH}/gen_zh.pth"))
    disc_h.load_state_dict(torch.load(f"{modelPATH}/disc_h.pth"))
    disc_z.load_state_dict(torch.load(f"{modelPATH}/disc_z.pth"))

g_optimizer = torch.optim.Adam(list(gen_hz.parameters()) + list(gen_zh.parameters()), lr=learning_rate)
d_optimizer = torch.optim.Adam(list(disc_h.parameters()) + list(disc_z.parameters()), lr=learning_rate)

real_label = torch.full((batch_size, 1, 1, 1), 1, dtype=torch.float32, requires_grad=False).to(DEVICE)
fake_label = torch.full((batch_size, 1, 1, 1), 0, dtype=torch.float32, requires_grad=False).to(DEVICE)


for epoch in range(epoch_start, n_EPOCH):
    for i, data in enumerate(trainloader):
        horses, zebras = data
        horses = horses.to(DEVICE)
        zebras = zebras.to(DEVICE)

        """train generator"""
        gen_hz.train(True)
        gen_zh.train(True)
        disc_h.train(False)
        disc_z.train(False)

        g_optimizer.zero_grad()

        fake_z = gen_hz(horses)
        recon_h = gen_zh(fake_z)
        fake_h = gen_zh(zebras)
        recon_z = gen_hz(fake_h)

        cyc_loss = (recon_loss(horses, recon_h) + recon_loss(zebras, recon_z)) * 5
        adv_loss = (gen_loss(disc_z(fake_z)) + gen_loss(disc_h(fake_h))) / 2

        g_loss = adv_loss + cyc_loss
        g_loss.backward()
        g_optimizer.step()

        """train discriminator"""
        gen_hz.train(False)
        gen_zh.train(False)
        disc_h.train(True)
        disc_z.train(True)

        d_optimizer.zero_grad()

        real_feat_H = disc_h(horses)
        real_feat_Z = disc_z(zebras)

        d_loss = dis_loss(disc_h(fake_h.detach()), real_feat_H)
        d_loss += (dis_loss(disc_z(fake_z.detach()), real_feat_Z))
        d_loss /= 2
        d_loss.backward()
        d_optimizer.step()

        if (i + 1) % 20 == 0:
            print("Epoch [ {}/{} ]  Step [ {}/{} ]  d_loss : {:.5f}  g_loss : {:.5f}".format(epoch, n_EPOCH, i + 1, len(trainloader), d_loss.item(), g_loss.item()))

    save_image(fake_z, os.path.join(resultPATH, 'fake_z{}.png'.format(epoch + 1)), normalize=True)
    save_image(fake_h, os.path.join(resultPATH, 'fake_h{}.png'.format(epoch + 1)), normalize=True)


print('Training Finished')

torch.save(gen_hz.state_dict(), f"{modelPATH}/gen_hz.pth")
torch.save(gen_zh.state_dict(), f"{modelPATH}/gen_zh.pth")
torch.save(disc_h.state_dict(), f"{modelPATH}/disc_h.pth")
torch.save(disc_z.state_dict(), f"{modelPATH}/disc_z.pth")

