# StyleGAN using ffhq dataset.

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import ImageDataset
from model import *
from functions import *
from vgg import *


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
print("Using Device:", DEVICE)

DATA_PATH = '/hdd1/data/ffhq/images'
batch_size = 32
noise_dim = 256
TRANSFORM = [transforms.ToTensor(),
             transforms.Resize(128),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

trainloader = torch.utils.data.DataLoader(ImageDataset(f"{DATA_PATH}", transform_=TRANSFORM, unaligned=True),
                                          batch_size=batch_size,
                                          drop_last=True,
                                          shuffle=True, num_workers=4)
print("number of train batches:", len(trainloader))

"""dataset load check"""
for i, batch in enumerate(trainloader):
    if i < 2:
        faces = batch.cpu()
        imshow(torchvision.utils.make_grid(faces))
    else:
        break


"""---------Training---------"""
learning_rate = 0.0002
epoch_start = 20
n_EPOCH = 40
resultPATH = mkdir('/hdd1/result/StyleGAN/result')
modelPATH = mkdir('/hdd1/result/StyleGAN')
feat_layers = ['conv4_1']

real_label = 1
fake_label = 0

net = StyleGAN(indim=512, norm='batch').cuda()
dis = Discriminator(input_dim=3).cuda()
enc = VGG19Normalized().cuda()
enc.eval()


if epoch_start != 0:
    net.load_state_dict(torch.load(f"{modelPATH}/gen.pth"))
    dis.load_state_dict(torch.load(f"{modelPATH}/disc.pth"))
    enc.load_state_dict(torch.load(f"{modelPATH}/stylemap.pth"))

g_opt = torch.optim.Adam(list(net.parameters()), lr=learning_rate)
d_opt = torch.optim.Adam(list(dis.parameters()), lr=learning_rate)

c = 0
for i_epo in range(epoch_start, n_EPOCH):
    for i, img in enumerate(trainloader):
        img = img.cuda().requires_grad_(False)
        vgg_feat_img = enc.get_features(img, feat_layers)[0]
        vgg_feat = torch.mean(vgg_feat_img, dim=(2, 3))


        '''train gen'''

        net.train(True)
        dis.train(False)
        for p_x in dis.parameters():
            p_x.requires_grad = False

        fake = net(vgg_feat)
        loss_rec = []
        for f_ in fake:
            h, w = f_.shape[2], f_.shape[3]
            img_ = F.interpolate(img, (h, w))
            loss_rec.append(recon_loss(f_, img_)[None])

        loss_rec = torch.cat(loss_rec).sum()
        loss_adv = gen_loss(dis(fake[-1]))

        vgg_feat_fake = enc.get_features(fake[-1], feat_layers)[0]
        loss_perc = recon_loss(vgg_feat_fake, vgg_feat_img)

        loss_gen = loss_rec + 0.1 * loss_adv + loss_perc

        g_opt.zero_grad()
        loss_gen.backward()
        g_opt.step()


        '''train dis'''

        net.train(False)
        dis.train(True)
        for p_x in dis.parameters():
            p_x.requires_grad = True

        real_feat = dis(img)
        fake_feat = dis(fake[-1].detach())

        loss_dis = dis_loss(fake_feat, real_feat)
        d_opt.zero_grad()
        loss_dis.backward()
        d_opt.step()

        if (i + 1) % 20 == 0:
            print("Epoch [ {}/{} ]  Step [ {}/{} ]  loss_dis: {:.5f}  loss_adv: {:.5f} loss_cyc : {:.5f}"
                  .format(i_epo, n_EPOCH, i + 1, len(trainloader), loss_dis.item(), loss_gen.item(),
                          loss_rec.item()))

        if c % 1000 == 0:
            fake_ = (fake[-1] + 1) / 2
            img = (img + 1) / 2
            save_image(fake_, os.path.join(resultPATH, '{}_rec.png'.format(c)), normalize=True)
            save_image(img, os.path.join(resultPATH, '{}_gt.png'.format(c)), normalize=True)

            # torch.save(net.state_dict(), f'tmp_weight/{c}_net_parms.pt')
        c += 1

print('Training Finished')

torch.save(net.state_dict(), f"{modelPATH}/gen.pth")
torch.save(dis.state_dict(), f"{modelPATH}/disc.pth")
torch.save(enc.state_dict(), f"{modelPATH}/enc.pth")

