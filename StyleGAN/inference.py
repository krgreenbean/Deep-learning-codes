from model import *
from dataset import *
from vgg import *


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
print("Using Device:", DEVICE)

DATA_PATH = '/hdd1/data/ffhq/images'
resultPATH = '/hdd1/result/StyleGAN'
gen_path = f"{resultPATH}/gen.pth"
disc_path = f"{resultPATH}/disc.pth"
enc_path = f"{resultPATH}/enc.pth"

feat_layers = ['conv4_1']
batch_size = 12
size = 4
criterion1 = nn.MSELoss
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(224),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

testloader = torch.utils.data.DataLoader(ImageDataset(f"{DATA_PATH}", split='test',
                                                      transform_=transform, debug_mode=False, unaligned=True),
                                         batch_size=batch_size,
                                         drop_last=True,
                                         shuffle=False, num_workers=4)
print("number of test batches:", len(testloader))

net = StyleGAN(indim=512, norm='batch').cuda()
net.load_state_dict(torch.load(gen_path))
net.eval()

dis = Discriminator(input_dim=3).cuda()
dis.load_state_dict(torch.load(disc_path))
dis.eval()

enc = VGG19Normalized().cuda()
enc.load_state_dict(torch.load(enc_path))
enc.eval()


"""test generate 10 batch images"""
with torch.no_grad():
    for i, img in enumerate(testloader):
        if i < 5:
            img = img.cuda().requires_grad_(False)
            vgg_feat_img = enc.get_features(img, feat_layers)[0]
            vgg_feat = torch.mean(vgg_feat_img, dim=(2, 3))
            fake = net(vgg_feat)

            save_image(fake, os.path.join(resultPATH, 'gen_faces{}.png'.format(i + 1)), normalize=True)
        else:
            break
