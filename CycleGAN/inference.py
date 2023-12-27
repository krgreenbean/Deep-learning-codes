from torchvision.utils import save_image

from model import *
from dataset import *


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
print("Using Device:", DEVICE)

DATA_PATH = '/hdd1/data/horse2zebra'
resultPATH = '/hdd1/result/CycleGAN'
gen_hz_path = f"{resultPATH}/gen_hz.pth"
gen_zh_path = f"{resultPATH}/gen_zh.pth"
disc_h_path = f"{resultPATH}/disc_h.pth"
disc_z_path = f"{resultPATH}/disc_z.pth"

batch_size = 12
criterion1 = nn.MSELoss
TRANSFORM = [transforms.ToTensor(),
             transforms.Resize(256),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

testloader = torch.utils.data.DataLoader(ImageDataset(f"{DATA_PATH}", split='test',
                                                      transform_=TRANSFORM, debug_mode=False, unaligned=True),
                                         batch_size=batch_size,
                                         drop_last=True,
                                         shuffle=False, num_workers=4)
print("number of test batches:", len(testloader))

gen_hz = Generator().cuda()
gen_hz.load_state_dict(torch.load(gen_hz_path))
gen_hz.eval()
gen_zh = Generator().cuda()
gen_zh.load_state_dict(torch.load(gen_zh_path))
gen_zh.eval()


"""test generate 10 batch images"""
with torch.no_grad():
    for i, batch in enumerate(testloader):
        if i < 5:
            horses = batch[0].cpu()
            zebras = batch[1].cpu()
            recon_zebras = gen_hz(horses.cuda())
            recon_horses = gen_zh(zebras.cuda())
            save_image(recon_zebras, os.path.join(resultPATH, 'recon_z{}.png'.format(i + 1)), normalize=True)
            save_image(recon_horses, os.path.join(resultPATH, 'recon_h{}.png'.format(i + 1)), normalize=True)
        else:
            break

