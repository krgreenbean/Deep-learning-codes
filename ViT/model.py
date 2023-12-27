from module import *


class ViT(nn.Module):
    def __init__(self,
                 in_dim: int = 3,
                 patch_size: int = 16,
                 emb_dim: int = 768,
                 img_size: int = 128,
                 num_classes: int = 10):
        super().__init__()
        self.vit = nn.Sequential(
            PatchEmbedding(in_dim, patch_size, emb_dim, img_size),  #4,768
            TransformerEncoder(),   #b,p,768
            ClassificationHead(emb_dim, num_classes))  #b,10)

    def forward(self, x):
        out = self.vit(x)
        return out
