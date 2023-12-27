import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

"""BASE architectures"""


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim: int = 3, patch_size: int =16, emb_dim: int = 768, image_size: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_dim, emb_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c (h) (w) -> b (h w) c'),    #B, hw, C
        )
        self.class_token = nn.Parameter(torch.randn(1, 1, emb_dim)) #1,1,768.  class embedding token
        self.pos_token = nn.Parameter(torch.rand((image_size//patch_size)**2 + 1, emb_dim))    #hw+1, C

    def forward(self, x: Tensor):
        batch_size, _, _, _ = x.shape   #b, c, h, w
        x = self.patch_embedding(x)
        cls_tokens = repeat(self.class_token, '() n c -> b n c', b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_token
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int=768, num_heads: int=8, dropout: float=0. ):
        super().__init__()
        self.emb_dim= emb_dim
        self.num_heads = num_heads
        self.K = nn.Linear(emb_dim, emb_dim)
        self.Q = nn.Linear(emb_dim, emb_dim)
        self.V = nn.Linear(emb_dim, emb_dim)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_dim, emb_dim)
        self.scaling = (self.emb_dim // num_heads) ** -0.5

    def forward(self, x: Tensor, mask: Tensor= None):
        Q = rearrange(self.Q(x), 'b p (H c) -> b H p c', H= self.num_heads)
        K = rearrange(self.K(x), 'b p (H c) -> b H p c', H= self.num_heads)
        V = rearrange(self.V(x), 'b p (H c) -> b H p c', H= self.num_heads)
        attention_map = torch.einsum('bHqc, bHkc -> bHqk', Q, K)
        attention_map = attention_map * self.scaling   #bHqk
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            attention_map.mask_fill(~mask, fill_value)

        attention = F.softmax(attention_map, dim=-1)
        attention = self.att_drop(attention)
        #qv matmul attention map= [8,8,394,394] ->matmul with V [8,8,394,768]= [8,8,394,768]
        out = torch.einsum('bHpp, bHvc -> bHpc',attention, V)
        out = rearrange(out, 'b H p c -> b p (H c)')   #concat all heads' final output,[8,394,8*768]
        out = self.projection(out)  #b p 8*C -> b p 8*C
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForward(nn.Module):
    def __init__(self, in_dim, expansion: int=4, drop_p: float=0.):
        super().__init__()
        block = []
        block = [
            nn.Linear(in_dim, in_dim*expansion),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(in_dim*expansion, in_dim)
        ]
        self.MLP = nn.Sequential(*block)

    def forward(self, x):
        x = self.MLP(x)
        return x


"""BLOCK architectures"""


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_dim: int = 768,
                 drop_p: float = 0,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0):
        super().__init__()
        block = []
        block = [
            Residual(nn.Sequential(
                MultiHeadAttention(),
                nn.Dropout(drop_p))),
            nn.LayerNorm(emb_dim),
            Residual(nn.Sequential(
                FeedForward(emb_dim, expansion= forward_expansion, drop_p= forward_drop_p),
                nn.Dropout(drop_p))),
            nn.LayerNorm(emb_dim)
            ]
        self.transformer = nn.Sequential(*block)

    def forward(self, x):
        x = self.transformer(x)
        return x   #b,p,C


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = TransformerEncoderBlock()

    def forward(self, x, depth: int = 12):
        for i in range(depth):
            x = self.transformer(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, emb_dim: int = 768, num_class: int = 10):
        super().__init__()

        self.classification = nn.Sequential(
            Reduce('b p c -> b c', reduction='mean'),  # leave one mean value per channel
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_class)  # b,C -> b,10
        )

    def forward(self, x):
        out = self.classification(x)
        return out   #b,10

