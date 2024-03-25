import math
from collections import deque
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from timm.models.layers import DropPath, Mlp

# from timm.models.vision_transformer import Attention
from functools import partial
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.autograd import Variable


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, img_size=(46, 30), dim_in=1280, embed_dim=768, norm_layer=None):
        super().__init__()
        H, W = img_size[0], img_size[1]
        self.img_size = img_size

        # self.proj = nn.Conv2d(1280, 512, kernel_size=3, stride=patch_size)
        self.proj = nn.Linear(dim_in, embed_dim)
        self.num_patches = H * W
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        self.num_patches = H * W
        # assert (
        #     H == self.img_size[0] and W == self.img_size[1]
        # ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj(x)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.k = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.v = nn.Linear(dim, dim * 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def projection(self, x, f_proj):
        B, N, C = x.shape
        out = (
            f_proj(x)
            .reshape(B, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        return out[0]
    
    def forward(self, x_q, x_k, x_v, attn_mask=None, size=None):
        B, N, C = x_q.shape
    
        q = self.projection(x_q, self.q)
        k = self.projection(x_k, self.k)
        v = self.projection(x_v, self.v)

        attn = (q @ k.transpose(-2, -1)) * self.scale 


        if attn_mask is not None:
            attn_mask = F.interpolate(
                attn_mask,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            attn_mask = rearrange(attn_mask, "b n h w -> b n (h w)").unsqueeze(-1)
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        attn = self.attn_drop(torch.sigmoid(attn))
        # ####################
        """
            Attention map:
            import torchvision
            torchvision.utils.save_image(rearrange(attn.clone().squeeze().unsqueeze(-1), 'b (h w) c -> b c h w', h=120, w=90).sum(dim=1).mean(0).unsqueeze(0).unsqueeze(1), "atten_main.png")
            import torchvision
            torchvision.utils.save_image(rearrange(attn.clone().squeeze().unsqueeze(-1), 'b (h w) c -> b c h w', w=int(1280/4), h=int(720/4)), "atten_main.png")
        """
        # ####################
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mode="CA"
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.forward_method = self.forward_ca if mode == "CA" else self.forward_sa

    def SDPAttention(self, q, k, v, attn_mask=None, size=None):
        output, attn_ = self.drop_path(self.attn(q, k, v, attn_mask=attn_mask, size=size))
        q = q + output
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q, attn_

    def forward_ca(self, x, attn_mask=None, size=None):
        f_v, f_a = x["visual"], x["audio"]
        f_v = self.norm1(f_v)
        f_a = self.norm1(f_a)

        # attn(q, k, v)
        # Attention on VISUAL feature based on AUDIO feature
        f_v, attn_v = self.SDPAttention(f_v, f_a, f_a, attn_mask, size=size)
        # Attention on AUDIO feature based on VISUAL feature
        f_a, attn_a = self.SDPAttention(f_a, f_v, f_v)
        return f_v, f_a, attn_v

    def forward_sa(self, x, attn_mask=None, size=None):
        f_v = self.norm1(x)
        # attn(q, k, v)
        f_v, attn_v = self.SDPAttention(f_v, f_v, f_v)
        return f_v, attn_v

    def forward(self, pack, **kwargs):
        return self.forward_method(pack, **kwargs)

class CROSS_ATTENTION(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        dim_in=1280,
    ):
        super().__init__()

        self.patch_embed_v = PatchEmbed(
            img_size=(128, 128), dim_in=dim_in, embed_dim=embed_dim
        )

        self.patch_embed_a = PatchEmbed(
            img_size=(1, 1), dim_in=dim_in, embed_dim=embed_dim
        )

        self.pos_embed_v = nn.Parameter(
            torch.zeros(1, self.patch_embed_v.num_patches, embed_dim)
        )
        self.pos_embed_a = nn.Parameter(
            torch.zeros(1, self.patch_embed_a.num_patches, embed_dim)
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # for i in range(depth):
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU,
                    mode="CA"
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, f_v, f_a, attn_mask=None, size=None):
        f_v = self.patch_embed_v(f_v)
        f_a = self.patch_embed_a(f_a)
        # f_v = self.pos_drop(f_v + self.pos_embed_v)
        # f_a = self.pos_drop(f_a + self.pos_embed_a)
        f_v = self.pos_drop(f_v)
        f_a = self.pos_drop(f_a)
        for layer in self.blocks:
            f_in = {"visual":f_v,"audio":f_a}
            f_v, f_a, attn_v = layer(f_in, attn_mask=attn_mask, size=size)
        f_v = self.norm(f_v)
        # f_a2v = self.norm(f_a2v)
        return f_v, f_a, attn_v
        # return f_v, torch.cat((f_a, f_a2v), dim=1)


class SELF_ATTENTION(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        dim_in=1280,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=(128, 128), dim_in=dim_in, embed_dim=embed_dim
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU,
                    mode="SA",
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, f_v):
        f_v = self.patch_embed(f_v)
        f_v = self.pos_drop(f_v)
        for layer in self.blocks:
            f_v, attn_v = layer(f_v)
        f_v = self.norm(f_v)
        # f_a2v = self.norm(f_a2v)
        return f_v, attn_v
        # return f_v, torch.cat((f_a, f_a2v), dim=1)
