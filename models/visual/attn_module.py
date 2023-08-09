import torch
from torch import nn
from timm.models.layers import DropPath, Mlp
from einops import rearrange


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, img_size=(0, 0), dim_in=1280, embed_dim=768, norm_layer=None):
        super().__init__()
        H, W = img_size[0], img_size[1]
        self.img_size = img_size
        self.num_patches = H * W
        self.proj = nn.Linear(dim_in, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
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
        self.scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, kv_, q_):
        B, N, C = q_.shape

        kv = (
            self.kv(kv_)
            .reshape(B, 1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q = (
            self.q(q_)
            .reshape(B, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        q = q[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # ####################
        """
            Attention map:
            torchvision.utils.save_image(rearrange(attn.clone().squeeze(), 'b (h w) c -> b c h w', h=48, w=24).sum(dim=1).mean(0).unsqueeze(0).unsqueeze(1), "atten_main.png")
        """
        # ####################

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


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

    def forward(self, x_in):
        x = x_in[:,:-1, :]
        x_aux = x_in[:,-1, None, :]

        x = self.norm1(x)
        x_aux = self.norm1(x_aux)

        # attn(kv, q)
        # CA for image feature based on audio feature
        x = x + self.drop_path(self.attn(x_aux, x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


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
        img_size=(0,0)
    ):
        super().__init__()
        # self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.patch_embed = PatchEmbed(
            img_size=(img_size[0], img_size[1]), dim_in=dim_in, embed_dim=embed_dim
        )
        self.audio_embed = nn.Linear(128, embed_dim)

        self.pos_embed_1 = nn.Parameter(
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
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward_features(self, x_img, x_aux):
        x_img = self.patch_embed(x_img)
        x_aux = self.audio_embed(x_aux)

        x_img = self.pos_drop(x_img + self.pos_embed_1)
        x_aux = self.pos_drop(x_aux)

        x_in = torch.cat((x_img, x_aux[:, None,:]),dim=1)
        x_img = self.blocks(x_in)
        x_img = self.norm(x_img)

        return x_img


    def forward(self, x, x_aux):
        x = self.forward_features(x, x_aux)
        return x