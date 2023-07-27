from typing import Tuple, Iterable

import torch.nn as nn
from einops import rearrange
from torch.nn.functional import dropout, gelu
import torch.nn.functional as F
from models.timm import trunc_normal_, drop_path
import torch
# from models.amm import AMM
# from models.amm_fca_2 import AMM
from models.amm_dsr_fca import AMM
Tuple4i = Tuple[int, int, int, int]


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        fan_out = (m.kernel_size[0] * m.kernel_size[1] * m.out_channels) // m.groups
        nn.init.normal_(m.weight, std=(2.0 / fan_out) ** 0.5)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MixFeedForward(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                 dropout_p: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Depth-wise convolution
        self.conv = nn.Conv2d(hidden_features, hidden_features, (3, 3), padding=(1, 1),
                              bias=True, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout_p = dropout_p

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = gelu(x)
        x = dropout(x, p=self.dropout_p, training=self.training)
        x = self.fc2(x)
        x = dropout(x, p=self.dropout_p, training=self.training)
        return x


class EfficientAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 dropout_p: float = 0.0, sr_ratio: int = 1):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f'expected dim {dim} to be a multiple of num_heads {num_heads}.')

        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout_p = dropout_p

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            sr_ratio_tuple = (sr_ratio, sr_ratio)
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio_tuple, stride=sr_ratio_tuple)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        q = self.q(x)
        q = rearrange(q, ('b hw (m c) -> b m hw c'), m=self.num_heads)

        if self.sr_ratio > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = self.sr(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)

        x = self.kv(x)
        x = rearrange(x, 'b d (a m c) -> a b m d c', a=2, m=self.num_heads)
        k, v = x.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = rearrange(x, 'b m hw c -> b hw (m c)')
        x = self.proj(x)
        x = dropout(x, p=self.dropout_p, training=self.training)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False,
                 dropout_p: float = 0.0, drop_path_p: float = 0.0, sr_ratio: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = EfficientAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       dropout_p=dropout_p, sr_ratio=sr_ratio)
        self.drop_path_p = drop_path_p
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = MixFeedForward(dim, dim, hidden_features=dim * 4, dropout_p=dropout_p)

    def forward(self, x, h, w):
        skip = x
        x = self.norm1(x)
        x = self.attn(x, h, w)
        x = drop_path(x, p=self.drop_path_p, training=self.training)
        x = x + skip

        skip = x
        x = self.norm2(x)
        x = self.ffn(x, h, w)
        x = drop_path(x, p=self.drop_path_p, training=self.training)
        x = x + skip

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size: Tuple[int, int], stride: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, h, w


class MixTransformerStage(nn.Module):
    def __init__(
        self,
        patch_embed: OverlapPatchEmbed,
        blocks: Iterable[TransformerBlock],
        norm: nn.LayerNorm,
        amm
    ):
        super().__init__()
        self.patch_embed = patch_embed
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm
        self.amm = amm

    def forward(self, x):
        x, h, w = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, h, w)
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.amm(x)
        return x


class MixTransformer(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: Tuple4i = (64, 128, 256, 512),
        num_heads: Tuple4i = (1, 2, 4, 8),
        qkv_bias: bool = False,
        dropout_p: float = 0.0,
        drop_path_p: float = 0.0,
        depths: Tuple4i = (3, 4, 6, 3),
        sr_ratios: Tuple4i = (8, 4, 2, 1),
    ):
        super().__init__()

        self.stages = nn.ModuleList()
        for l in range(len(depths)):
            blocks = [
                TransformerBlock(dim=embed_dims[l], num_heads=num_heads[l], qkv_bias=qkv_bias,
                                 dropout_p=dropout_p, sr_ratio=sr_ratios[l],
                                 drop_path_p=drop_path_p * (sum(depths[:l])+i) / (sum(depths)-1))
                for i in range(depths[l])
            ]
            if l == 0:
                patch_embed = OverlapPatchEmbed((7, 7), stride=4, in_chans=in_chans,
                                                embed_dim=embed_dims[l])
            else:
                patch_embed = OverlapPatchEmbed((3, 3), stride=2, in_chans=embed_dims[l - 1],
                                                embed_dim=embed_dims[l])
            norm = nn.LayerNorm(embed_dims[l], eps=1e-6)
            amm = AMM(embed_dims[l], 16)
            self.stages.append(MixTransformerStage(patch_embed, blocks, norm, amm))

        # self.conv1 = nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False, dilation=1)
        # self.conv2 = nn.Conv2d(512, 512, 3, stride=1, padding=2, bias=False, dilation=2)
        # self.conv3 = nn.Conv2d(512, 512, 3, stride=1, padding=3, bias=False, dilation=3)
        self.init_weights()

    def init_weights(self):
        self.apply(_init_weights)

    def forward(self, x):
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        f_ori = outputs[3]

        f_1 = f_ori

        # f_2 = self.conv2(f_ori)
        # f_2 = F.relu(f_2)
        # f_3 = self.conv3(f_ori)
        # f_3 = conv3(f_ori)
        # f_he = (f_1 + f_2) / 2

        return f_1


def _mit_bx(embed_dims: Tuple4i, depths: Tuple4i) -> MixTransformer:
    return MixTransformer(
        embed_dims=embed_dims,
        num_heads=(1, 2, 5, 8),
        qkv_bias=True,
        depths=depths,
        sr_ratios=(8, 4, 2, 1),
        dropout_p=0.0,
        drop_path_p=0.1,
    )

def _load_pretrained_weights_(model):
    # state_dict = torch.hub.load_state_dict_from_url(model_url, progress=progress)
    state_dict = torch.load("./weights/mit_b2.pth", map_location='cpu')
    # state_dict = torch.hub.load_state_dict_from_url(model_url, progress=progress)
    del_keys = ['head.weight', 'head.bias']
    for k in del_keys:
        del state_dict[k]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('decode_head'):
            if k.endswith('.proj.weight'):
                k = k.replace('.proj.weight', '.weight')
                v = v[..., None, None]
            elif k.endswith('.proj.bias'):
                k = k.replace('.proj.bias', '.bias')
            elif '.linear_fuse.conv.' in k:
                k = k.replace('.linear_fuse.conv.', '.linear_fuse.')
            elif '.linear_fuse.bn.' in k:
                k = k.replace('.linear_fuse.bn.', '.bn.')

            if '.linear_c4.' in k:
                k = k.replace('.linear_c4.', '.layers.0.')
            elif '.linear_c3.' in k:
                k = k.replace('.linear_c3.', '.layers.1.')
            elif '.linear_c2.' in k:
                k = k.replace('.linear_c2.', '.layers.2.')
            elif '.linear_c1.' in k:
                k = k.replace('.linear_c1.', '.layers.3.')
        else:
            if 'patch_embed1.' in k:
                k = k.replace('patch_embed1.', 'stages.0.patch_embed.')
            elif 'patch_embed2.' in k:
                k = k.replace('patch_embed2.', 'stages.1.patch_embed.')
            elif 'patch_embed3.' in k:
                k = k.replace('patch_embed3.', 'stages.2.patch_embed.')
            elif 'patch_embed4.' in k:
                k = k.replace('patch_embed4.', 'stages.3.patch_embed.')
            elif 'block1.' in k:
                k = k.replace('block1.', 'stages.0.blocks.')
            elif 'block2.' in k:
                k = k.replace('block2.', 'stages.1.blocks.')
            elif 'block3.' in k:
                k = k.replace('block3.', 'stages.2.blocks.')
            elif 'block4.' in k:
                k = k.replace('block4.', 'stages.3.blocks.')
            elif 'norm1.' in k:
                k = k.replace('norm1.', 'stages.0.norm.')
            elif 'norm2.' in k:
                k = k.replace('norm2.', 'stages.1.norm.')
            elif 'norm3.' in k:
                k = k.replace('norm3.', 'stages.2.norm.')
            elif 'norm4.' in k:
                k = k.replace('norm4.', 'stages.3.norm.')

            if '.mlp.dwconv.dwconv.' in k:
                k = k.replace('.mlp.dwconv.dwconv.', '.mlp.conv.')

            if '.mlp.' in k:
                k = k.replace('.mlp.', '.ffn.')
        new_state_dict[k] = v
    # model.load_state_dict(new_state_dict, strict=False)
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("mit amm missing_keys: ", missing_keys)
        print("mit amm unexpected_keys: ", unexpected_keys)
def mit_b0():
    return _mit_bx(embed_dims=(32, 64, 160, 256), depths=(2, 2, 2, 2))


def mit_b1():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(2, 2, 2, 2))


def mit_b2():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 4, 6, 3))


def mit_b3():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 4, 18, 3))


def mit_b4():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 8, 27, 3))


def mit_b5():
    return _mit_bx(embed_dims=(64, 128, 320, 512), depths=(3, 6, 40, 3))


def mit_amm_backbone(pretrained=True):
    model = mit_b2()
    if pretrained:
        _load_pretrained_weights_(model)

    return model