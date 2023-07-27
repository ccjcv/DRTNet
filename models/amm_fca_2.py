import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from models.FcaNet.fca_layer import MultiSpectralAttentionLayer
from einops import rearrange
class MEAttention(nn.Module):
    def __init__(self, dim):
        super(MEAttention, self).__init__()
        self.num_heads = 8
        self.coef = 4
        self.query_liner = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.coef * self.num_heads
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.proj = nn.Linear(dim * self.coef, dim)

    def forward(self, x):#(b,c,h,w)
        x = rearrange(x, 'b c h w -> b (h w) c')
        B, N, C = x.shape
        x = self.query_liner(x)
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1,
                                                     3)  #(1, 32, 225, 32)

        attn = self.linear_0(x)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))

        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)#(24,256,512)
        x = rearrange(x, 'b (h w) c -> b c h w', h=16, w=16)

        return x
def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = torch.exp(-(x - mean) ** 2 / (2 * std ** 2)) / sigma
    return x_out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate_fca(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, c2wh=dict([(64,56), (128,28), (320,14) ,(512,7)])):
        super(ChannelGate_fca, self).__init__()
        self.att = MultiSpectralAttentionLayer(gate_channels, c2wh[gate_channels], c2wh[gate_channels],
                                          reduction=reduction_ratio, freq_sel_method='top16')

    def forward(self, x):
        channel_att_sum = self.att(x)
        x = x * channel_att_sum
        return x

class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.pool = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_pool = self.pool(x)
        x_out = self.spatial(x_pool)

        # Gauss modulation
        mean = torch.mean(x_out).detach()
        std = torch.std(x_out).detach()
        scale = GaussProjection(x_out, mean, std)

        # scale = scale / torch.max(scale)
        return x * scale

class AMM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(AMM, self).__init__()
        self.ChannelAMM = ChannelGate_fca(gate_channels, reduction_ratio)
        self.SpatialAMM = SpatialGate()
        # self.E_Attention = MEAttention(gate_channels)

    def forward(self, x):
        x_out = self.ChannelAMM(x)
        x_out = self.SpatialAMM(x_out)
        return x_out