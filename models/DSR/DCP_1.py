import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.FcaNet.fca_layer import MultiSpectralAttentionLayer
def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = torch.exp(-(x - mean) ** 2 / (2 * std ** 2)) / sigma
    return x_out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
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
        x_pool = self.pool(x)#(8,2,32,32)
        x_out = self.spatial(x_pool)#(8,1,32,32)

        # Gauss modulation
        # mean = torch.mean(x_out).detach()
        # std = torch.std(x_out).detach()
        # scale = GaussProjection(x_out, mean, std)

        # scale = scale / torch.max(scale)
        return x_out
class DCP_1(nn.Module):
    def __init__(self, inplanes = 320):
        super().__init__()
        inplanes_fen = int(inplanes / 4)
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.convp_1 = nn.Conv2d(inplanes_fen, inplanes_fen, kernel_size=3, stride=1, padding=1)
        self.bnp_1 = nn.BatchNorm2d(inplanes_fen)
        self.convp_2 = nn.Conv2d(inplanes_fen, inplanes_fen, kernel_size=3 + 1 * 2, stride=1, padding=1 + 1)
        self.bnp_2 = nn.BatchNorm2d(inplanes_fen)
        self.convp_3 = nn.Conv2d(inplanes_fen, inplanes_fen, kernel_size=3 + 2 * 2, stride=1, padding=1 + 2)
        self.bnp_3 = nn.BatchNorm2d(inplanes_fen)
        self.convp_4 = nn.Conv2d(inplanes_fen, inplanes_fen, kernel_size=3 + 3 * 2, stride=1, padding=1 + 3)
        self.bnp_4 = nn.BatchNorm2d(inplanes_fen)
        c2wh = dict([(64, 56), (128, 28), (320, 14), (512, 7)])
        self.fca = MultiSpectralAttentionLayer(inplanes, c2wh[inplanes], c2wh[inplanes],
                                          reduction=16, freq_sel_method = 'top16')

        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        self.SpatialAMM = SpatialGate()

        self.conv3 = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)

    def forward(self, x):
        b, c, h, w = x.size()
        x_in = F.relu(self.bn1(self.conv1(x)), inplace=True)#(8,256,32,32)
        channel_fen = int(c / 4)
        x_1 = x_in[:, :channel_fen, :, :]
        x_2 = x_in[:, channel_fen: 2 * channel_fen, :, :]
        x_3 = x_in[:, 2 * channel_fen: 3 * channel_fen, :, :]
        x_4 = x_in[:, 3 * channel_fen:, :, :]

        x_1 = F.relu(self.bnp_1(self.convp_1(x_1)), inplace=True)
        x_2 = F.relu(self.bnp_2(self.convp_2(x_2)), inplace=True)
        x_3 = F.relu(self.bnp_3(self.convp_3(x_3)), inplace=True)
        x_4 = F.relu(self.bnp_4(self.convp_4(x_4)), inplace=True)

        f_fca, w_fc = self.fca(x_in)
        w_fc_1 = w_fc[:, :channel_fen, :, :]
        w_fc_2 = w_fc[:, channel_fen: 2 * channel_fen, :, :]
        w_fc_3 = w_fc[:, 2 * channel_fen: 3 * channel_fen, :, :]
        w_fc_4 = w_fc[:, 3 * channel_fen:, :, :]

        # Gauss modulation
        # mean_1 = torch.mean(w_fc_1).detach()
        # std_1 = torch.std(w_fc_2).detach()
        # scale_1 = GaussProjection(w_fc_1, mean_1, std_1).expand_as(x_1)
        #
        # mean_2 = torch.mean(w_fc_2).detach()
        # std_2 = torch.std(w_fc_2).detach()
        # scale_2 = GaussProjection(w_fc_2, mean_2, std_2).expand_as(x_2)
        #
        # mean_3 = torch.mean(w_fc_3).detach()
        # std_3 = torch.std(w_fc_3).detach()
        # scale_3 = GaussProjection(w_fc_3, mean_3, std_3).expand_as(x_3)
        #
        # mean_4 = torch.mean(w_fc_4).detach()
        # std_4 = torch.std(w_fc_4).detach()
        # scale_4 = GaussProjection(w_fc_4, mean_4, std_4).expand_as(x_4)

        # y1 = x_1 * scale_1
        # y2 = x_2 * scale_2
        # y3 = x_3 * scale_3
        # y4 = x_4 * scale_4
        y1 = x_1 * w_fc_1
        y2 = x_2 * w_fc_2
        y3 = x_3 * w_fc_3
        y4 = x_4 * w_fc_4
        y_all = torch.cat((y1, y2, y3, y4), dim=1)
        y_channel = F.relu(self.bn2(self.conv2(y_all)), inplace=True)  # (8,256,32,32)


        w_spatial = self.SpatialAMM(x_in)
        #进行条形池化
        x_ph_1, x_pw_1 = self.pool_h(x_1), self.pool_w(x_1)
        x_pool_1 = torch.matmul(x_ph_1, x_pw_1)
        x_ph_2, x_pw_2 = self.pool_h(x_2), self.pool_w(x_2)
        x_pool_2 = torch.matmul(x_ph_2, x_pw_2)
        x_ph_3, x_pw_3 = self.pool_h(x_3), self.pool_w(x_3)
        x_pool_3 = torch.matmul(x_ph_3, x_pw_3)
        x_ph_4, x_pw_4 = self.pool_h(x_4), self.pool_w(x_4)
        x_pool_4 = torch.matmul(x_ph_4, x_pw_4)


        x_spatial_1 = x_pool_1 * w_spatial
        x_spatial_2 = x_pool_2 * w_spatial
        x_spatial_3 = x_pool_3 * w_spatial
        x_spatial_4 = x_pool_4 * w_spatial
        y_spatial = torch.cat((x_spatial_1, x_spatial_2, x_spatial_3, x_spatial_4), dim=1)

        y_out = y_channel + y_spatial + x
        return y_out


