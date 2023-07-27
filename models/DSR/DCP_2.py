import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.FcaNet.fca_layer import MultiSpectralAttentionLayer
def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = torch.exp(-(x - mean) ** 2 / (2 * std ** 2)) / sigma
    return x_out
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class DCP_2(nn.Module):
    def __init__(self, inplanes = 320):
        super().__init__()
        inplanes_fen = int(inplanes / 4)
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.convp_1 = nn.Conv2d(inplanes_fen, inplanes_fen, kernel_size=3, stride=1, padding=1)
        self.convp_2 = nn.Conv2d(inplanes_fen, inplanes_fen, kernel_size=3 + 1 * 2, stride=1, padding=1 + 1)
        self.convp_3 = nn.Conv2d(inplanes_fen, inplanes_fen, kernel_size=3 + 2 * 2, stride=1, padding=1 + 2)
        self.convp_4 = nn.Conv2d(inplanes_fen, inplanes_fen, kernel_size=3 + 3 * 2, stride=1, padding=1 + 3)

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(inplanes, inplanes // 16),
            nn.ReLU(),
            nn.Linear(inplanes // 16, inplanes)
        )

        c2wh = dict([(64, 56), (128, 28), (320, 14), (512, 7)])
        self.fca = MultiSpectralAttentionLayer(inplanes, c2wh[inplanes], c2wh[inplanes],
                                          reduction=16, freq_sel_method = 'top16')

        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)

    def forward(self, x):
        b, c, h, w = x.size()
        x_in = F.relu(self.bn1(self.conv1(x)), inplace=True)#(8,256,32,32)
        channel_fen = int(c / 4)
        x_1 = x_in[:, :channel_fen, :, :]
        x_2 = x_in[:, channel_fen: 2 * channel_fen, :, :]
        x_3 = x_in[:, 2 * channel_fen: 3 * channel_fen, :, :]
        x_4 = x_in[:, 3 * channel_fen:, :, :]

        x_1 = self.convp_1(x_1)
        x_2 = self.convp_2(x_2)
        x_3 = self.convp_3(x_3)
        x_4 = self.convp_4(x_4)

        avg_pool = F.avg_pool2d(x_in, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        w_fc = self.mlp(avg_pool)
        w_fc = w_fc.unsqueeze(2).unsqueeze(3)

        # f_fca, w_fc = self.fca(x_in)
        w_fc_1 = w_fc[:, :channel_fen, :, :]
        w_fc_2 = w_fc[:, channel_fen: 2 * channel_fen, :, :]
        w_fc_3 = w_fc[:, 2 * channel_fen: 3 * channel_fen, :, :]
        w_fc_4 = w_fc[:, 3 * channel_fen:, :, :]

        # Gauss modulation
        mean_1 = torch.mean(w_fc_1).detach()
        std_1 = torch.std(w_fc_2).detach()
        scale_1 = GaussProjection(w_fc_1, mean_1, std_1).expand_as(x_1)

        mean_2 = torch.mean(w_fc_2).detach()
        std_2 = torch.std(w_fc_2).detach()
        scale_2 = GaussProjection(w_fc_2, mean_2, std_2).expand_as(x_2)

        mean_3 = torch.mean(w_fc_3).detach()
        std_3 = torch.std(w_fc_3).detach()
        scale_3 = GaussProjection(w_fc_3, mean_3, std_3).expand_as(x_3)

        mean_4 = torch.mean(w_fc_4).detach()
        std_4 = torch.std(w_fc_4).detach()
        scale_4 = GaussProjection(w_fc_4, mean_4, std_4).expand_as(x_4)

        y1 = x_1 * scale_1
        y2 = x_2 * scale_2
        y3 = x_3 * scale_3
        y4 = x_4 * scale_4
        y_all = torch.cat((y1, y2, y3, y4), dim=1)
        y_out = F.relu(self.bn2(self.conv2(y_all)), inplace=True)  # (8,256,32,32)
        y_out = y_out + x
        return y_out