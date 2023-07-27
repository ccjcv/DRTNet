import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from models.DSR.DCP_1 import DCP_1
# def GaussProjection(x, mean, std):
#     sigma = math.sqrt(2 * math.pi) * std
#     x_out = torch.exp(-(x - mean) ** 2 / (2 * std ** 2)) / sigma
#     return x_out


class AMM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(AMM, self).__init__()
        # self.ChannelAMM = ChannelGate_fca(gate_channels, reduction_ratio)
        # self.SpatialAMM = SpatialGate()
        self.channelAMM_dsr = DCP_1(gate_channels)

    def forward(self, x):
        # x_out = self.ChannelAMM(x)
        # x_out = self.SpatialAMM(x_out)
        x_out = self.channelAMM_dsr(x)
        return x_out