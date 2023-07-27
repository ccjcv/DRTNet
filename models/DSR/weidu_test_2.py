import torch
import torch.nn as nn
from models.DSR.DCP_1 import DCP_1
from models.FcaNet.fca_layer import MultiSpectralAttentionLayer

c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
planes = 320
model = DCP_1(inplanes=320)
# model = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes],
#                                   reduction=16, freq_sel_method = 'top16')

img = torch.randn(8, 320, 32, 32)
out = model(img)
print(out.size())