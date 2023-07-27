import torch
import torch.nn as nn
from models.FcaNet.fca_layer import MultiSpectralAttentionLayer

c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
planes = 256
att = MultiSpectralAttentionLayer(planes * 4, c2wh[planes], c2wh[planes],
                                  reduction=16, freq_sel_method = 'top16')

img = torch.randn(8, 1024, 16, 16)
out = att(img)
print(out.size())