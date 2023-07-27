from segformer.model import segformer_b2
import torch
from torch import nn
import torch.nn.functional as F
class three_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = segformer_b2(pretrained=True, num_classes=2)
        # self.weight_level_1024_9 = nn.Conv2d(768, 3, 1)
        self.weight_level_9_3 = nn.Conv2d(6, 3, 1)



    def forward(self, x):
        img_size = x.size()
        img_small = F.interpolate(x, scale_factor=0.7, mode='bilinear',
                                  align_corners=True,
                                  recompute_scale_factor=True)

        img_large = F.interpolate(x, scale_factor=1.5, mode='bilinear',
                                  align_corners=True,
                                  recompute_scale_factor=True)
        logit_ori = self.model(x)
        logit_small = self.model(img_small)
        logit_small = F.interpolate(logit_small, size=(img_size[2], img_size[3]), mode='bilinear', align_corners=True)
        logit_large = self.model(img_large)
        logit_large = F.interpolate(logit_large, size=(img_size[2], img_size[3]), mode='bilinear', align_corners=True)
        #然后对这三个特征计算每个尺度上的权重 (8, 1024, 256, 256)
        # weight_ori_9 = self.weight_level_1024_9(weight_x_ori)
        # weight_ori_9 = F.interpolate(weight_ori_9, size=(img_size[2], img_size[3]), mode='bilinear', align_corners=True)
        # weight_small_9 = self.weight_level_1024_9(weight_x_small)
        # weight_small_9 = F.interpolate(weight_small_9, size=(img_size[2], img_size[3]), mode='bilinear', align_corners=True)
        # weight_large_9 = self.weight_level_1024_9(weight_x_large)
        # weight_large_9 = F.interpolate(weight_large_9, size=(img_size[2], img_size[3]), mode='bilinear', align_corners=True)
        # weight_jia_9 = torch.cat((weight_ori_9, weight_small_9, weight_large_9), 1)
        weight_jia_9 = torch.cat((logit_ori, logit_small, logit_large), 1)
        # level_weight_3 = self.weight_level_9_3(weight_jia_9)
        level_weight_3 = self.weight_level_9_3(weight_jia_9)
        level_weight_3 = F.softmax(level_weight_3, dim=1)
        weight_ori = level_weight_3[:, 0:1, :, :]
        weight_small = level_weight_3[:, 1:2, :, :]
        weight_large = level_weight_3[:, 2:3, :, :]
        return logit_ori, logit_small, logit_large, weight_ori, weight_small, weight_large

