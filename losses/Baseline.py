import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class BaselineLoss(nn.Module):
    
    def __init__(self, num_classes):
        super(BaselineLoss, self).__init__()
        self.num_classes = num_classes
        self.soft_max = nn.Softmax(dim=1)
        
    def forward(self, y_pred, ycrf, yret):
        y_pred = self.soft_max(y_pred)
        loss_ce = self.get_loss_ce(y_pred, ycrf, yret)
        return loss_ce
    
    def get_loss_ce(self, y_pred, ycrf, yret):
        n_classes_arr=torch.arange(self.num_classes).cuda()
        
        # s_class = (ycrf[:,:,:,None] == n_classes_arr) & (yret[:,:,:,None] == n_classes_arr)
        s_class = (ycrf[:, :, :, None] == n_classes_arr)
        s_class = torch.permute(s_class, (0, 3, 1, 2)) 
        
        denom = torch.sum(s_class)
        num = torch.sum(torch.log(y_pred[s_class]))
        return -num/denom 