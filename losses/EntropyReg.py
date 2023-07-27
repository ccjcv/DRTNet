import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class EntropyRegularizationLoss(nn.Module):
    
    def __init__(self, num_classes, lambda_wgt):
        super(EntropyRegularizationLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_wgt = lambda_wgt
        self.soft_max = nn.Softmax(dim=1)
        
    def forward(self, y_pred, ycrf, yret):
        y_pred = self.soft_max(y_pred)
        loss_ce = self.get_loss_ce(y_pred, ycrf, yret)
        loss_er = self.get_loss_er(y_pred, ycrf, yret)
        loss = loss_ce + self.lambda_wgt * loss_er
        return loss
    
    def get_loss_ce(self, y_pred, ycrf, yret):
        n_classes_arr=torch.arange(self.num_classes).cuda()
        
        s_class = (ycrf[:,:,:,None] == n_classes_arr) & (yret[:,:,:,None] == n_classes_arr)
        s_class = torch.permute(s_class, (0, 3, 1, 2)) 
        
        denom = torch.sum(s_class)
        num = torch.sum(torch.log(y_pred[s_class]))
        return -num/denom 
    
    def get_loss_er(self, y_pred, ycrf, yret):
        n_classes_arr=torch.arange(self.num_classes).cuda()
        
        not_s_class = torch.logical_not((ycrf[:,:,:,None] == n_classes_arr) & (yret[:,:,:,None] == n_classes_arr))  
        not_s_class = torch.permute(not_s_class, (0, 3, 1, 2)) 
        
        denom = torch.sum(not_s_class)
        numer = 0
        
        for i in range(self.num_classes):
            t = not_s_class[:,i,:,:]
            numer += torch.sum(y_pred[:,i,:,:][t] * torch.log(y_pred[:,i,:,:][t]))
            
        return -numer/denom