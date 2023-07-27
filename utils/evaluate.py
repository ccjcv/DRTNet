from utils.metric import Evaluator

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def evaluate(cfg, data_loader, model):
    """
    Evaluation function

    Evaluate the model using the given dataloader.

    Inputs:
    - cfg: config file
    - data_loader: dataloader
    - model: model

    Outputs:
    - Mean Accuracy and IoU
    """
    with torch.no_grad():
        model.eval()
        evaluator = Evaluator(cfg.DATA.NUM_CLASSES)
        evaluator.reset()
        for batch in tqdm(data_loader):
            #img, masks, fn, kuang = batch
            img, masks, fn = batch
            ygt = masks[0]
            # Forward pass
            img = img.to('cuda')
            img_size = img.size()
            # logit, feature_map = model(img, (img_size[2], img_size[3]))
            #logit, logit_small, logit_large, weight_ori, weight_small, weight_large = model(img)

            # logit, bcm = model(img)
            logit = model(img)
            log_ = F.interpolate(logit, (img_size[2], img_size[3]), mode='bilinear', align_corners=False)
            pred = torch.argmax(log_, dim=1)
            pred = pred.cpu().detach().numpy()
            ygt = ygt.cpu().detach().numpy()
            evaluator.add_batch(ygt, pred)
        # Calculate final metrics
        accuracy = evaluator.MACU()
        iou = evaluator.MIOU()
        cls_iou = evaluator.CLS_IOU()
        dsc = evaluator.dsc()
        return accuracy, iou, cls_iou, dsc