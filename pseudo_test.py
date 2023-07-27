import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import data.transforms_seg as Trs
from configs.defaults import _C
from data.voc import VOC_seg, CLASSES
from utils.wandb import init_wandb, wandb_log_seg
import torch.nn.functional as F
from utils.metric import scores
from medpy.metric import binary
import surface_distance as surfdist
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import roc_curve, auc

class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # if not np.any(x):
        #     x[0][0] = 1.0
        # elif not np.any(y):
        #     y[0][0] = 1.0

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.percentile(distances[indexes], 95))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred = pred.unsqueeze(1).byte()
        target = target.unsqueeze(1).byte()


        # pred = (pred > 0.5).byte()
        # target = (target > 0.5).byte()
        if torch.sum(pred) == 0:
            pred[0][0][0][0] = 1
            # print(pred)
            # print(torch.sum(pred))
        # print(pred.shape)
        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
            ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
            ).float()

        # print(right_hd, ' ', left_hd)

        return torch.max(right_hd, left_hd)

hd_metric = HausdorffDistance()
def val(cfg, data_loader):
    label_trues = []
    label_preds = []
    hd95 = 0
    asd = 0
    pre = 0
    rec = 0
    hd_2 = 0
    auc_roc = 0
    sen = 0

    Recall_trfe = 0
    Specificity_trfe = 0
    Precision_trfe = 0
    F1_trfe = 0
    accuracy_trfe = 0
    for batch in tqdm(data_loader):
        img, masks, fn = batch
        ygt_ori = masks[0]
        ycrf_ori = masks[1]
        ygt = ygt_ori.cpu().detach().numpy().astype(int)
        ycrf = ycrf_ori.cpu().detach().numpy().astype(int)
        label_preds.append(ycrf)
        label_trues.append(ygt)

        # roc
        pred_binary = (ycrf >= 0.5)
        pred_binary_inverse = (pred_binary == 0)
        gt_binary = (ygt >= 0.5)
        gt_binary_inverse = (gt_binary == 0)
        TP = (np.multiply(pred_binary, gt_binary)).sum()
        FP = (np.multiply(pred_binary, gt_binary_inverse)).sum()
        TN = (np.multiply(pred_binary_inverse, gt_binary_inverse)).sum()
        FN = (np.multiply(pred_binary_inverse, gt_binary)).sum()
        if TP.item() == 0:
            TP = 1
        Recall_trfe += TP / (TP + FN)
        Specificity_trfe += TN / (TN + FP)
        Precision_trfe += TP / (TP + FP)

        accuracy_trfe += (TP + TN) / (TP + FP + FN + TN)
        fpr, tpr, threshold = roc_curve(gt_binary.flatten(), pred_binary.flatten())
        auc_roc += auc(fpr, tpr)

        if ycrf.sum() > 0:
            hd95 += binary.hd95(ycrf, ygt)
            asd += binary.asd(ycrf, ygt)


        else:
            hd95 += 0
            asd += 0
        pre += binary.precision(ycrf, ygt)
        rec += binary.recall(ycrf, ygt)
        sen += binary.sensitivity(ycrf, ygt)
        hd_2 += hd_metric.compute(ycrf_ori, ygt_ori)

    # Calculate final results
    results = scores(label_trues, label_preds, cfg.DATA.NUM_CLASSES)
    accuracy = results["Mean Accuracy"]
    iou = results["Mean IoU"]
    cls_iou = results["Class IoU"]
    dsc = results["dsc"]
    HD95 = hd95 / len(data_loader)
    hd_2 = hd_2 / len(data_loader)
    ASD = asd / len(data_loader)
    precision = pre / len(data_loader)
    recall = rec / len(data_loader)
    sensitivity = sen / len(data_loader)
    p_acc = results["Pixel Accuracy"]
    F1 = 2 * precision * recall / (precision + recall)
    auc_out = auc_roc / len(data_loader)

    Recall_trfe_out = Recall_trfe / len(data_loader)
    Specificity_trfe_out = Specificity_trfe / len(data_loader)
    Precision_trfe_out = Precision_trfe / len(data_loader)
    F1_trfe_out = 2 * Precision_trfe_out * Recall_trfe_out / (Precision_trfe_out + Recall_trfe_out)
    accuracy_trfe_out = accuracy_trfe / len(data_loader)

    print("CRF Validation Mean Accuracy ", accuracy)
    print("CRF Validation Mean IoU ", iou)
    print("CRF Validation cls IoU(decide) ", cls_iou)
    print("CRF Validation dsc(decide) ", dsc)
    print("CRF Validation hd95(decide) ", HD95)
    print("CRF Validation ASD(decide) ", ASD)
    print("CRF Validation precision ", precision)
    print("CRF Validation recall ", recall)
    print("CRF Validation sensitivity ", sensitivity)
    print("CRF Validation pixel accuracy ", p_acc)
    print("CRF Validation F1 ", F1)
    print("CRF Validation auc ", auc_out)
    print("recall_trfe(decide) ", Recall_trfe_out)
    print("Spe_trfe ", Specificity_trfe_out)
    print("Pre_trfe ", Precision_trfe_out)
    print("F1_trfe(decide) ", F1_trfe_out)
    print("acc_trfe(decide) ", accuracy_trfe_out)
def main(cfg):
    if cfg.SEED:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)

    if cfg.WANDB.MODE:
        init_wandb(cfg)

    if cfg.DATA.MODE == "train_weak":
        tr_transforms = Trs.Compose([
            Trs.RandomScale(0.5, 1.5),
            Trs.ResizeRandomCrop(cfg.DATA.CROP_SIZE),
            Trs.RandomHFlip(0.5),
            Trs.ColorJitter(0.5, 0.5, 0.5, 0),
            Trs.Normalize_Caffe(),
        ])
    elif cfg.DATA.MODE == "val":
        tr_transforms = Trs.Compose([
            # Trs.FixedResize(256),
            Trs.Normalize_Caffe(),
        ])
    else:
        print("Incorrect Mode provided!")
        return

    dataset = VOC_seg(cfg, tr_transforms)
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=True)
    val(cfg, data_loader)

def get_args():
    """
    Get the arguments from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./configs/stage3_vgg.yml")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    main(cfg)