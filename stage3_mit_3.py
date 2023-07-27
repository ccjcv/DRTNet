import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import data.transforms_seg as Trs
from configs.defaults import _C
from data.voc import VOC_seg, CLASSES
from losses.Baseline import BaselineLoss
from losses.Bootstraping import BootstrapingLoss
from losses.EntropyReg import EntropyRegularizationLoss
from losses.NAL import NoiseAwareLoss
from models.PolyScheduler import PolynomialLR
from utils.densecrf import dense_crf
from utils.evaluate import evaluate
from utils.wandb import init_wandb, wandb_log_seg
import torch.nn.functional as F
# from losses.JSD_loss import calc_jsd_multiscale as calc_jsd_temp
# from losses.JSD_loss_2 import calc_jsd_multiscale_2 as calc_jsd_temp
from losses.JSD_loss_urpc_2 import calc_jsd_multiscale_2 as calc_jsd_temp
from utils.optimizer import PolyWarmupAdamW
from segformer.three_model import three_model

def train(cfg, train_loader, model, checkpoint):
    """
    Training function

    Train the model using the training dataloader and the last saved checkpoint.

    Inputs:
    - cfg: config file
    - train_loader: training dataloader
    - model: model
    - checkpoint: state dict of the model, optimizer, scheduler, etc.

    Outputs:
    - Trained model saved locally and on wandb
    """
    model = model.cuda()
    if cfg.MODEL.LOSS == "NAL":
        criterion = NoiseAwareLoss(cfg.DATA.NUM_CLASSES,
                                   cfg.MODEL.DAMP,
                                   cfg.MODEL.LAMBDA)
    elif cfg.MODEL.LOSS == "ER":
        criterion = EntropyRegularizationLoss(cfg.DATA.NUM_CLASSES,
                                              cfg.MODEL.LAMBDA)

    elif cfg.MODEL.LOSS == "BS":
        criterion = BootstrapingLoss(cfg.DATA.NUM_CLASSES,
                                     cfg.MODEL.LAMBDA)
        beta = 1.0

    elif cfg.MODEL.LOSS == "BASELINE":
        criterion = BaselineLoss(cfg.DATA.NUM_CLASSES)

    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)

    # lr = cfg.SOLVER.LR
    # wd = cfg.SOLVER.WEIGHT_DECAY

    # Creating optimizer for the both models
    # if cfg.NAME == "SegNet_VGG":
    #     params = model.get_params()
    #     optimizer = optim.SGD(
    #         [{"params": params[0], "lr": lr, "weight_decay": wd},
    #          {"params": params[1], "lr": lr, "weight_decay": 0.0},
    #          {"params": params[2], "lr": 10 * lr, "weight_decay": wd},
    #          {"params": params[3], "lr": 10 * lr, "weight_decay": 0.0}],
    #         lr=lr,
    #         weight_decay=wd,
    #         momentum=cfg.SOLVER.MOMENTUM
    #     )
    # elif cfg.NAME == "SegNet_ASPP":
    #     optimizer = optim.SGD(
    #         params=[
    #             {
    #                 "params": model.get_1x_lr_params(),
    #                 "lr": lr,
    #                 "weight_decay": wd
    #             },
    #             {
    #                 "params": model.get_10x_lr_params(),
    #                 "lr": 10 * lr,
    #                 "weight_decay": wd
    #             }
    #         ],
    #         lr=lr,
    #         weight_decay=wd,
    #         momentum=cfg.SOLVER.MOMENTUM
    #     )

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": params_to_optimize,
                "lr": 0.0001,
                "weight_decay": 0.0001,
            },
        ],
        lr=0.0001,
        weight_decay=0.0001,
        betas=[0.9, 0.999],
        warmup_iter=1500,
        max_iter=23805,#23805
        warmup_ratio=1e-6,
        power=1.0
    )


    # optimizer = torch.optim.Adam(params_to_optimize, lr=0.0001, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES, gamma=0.1)
    # Poly learning rate scheduler according to the paper
    # scheduler = PolynomialLR(optimizer,
    #                          step_size=cfg.SOLVER.STEP_SIZE,
    #                          iter_max=cfg.SOLVER.MAX_ITER,
    #                          power=cfg.SOLVER.GAMMA)
    curr_it = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        # scheduler.load_state_dict(checkpoint['sched_state_dict'])
        curr_it = checkpoint['iter']
    # weight = nn.Parameter(torch.Tensor(3))
    # weight.data.fill_(1)
    # weight.to('cuda')
    iterator = iter(train_loader)

    for it in tqdm(range(curr_it + 1, cfg.SOLVER.MAX_ITER + 1)):
        try:
            sample = next(iterator)
        except:
            iterator = iter(train_loader)
            sample = next(iterator)
        img, masks, fn = sample  # VOC_seg dataloader returns image and the corresponing (pseudo) label
        # img_small = F.interpolate(img, scale_factor=args.scale_factor, mode='bilinear',
        #                           align_corners=True,
        #                           recompute_scale_factor=True)
        #
        # img_large = F.interpolate(img, scale_factor=args.scale_factor2, mode='bilinear',
        #                           align_corners=True,
        #                           recompute_scale_factor=True)
        ygt, ycrf, yret = masks

        model.train()

        # Forward pass
        img = img.to('cuda')
        # img_small = img_small.to('cuda')
        # img_large = img_large.to('cuda')        # img_size = img.size()
        # img_size_small = img_small.size()
        # img_size_large = img_large.size()
        # 普通的就用到logit
        logit, logit_small, logit_large, weight_ori, weight_small, weight_large = model(img)
        # logit_small, feature_map_small = model(img_small, (img_size_small[2], img_size_small[3]))
        # logit_small = F.interpolate(logit_small, size=(img_size[2], img_size[3]), mode='bilinear', align_corners=True)
        # logit_large, feature_map_large = model(img_large, (img_size_large[2], img_size_large[3]))
        # logit_large = F.interpolate(logit_large, size=(img_size[2], img_size[3]), mode='bilinear', align_corners=True)

        # weight = nn.Parameter(torch.Tensor(3))
        # weight.data.fill_(1)
        # weight.to('cuda')
        # weight_2 zishiying

        # Loss calculation
        # if cfg.MODEL.LOSS == "NAL":
        #     ycrf = ycrf.cuda().long()
        #     yret = yret.cuda().long()
        #     classifier_weight = torch.clone(model.classifier.weight.data)
        #     loss = criterion(logit,
        #                      ycrf,
        #                      yret,
        #                      feature_map,
        #                      classifier_weight)

        if cfg.MODEL.LOSS == "BS":
            ycrf = ycrf.cuda().long()
            yret = yret.cuda().long()
            loss = criterion(logit,
                             ycrf,
                             yret,
                             beta)
            beta = beta * ((1 - float(it) / 26000) ** 0.9)

        elif cfg.MODEL.LOSS == "ER" or cfg.MODEL.LOSS == "BASELINE":
            ycrf = ycrf.cuda().long()
            yret = yret.cuda().long()
            loss = criterion(logit,
                             ycrf,
                             yret)

        elif cfg.MODEL.LOSS == "CE_CRF":
            ycrf = ycrf.cuda().long()
            loss = criterion(logit, ycrf)
        elif cfg.MODEL.LOSS == "CE_RET":
            yret = yret.cuda().long()
            loss = criterion(logit, yret)
        elif cfg.MODEL.LOSS == "consistency loss":
            # loss_ce, consistency_loss, mixture_label = calc_jsd_temp(
            #     ycrf.cuda().long(), logit, logit_small, logit_large, weight.cuda(), threshold=0.9)
            loss_ce, consistency_loss, mixture_label = calc_jsd_temp(ycrf.cuda().long(), logit, logit_small,
                                                                     logit_large, weight_ori, weight_small,
                                                                     weight_large, threshold=0.9)
            loss = loss_ce + 1 * consistency_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Update the learning rate using poly scheduler
        # scheduler.step()
        train_loss_ce = loss_ce.item()
        train_loss_cons = consistency_loss.item()
        train_loss = loss.item()
        # Logging Loss and LR on wandb
        wandb_log_seg(train_loss, train_loss_ce, train_loss_cons, optimizer.param_groups[0]["lr"], it)

        save_dir = "./ckpts/"
        if it % 1000 == 0 or it == cfg.SOLVER.MAX_ITER:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                # 'sched_state_dict': scheduler.state_dict(),
                'iter': it,
            }
            torch.save(checkpoint, save_dir + str(it) + '.pth')
            if cfg.WANDB.MODE:
                # if it == 22000 or it == cfg.SOLVER.MAX_ITER:
                if it == cfg.SOLVER.MAX_ITER:
                    wandb.save(save_dir + str(it) + '.pth')
            # Evaluate the train_loader at this checkpoint and Log the metrics on WandB
            accuracy, iou, cls_iou, dsc = evaluate(cfg, train_loader, model)
            if cfg.WANDB.MODE:
                # Log on WandB
                wandb.log({
                    "Mean IoU": iou,
                    "Mean Accuracy": accuracy,
                    "cls IoU 0": cls_iou[0],
                    "dsc 0": dsc[0],
                    "cls IoU 1": cls_iou[1],
                    "dsc 1": dsc[1]
                })


def val(cfg, data_loader, model, checkpoint):
    """
    Validation function

    Evaluate the model using the validation dataloader and the last saved checkpoint.
    And apply the CRF post-processing to the predicted masks.

    Inputs:
    - cfg: config file
    - data_loader: validation dataloader
    - model: model
    - checkpoint: state dict of the model, optimizer, scheduler, etc.

    Outputs:
    - Validation metrics saved locally and on wandb
    """
    model = model.cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    accuracy, iou, cls_iou, dsc = evaluate(cfg, data_loader, model)
    print("Validation Mean Accuracy ", accuracy)
    print("Validation Mean IoU ", iou)
    print("Validation Cls IoU ", cls_iou)
    print("Validation dsc ", dsc)
    wandb.run.summary["Validation Mean Accuracy"] = accuracy
    wandb.run.summary["Validation Mean IoU"] = iou
    # Evaluating the validation dataloader after CRF post-processing
    crf_accuracy, crf_iou, crf_cls_iou, dsc, HD95, ASD, precision, recall, p_acc = dense_crf(cfg, data_loader, model)
    wandb.run.summary["CRF Validation Mean Accuracy"] = crf_accuracy
    wandb.run.summary["CRF Validation Mean IoU"] = crf_iou
    print("CRF Validation Mean Accuracy ", crf_accuracy)
    print("CRF Validation Mean IoU ", crf_iou)
    print("CRF Validation cls IoU ", crf_cls_iou)
    print("CRF Validation dsc ", dsc)
    print("CRF Validation hd95 ", HD95)
    print("CRF Validation ASD ", ASD)
    print("CRF Validation precision ", precision)
    print("CRF Validation recall ", recall)
    print("CRF Validation pixel accuracy ", p_acc)





def main(cfg):
    """
    Main function

    Call the train and val functions according to the mode.

    Inputs:
    - cfg: config file

    Outputs:

    """
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
            Trs.Normalize_Caffe(),
        ])
    else:
        print("Incorrect Mode provided!")
        return

    dataset = VOC_seg(cfg, tr_transforms)
    if cfg.DATA.MODE == "train_weak":
        data_loader = DataLoader(dataset,
                                 batch_size=cfg.DATA.BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=4,
                                 pin_memory=True,
                                 drop_last=True)

    elif cfg.DATA.MODE == "val":
        data_loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=4,
                                 pin_memory=True,
                                 drop_last=True)

    # model = segformer_b2(pretrained=True, num_classes=cfg.DATA.NUM_CLASSES)
    model = three_model()

    # Save model locally and then on wandb
    save_dir = './ckpts/'
    # Load pretrained model from wandb if present
    if cfg.WANDB.CHECKPOINT:
        wandb_checkpoint = wandb.restore('ckpts/' + cfg.WANDB.CHECKPOINT)
        print(wandb_checkpoint)
        checkpoint = torch.load(wandb_checkpoint.name)
        print("WandB checkpoint Loaded with iteration: ", checkpoint['iter'])
    else:
        print("WandB checkpoint not Loaded")
        checkpoint = None
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Call the appropriate mode from main()
    if cfg.DATA.MODE == "train_weak":
        train(cfg, data_loader, model, checkpoint)
    elif cfg.DATA.MODE == "val":
        val(cfg, data_loader, model, checkpoint)


def get_args():
    """
    Get the arguments from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./configs/stage3_vgg.yml")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    parser.add_argument("--scale_factor", type=float, default=0.7,
                        help="scale_factor of downsample the image")
    parser.add_argument("--scale_factor2", type=float, default=1.5,
                        help="scale_factor of upsample the image")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    main(cfg)