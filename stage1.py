import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import data.transforms_bbox as Tr
from data.voc import VOC_box
from configs.defaults import _C
# from models.ClsNet import Labeler
# from models.ClsNet_CNN import Labeler
from models.ClsNet_two_mit import Labeler
from utils.optimizer import PolyWarmupAdamW

import wandb
from utils.wandb import init_wandb, wandb_log

def my_collate(batch):
    '''
    This is to assign a batch-wise index for each box.
    '''
    sample = {}
    img = []
    bboxes = []
    bg_mask = []
    batchID_of_box = []
    for batch_id, item in enumerate(batch):
        img.append(item[0])
        bboxes.append(item[1]) 
        bg_mask.append(item[2])
        for _ in range(len(item[1])):
            batchID_of_box += [batch_id]
    sample["img"] = torch.stack(img, dim=0)
    sample["bboxes"] = torch.cat(bboxes, dim=0)
    sample["bg_mask"] = torch.stack(bg_mask, dim=0)[:,None]
    sample["batchID_of_box"] = torch.tensor(batchID_of_box, dtype=torch.long)
    return sample
def _load_pretrained_weights_(model):
    # state_dict = torch.hub.load_state_dict_from_url(model_url, progress=progress)
    state_dict = torch.load("./weights/mit_b2.pth", map_location='cpu')
    # state_dict = torch.hub.load_state_dict_from_url(model_url, progress=progress)
    del_keys = ['head.weight', 'head.bias']
    for k in del_keys:
        del state_dict[k]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('decode_head'):
            if k.endswith('.proj.weight'):
                k = k.replace('.proj.weight', '.weight')
                v = v[..., None, None]
            elif k.endswith('.proj.bias'):
                k = k.replace('.proj.bias', '.bias')
            elif '.linear_fuse.conv.' in k:
                k = k.replace('.linear_fuse.conv.', '.linear_fuse.')
            elif '.linear_fuse.bn.' in k:
                k = k.replace('.linear_fuse.bn.', '.bn.')

            if '.linear_c4.' in k:
                k = k.replace('.linear_c4.', '.layers.0.')
            elif '.linear_c3.' in k:
                k = k.replace('.linear_c3.', '.layers.1.')
            elif '.linear_c2.' in k:
                k = k.replace('.linear_c2.', '.layers.2.')
            elif '.linear_c1.' in k:
                k = k.replace('.linear_c1.', '.layers.3.')
        else:
            if 'patch_embed1.' in k:
                k = k.replace('patch_embed1.', 'stages.0.patch_embed.')
            elif 'patch_embed2.' in k:
                k = k.replace('patch_embed2.', 'stages.1.patch_embed.')
            elif 'patch_embed3.' in k:
                k = k.replace('patch_embed3.', 'stages.2.patch_embed.')
            elif 'patch_embed4.' in k:
                k = k.replace('patch_embed4.', 'stages.3.patch_embed.')
            elif 'block1.' in k:
                k = k.replace('block1.', 'stages.0.blocks.')
            elif 'block2.' in k:
                k = k.replace('block2.', 'stages.1.blocks.')
            elif 'block3.' in k:
                k = k.replace('block3.', 'stages.2.blocks.')
            elif 'block4.' in k:
                k = k.replace('block4.', 'stages.3.blocks.')
            elif 'norm1.' in k:
                k = k.replace('norm1.', 'stages.0.norm.')
            elif 'norm2.' in k:
                k = k.replace('norm2.', 'stages.1.norm.')
            elif 'norm3.' in k:
                k = k.replace('norm3.', 'stages.2.norm.')
            elif 'norm4.' in k:
                k = k.replace('norm4.', 'stages.3.norm.')

            if '.mlp.dwconv.dwconv.' in k:
                k = k.replace('.mlp.dwconv.dwconv.', '.mlp.conv.')

            if '.mlp.' in k:
                k = k.replace('.mlp.', '.ffn.')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)

def main(cfg):
    """
    Main function

    Create dataloaders, train the model, and save the trained model.

    Inputs:
    - cfg: config file

    Outputs:
    - Trained model saved locally and on wandb
    """
    if cfg.SEED:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)

    tr_transforms = Tr.Compose([
        Tr.RandomScale(0.5, 1.5),
        Tr.ResizeRandomCrop(cfg.DATA.CROP_SIZE), 
        Tr.RandomHFlip(0.5), 
        Tr.ColorJitter(0.5,0.5,0.5,0),
        Tr.Normalize_Caffe(),
        # Tr.normalize(),
    ])
    trainset = VOC_box(cfg, tr_transforms)
    train_loader = DataLoader(trainset, batch_size=cfg.DATA.BATCH_SIZE, collate_fn=my_collate, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    model = Labeler(cfg.DATA.NUM_CLASSES, cfg.MODEL.ROI_SIZE, cfg.MODEL.GRID_SIZE).cuda()
    #gaidong1
    # model.backbone.load_state_dict(torch.load(f"./weights/{cfg.MODEL.WEIGHTS}"), strict=False)
    #MIT weights
    # _load_pretrained_weights_(model.backbone)

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
        warmup_iter=2000,
        max_iter=24000,
        warmup_ratio=1e-6,
        power=1.0
    )

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Initializing W&B
    init_wandb(cfg)
    
    model.train()
    iterator = iter(train_loader)
    storages = {"CE": 0,}
    interval_verbose = cfg.SOLVER.MAX_ITER // 40

    for it in range(1, cfg.SOLVER.MAX_ITER+1):

        try:
            sample = next(iterator)
        except:
            iterator = iter(train_loader)
            sample = next(iterator)

        img = sample["img"]
        bboxes = sample["bboxes"]
        bg_mask = sample["bg_mask"]
        batchID_of_box = sample["batchID_of_box"]
        ind_valid_bg_mask = bg_mask.mean(dim=(1,2,3)) > 0.125 # This is because VGG16 has output stride of 8.

        logits, logits_2, cam, cam2 = model(img.cuda(), bboxes, batchID_of_box, bg_mask.cuda(), ind_valid_bg_mask, GAP=cfg.MODEL.GAP)#(150,2,1,1)
        #, logits_2, cam, cam2
        #logits = model(img.cuda(), bboxes, batchID_of_box, bg_mask.cuda(), ind_valid_bg_mask,
        #                                    GAP=cfg.MODEL.GAP)  # (150,2,1,1)
        logits = logits[...,0,0]#(150,2)
        logits_2 = logits_2[..., 0, 0]  # (150,2)
        fg_t = bboxes[:,-1][:,None].expand(bboxes.shape[0], np.prod(cfg.MODEL.ROI_SIZE))#(9,4)
        fg_t = fg_t.flatten().long()#36
        target = torch.zeros(logits.shape[0], dtype=torch.long)#(150)
        target[:fg_t.shape[0]] = fg_t#(150)

        loss_1 = criterion(logits, target.cuda())
        loss_2 = criterion(logits_2, target.cuda())
        loss_cps = torch.mean(torch.abs(cam[1:,:,:] - cam2[1:,:,:]))
        loss = 0.5 * loss_1 + 0.5 * loss_2 + 0.5 * loss_cps#0.05
        # loss = loss_1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        storages["CE"] += loss.item()

        if it % interval_verbose == 0:
            for k in storages.keys(): storages[k] /= interval_verbose
            for k in storages.keys(): storages[k] = 0

        # Logging on W&B
        wandb_log(loss.item(), optimizer.param_groups[0]["lr"], it)

    torch.save(model.state_dict(), f"./weights/{cfg.NAME}.pt")
    wandb.save(f"./weights/{cfg.NAME}.pt")

    wandb.finish()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./configs/stage1.yml")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    main(cfg)