import os
import wandb
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

import data.transforms_bbox as Tr
from data.voc import VOC_box
from configs.defaults import _C
from models.ClsNet import Labeler, pad_for_grid
# from models.ClsNet_mit import Labeler, pad_for_grid
from utils.densecrf import DENSE_CRF
from pytorch_grad_cam.utils.image import show_cam_on_image


def main(cfg):
    if cfg.SEED:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)

    tr_transforms = Tr.Normalize_Caffe()
    # tr_transforms = Tr.normalize()
    trainset = VOC_box(cfg, tr_transforms)
    train_loader = DataLoader(trainset, batch_size=1)
    
    model = Labeler(cfg.DATA.NUM_CLASSES, cfg.MODEL.ROI_SIZE, cfg.MODEL.GRID_SIZE).cuda()

    # Restore the model saved on WandB
    model_stage_1 = wandb.restore(cfg.WANDB.RESTORE_NAME, run_path=cfg.WANDB.RESTORE_RUN_PATH)
    model.load_state_dict(torch.load(model_stage_1.name))

    WEIGHTS = torch.clone(model.classifier.weight.data)
    model.eval()
    
    bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std = cfg.MODEL.DCRF
    dCRF = DENSE_CRF(bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)
    
    if cfg.SAVE_PSEUDO_LABLES:
        folder_name = os.path.join(cfg.DATA.ROOT, cfg.NAME)
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        save_paths = []
        for txt in ("Y_crf", "Y_ret", "Y_crf_u0", "Y_fg"):
            sub_folder = folder_name + f"/{txt}"
            if not os.path.isdir(sub_folder):
                os.mkdir(sub_folder)
            save_paths += [os.path.join(sub_folder, "{}.png")]
            
    with torch.no_grad():
        for it, (img, bboxes, bg_mask) in enumerate(tqdm(train_loader)):
            '''
            img     : (1,3,H,W) float32
            bboxes  : (1,K,5)   float32
            bg_mask : (1,H,W)   float32
            '''
            fn = trainset.filenames[it]

            rgb_img = np.array(Image.open(trainset.img_path.format(fn))) # RGB input image
            bboxes = bboxes[0] # (1,K,5) --> (K,5) bounding boxes
            bg_mask = bg_mask[None] # (1,H,W) --> (1,1,H,W) background mask

            img_H, img_W = img.shape[-2:]
            norm_H, norm_W = (img_H-1)/2, (img_W-1)/2
            bboxes[:,[0,2]] = bboxes[:,[0,2]]*norm_W + norm_W
            bboxes[:,[1,3]] = bboxes[:,[1,3]]*norm_H + norm_H
            bboxes = bboxes.long()
            gt_labels = bboxes[:,4].unique()

            
            features = model.get_features(img.cuda()) # Output from the model backbone
            # f_1, f_2, f_3 = model.get_features(img.cuda())  # Output from the model backbone
            # features = f_2


            features = F.interpolate(features, img.shape[-2:], mode='bilinear', align_corners=True)
            padded_features = pad_for_grid(features, cfg.MODEL.GRID_SIZE)
            padded_bg_mask = pad_for_grid(bg_mask.cuda(), cfg.MODEL.GRID_SIZE)
            grid_bg, valid_gridIDs = model.get_grid_bg_and_IDs(padded_bg_mask, cfg.MODEL.GRID_SIZE)
            bg_protos = model.get_bg_prototypes(padded_features, padded_bg_mask, grid_bg, cfg.MODEL.GRID_SIZE)
            bg_protos = bg_protos[0,valid_gridIDs] # (1,GS**2,dims,1,1) --> (len(valid_gridIDs),dims,1,1)
            normed_bg_p = F.normalize(bg_protos)
            normed_f = F.normalize(features)

            # Background attention maps (u0)
            bg_attns = F.relu(torch.sum(normed_bg_p*normed_f, dim=1))
            bg_attn = torch.mean(bg_attns, dim=0, keepdim=True) # (len(valid_gridIDs),H,W) --> (1,H,W)
            bg_attn_test = bg_attn.cpu().numpy()
            bg_attn[bg_attn < cfg.MODEL.BG_THRESHOLD * bg_attn.max()] = 0
            Bg_unary = torch.clone(bg_mask[0]) # (1,H,W)
            region_inside_bboxes = Bg_unary[0]==0 # (H,W)
            Bg_unary[:,region_inside_bboxes] = bg_attn[:,region_inside_bboxes].detach().cpu()
            # Bg_unary_test = Bg_unary.numpy()
            
            # CAMS for foreground classes (uc)
            Fg_unary = []
            for uni_cls in gt_labels:
                w_c = WEIGHTS[uni_cls][None]
                raw_cam = F.relu(torch.sum(w_c*features, dim=1)) # (1,H,W)
                normed_cam = torch.zeros_like(raw_cam)
                for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                    denom = raw_cam[:,hmin:hmax,wmin:wmax].max() + 1e-12
                    normed_cam[:,hmin:hmax,wmin:wmax] = raw_cam[:,hmin:hmax,wmin:wmax] / denom
                Fg_unary += [normed_cam]
            Fg_unary = torch.cat(Fg_unary, dim=0).detach().cpu()
            # Fg_unary_test = Fg_unary.numpy()#you


            # CAMS for background classes (ub)
            w_c_bg = WEIGHTS[0][None]
            raw_cam_bg = F.relu(torch.sum(w_c_bg*features, dim=1)) # (1,H,W)
            normed_cam_bg = raw_cam_bg.clone().detach()
            for uni_cls in gt_labels:
              for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                normed_cam_bg[:,hmin:hmax,wmin:wmax] = 0
            normed_cam_bg = (normed_cam_bg / normed_cam_bg.max()).detach().cpu()
            # normed_cam_bg_test = normed_cam_bg.numpy()#you

            rgb_img_01 = rgb_img / 255
            Y_fg = show_cam_on_image(rgb_img_01, Fg_unary[0, :])
            Y_fg = Y_fg[:, :, ::-1]


            # Final unary by contacinating foreground and background unaries
            unary = torch.cat((Bg_unary, Fg_unary), dim=0)
            # unary_test = unary.numpy()
            unary[:,region_inside_bboxes] = torch.softmax(unary[:,region_inside_bboxes], dim=0)
            unary_test = unary.numpy()
            refined_unary = dCRF.inference(rgb_img, unary.numpy())

            # Unary witout background attn
            unary_u0 = torch.cat((normed_cam_bg, Fg_unary), dim=0)
            unary_u0[:, region_inside_bboxes] = torch.softmax(unary_u0[:, region_inside_bboxes], dim=0)
            refined_unary_u0 = dCRF.inference(rgb_img, unary_u0.numpy())

            
            # (Out of bboxes) reset Fg scores to zero
            for idx_cls, uni_cls in enumerate(gt_labels,1):
                mask = np.zeros((img_H,img_W))
                for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                    mask[hmin:hmax,wmin:wmax] = 1
                refined_unary[idx_cls] *= mask
                refined_unary_u0[idx_cls] *= mask

            # Y_crf and Y_crf_u0
            tmp_mask = refined_unary.argmax(0)
            tmp_mask_u0 = refined_unary_u0.argmax(0)
            Y_crf = np.zeros_like(tmp_mask, dtype=np.uint8)
            Y_crf_u0 = np.zeros_like(tmp_mask_u0, dtype=np.uint8)
            for idx_cls, uni_cls in enumerate(gt_labels,1):
                Y_crf[tmp_mask==idx_cls] = uni_cls
                Y_crf_u0[tmp_mask_u0==idx_cls] = uni_cls
            Y_crf[tmp_mask==0] = 0
            Y_crf_u0[tmp_mask_u0==0] = 0


            # Y_ret
            tmp_Y_crf = torch.from_numpy(Y_crf) # (H,W)
            gt_labels_with_Bg = [0] + gt_labels.tolist()
            corr_maps = []
            for uni_cls in gt_labels_with_Bg:
                indices = tmp_Y_crf==uni_cls
                if indices.sum():
                    normed_p = F.normalize(features[...,indices].mean(dim=-1))   # (1,dims)
                    corr = F.relu((normed_f*normed_p[...,None,None]).sum(dim=1)) # (1,H,W)
                else:
                    normed_w = F.normalize(WEIGHTS[uni_cls][None])
                    corr = F.relu((normed_f*normed_w).sum(dim=1)) # (1,H,W)
                corr_maps.append(corr)
            corr_maps = torch.cat(corr_maps) # (1+len(gt_labels),H,W)
            
            
            # (Out of bboxes) reset Fg correlations to zero
            for idx_cls, uni_cls in enumerate(gt_labels_with_Bg):
                if uni_cls == 0:
                    corr_maps[idx_cls, ~region_inside_bboxes] = 1
                else:
                    mask = torch.zeros(img_H,img_W).type_as(corr_maps)
                    for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                        mask[hmin:hmax,wmin:wmax] = 1
                    corr_maps[idx_cls] *= mask

            tmp_mask = corr_maps.argmax(0).detach().cpu().numpy()
            Y_ret = np.zeros_like(tmp_mask, dtype=np.uint8)
            for idx_cls, uni_cls in enumerate(gt_labels,1):
                Y_ret[tmp_mask==idx_cls] = uni_cls
            Y_ret[tmp_mask==0] = 0
            

            if cfg.SAVE_PSEUDO_LABLES:
                for pseudo, save_path in zip([Y_crf, Y_ret, Y_crf_u0, Y_fg], save_paths):
                    Image.fromarray(pseudo).save(save_path.format(fn))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./configs/stage2.yml")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    main(cfg)