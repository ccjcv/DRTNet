import os

import cv2

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
# from models.ClsNet import Labeler, pad_for_grid
# from models.ClsNet_AMR import Labeler, pad_for_grid
from models.ClsNet_two_mit import Labeler, pad_for_grid
from utils.densecrf import DENSE_CRF
from pytorch_grad_cam.utils.image import show_cam_on_image


def main(cfg):
    if cfg.SEED:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)

    tr_transforms = Tr.Normalize_Caffe()
    trainset = VOC_box(cfg, tr_transforms)
    train_loader = DataLoader(trainset, batch_size=1)

    model = Labeler(cfg.DATA.NUM_CLASSES, cfg.MODEL.ROI_SIZE, cfg.MODEL.GRID_SIZE).cuda()

    # Restore the model saved on WandB
    model_stage_1 = wandb.restore(cfg.WANDB.RESTORE_NAME, run_path=cfg.WANDB.RESTORE_RUN_PATH)
    model.load_state_dict(torch.load(model_stage_1.name))

    WEIGHTS = torch.clone(model.classifier.weight.data)
    WEIGHTS_2 = torch.clone(model.classifier2.weight.data)
    model.eval()

    bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std = cfg.MODEL.DCRF
    dCRF = DENSE_CRF(bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)

    if cfg.SAVE_PSEUDO_LABLES:
        folder_name = os.path.join(cfg.DATA.ROOT, cfg.NAME)
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        save_paths = []
        for txt in ("Y_crf", "Y_ret", "Y_crf_u0", "Y_fg_he", "Y_fg_1", "Y_fg_2",
                    "cam_fg1", "cam_fg2", "cam_fg_he", "cam_bg1", "cam_bg2",
                    "cam_fg1_bg1", "cam_fg2_bg2"):
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

            rgb_img = np.array(Image.open(trainset.img_path.format(fn)))  # RGB input image
            bboxes = bboxes[0]  # (1,K,5) --> (K,5) bounding boxes
            bg_mask = bg_mask[None]  # (1,H,W) --> (1,1,H,W) background mask

            img_H, img_W = img.shape[-2:]
            norm_H, norm_W = (img_H - 1) / 2, (img_W - 1) / 2
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * norm_W + norm_W
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * norm_H + norm_H
            bboxes = bboxes.long()
            gt_labels = bboxes[:, 4].unique()

            f_1, f_2 = model.get_features(img.cuda())  # Output from the model backbone
            #fen 3 lu

            f_1 = F.interpolate(f_1, img.shape[-2:], mode='bilinear', align_corners=True)
            padded_features_1 = pad_for_grid(f_1, cfg.MODEL.GRID_SIZE)
            padded_bg_mask_1 = pad_for_grid(bg_mask.cuda(), cfg.MODEL.GRID_SIZE)
            grid_bg_1, valid_gridIDs_1 = model.get_grid_bg_and_IDs(padded_bg_mask_1, cfg.MODEL.GRID_SIZE)
            bg_protos_1 = model.get_bg_prototypes(padded_features_1, padded_bg_mask_1, grid_bg_1, cfg.MODEL.GRID_SIZE)
            bg_protos_1 = bg_protos_1[0, valid_gridIDs_1]  # (1,GS**2,dims,1,1) --> (len(valid_gridIDs),dims,1,1)
            normed_bg_p_1 = F.normalize(bg_protos_1)
            normed_f_1 = F.normalize(f_1)

            # Background attention maps (u0)
            bg_attns_1 = F.relu(torch.sum(normed_bg_p_1 * normed_f_1, dim=1))
            bg_attn_1 = torch.mean(bg_attns_1, dim=0, keepdim=True)  # (len(valid_gridIDs),H,W) --> (1,H,W)
            # zuidazhi = bg_attn_1.max()
            # bg_attn_test = bg_attn_1.cpu().numpy()
            bg_attn_1[bg_attn_1 < cfg.MODEL.BG_THRESHOLD * bg_attn_1.max()] = 0
            Bg_unary_1 = torch.clone(bg_mask[0])  # (1,H,W)
            region_inside_bboxes_1 = Bg_unary_1[0] == 0  # (H,W)
            Bg_unary_1[:, region_inside_bboxes_1] = bg_attn_1[:, region_inside_bboxes_1].detach().cpu()
            # Bg_unary_test = Bg_unary.numpy()

            # CAMS for foreground classes (uc)
            Fg_unary_1 = []
            for uni_cls in gt_labels:
                w_c = WEIGHTS[uni_cls][None]
                raw_cam_1 = F.relu(torch.sum(w_c * f_1, dim=1))  # (1,H,W)
                normed_cam_1 = torch.zeros_like(raw_cam_1)
                for wmin, hmin, wmax, hmax, _ in bboxes[bboxes[:, 4] == uni_cls]:
                    denom_1 = raw_cam_1[:, hmin:hmax, wmin:wmax].max() + 1e-12
                    normed_cam_1[:, hmin:hmax, wmin:wmax] = raw_cam_1[:, hmin:hmax, wmin:wmax] / denom_1
                Fg_unary_1 += [normed_cam_1]
            Fg_unary_1 = torch.cat(Fg_unary_1, dim=0).detach().cpu()
            # Fg_unary_1_test = Fg_unary_1.numpy()#you

            # CAMS for background classes (ub)
            w_c_bg_1 = WEIGHTS[0][None]
            raw_cam_bg_1 = F.relu(torch.sum(w_c_bg_1 * f_1, dim=1))  # (1,H,W)
            normed_cam_bg_1 = raw_cam_bg_1.clone().detach()
            for uni_cls in gt_labels:
                for wmin, hmin, wmax, hmax, _ in bboxes[bboxes[:, 4] == uni_cls]:
                    normed_cam_bg_1[:, hmin:hmax, wmin:wmax] = 0
            normed_cam_bg_1 = (normed_cam_bg_1 / normed_cam_bg_1.max()).detach().cpu()
            # normed_cam_bg_1_test = normed_cam_bg_1.numpy()#you

            # Final unary by contacinating foreground and background unaries
            unary_1 = torch.cat((Bg_unary_1, Fg_unary_1), dim=0)
            # unary_1_test = unary_1.numpy()


            # Unary witout background attn
            unary_u0_1 = torch.cat((normed_cam_bg_1, Fg_unary_1), dim=0)
            # unary_u0_1_test = unary_u0_1.numpy()


            #dierge
            f_2 = F.interpolate(f_2, img.shape[-2:], mode='bilinear', align_corners=True)
            padded_features_2 = pad_for_grid(f_2, cfg.MODEL.GRID_SIZE)
            padded_bg_mask_2 = pad_for_grid(bg_mask.cuda(), cfg.MODEL.GRID_SIZE)
            grid_bg_2, valid_gridIDs_2 = model.get_grid_bg_and_IDs(padded_bg_mask_2, cfg.MODEL.GRID_SIZE)
            bg_protos_2 = model.get_bg_prototypes(padded_features_2, padded_bg_mask_2, grid_bg_2, cfg.MODEL.GRID_SIZE)
            bg_protos_2 = bg_protos_2[0, valid_gridIDs_2]  # (1,GS**2,dims,1,1) --> (len(valid_gridIDs),dims,1,1)
            normed_bg_p_2 = F.normalize(bg_protos_2)
            normed_f_2 = F.normalize(f_2)

            # Background attention maps (u0)
            bg_attns_2 = F.relu(torch.sum(normed_bg_p_2 * normed_f_2, dim=1))
            bg_attn_2 = torch.mean(bg_attns_2, dim=0, keepdim=True)  # (len(valid_gridIDs),H,W) --> (1,H,W)
            # bg_attn_test = bg_attn_2.cpu().numpy()
            bg_attn_2[bg_attn_2 < cfg.MODEL.BG_THRESHOLD * bg_attn_2.max()] = 0
            Bg_unary_2 = torch.clone(bg_mask[0])  # (1,H,W)
            region_inside_bboxes_2 = Bg_unary_2[0] == 0  # (H,W)
            Bg_unary_2[:, region_inside_bboxes_2] = bg_attn_2[:, region_inside_bboxes_2].detach().cpu()
            # Bg_unary_2_test = Bg_unary_2.numpy()

            # CAMS for foreground classes (uc)
            Fg_unary_2 = []
            for uni_cls in gt_labels:
                w_c = WEIGHTS_2[uni_cls][None]
                raw_cam_2 = F.relu(torch.sum(w_c * f_2, dim=1))  # (1,H,W)
                normed_cam_2 = torch.zeros_like(raw_cam_2)
                for wmin, hmin, wmax, hmax, _ in bboxes[bboxes[:, 4] == uni_cls]:
                    denom_2 = raw_cam_2[:, hmin:hmax, wmin:wmax].max() + 1e-12
                    normed_cam_2[:, hmin:hmax, wmin:wmax] = raw_cam_2[:, hmin:hmax, wmin:wmax] / denom_2
                Fg_unary_2 += [normed_cam_2]
            Fg_unary_2 = torch.cat(Fg_unary_2, dim=0).detach().cpu()
            # Fg_unary_test = Fg_unary.numpy()#you

            # CAMS for background classes (ub)
            w_c_bg_2 = WEIGHTS_2[0][None]
            raw_cam_bg_2 = F.relu(torch.sum(w_c_bg_2 * f_2, dim=1))  # (1,H,W)
            normed_cam_bg_2 = raw_cam_bg_2.clone().detach()
            for uni_cls in gt_labels:
                for wmin, hmin, wmax, hmax, _ in bboxes[bboxes[:, 4] == uni_cls]:
                    normed_cam_bg_2[:, hmin:hmax, wmin:wmax] = 0
            normed_cam_bg_2 = (normed_cam_bg_2 / normed_cam_bg_2.max()).detach().cpu()
            # normed_cam_bg_test = normed_cam_bg_2.numpy()#you

            # gai dong 1
            #Fg_unary_2_he = 0.5 * Fg_unary_1 + 0.5 * Fg_unary_2
            Fg_unary_2_he = torch.max(Fg_unary_1, Fg_unary_2)
            # Fg_unary_2_he[Fg_unary_2_he >= 1.0] = 1.0
            fg_unary_2_he_test = Fg_unary_2_he.numpy()
            # Fg_unary_2_he = Fg_unary_1
            rgb_img_01 = rgb_img / 255
            Y_fg_he = show_cam_on_image(rgb_img_01, Fg_unary_2_he[0, :])
            Y_fg_he = Y_fg_he[:, :, ::-1]
            Y_fg_1 = show_cam_on_image(rgb_img_01, Fg_unary_1[0, :])
            Y_fg_1 = Y_fg_1[:, :, ::-1]
            Y_fg_2 = show_cam_on_image(rgb_img_01, Fg_unary_2[0, :])
            Y_fg_2 = Y_fg_2[:, :, ::-1]
            fg_unary_1_test = Fg_unary_1[0, :].detach().cpu().numpy()
            cam_fg1 = cv2.applyColorMap(np.uint8(255*Fg_unary_1[0, :].detach().cpu().numpy()), cv2.COLORMAP_JET)
            cam_fg1_bg1 = cam_fg1
            cam_fg1 = cam_fg1[:, :, ::-1]
            cam_fg_he = cv2.applyColorMap(np.uint8(255*Fg_unary_2_he[0, :].detach().cpu().numpy()), cv2.COLORMAP_JET)
            cam_fg_he = cam_fg_he[:, :, ::-1]
            cam_fg2 = cv2.applyColorMap(np.uint8(255*Fg_unary_2[0, :].detach().cpu().numpy()), cv2.COLORMAP_JET)
            cam_fg2_bg2 = cam_fg2
            cam_fg2 = cam_fg2[:, :, ::-1]
            cam_bg1 = cv2.applyColorMap(np.uint8(255*Bg_unary_1[0, :].detach().cpu().numpy()), cv2.COLORMAP_JET)
            cam_bg1 = cam_bg1[:, :, ::-1]
            cam_bg2 = cv2.applyColorMap(np.uint8(255*Bg_unary_2[0, :].detach().cpu().numpy()), cv2.COLORMAP_JET)
            cam_bg2 = cam_bg2[:, :, ::-1]
            # Final unary by contacinating foreground and background unaries
            unary_2_he = torch.cat((Bg_unary_1, Fg_unary_2_he), dim=0)
            # unary_2_test = unary_2.numpy()
            # unary_2[:, region_inside_bboxes_2] = torch.softmax(unary_2[:, region_inside_bboxes_2], dim=0)
            # # unary = unary.numpy()
            # refined_unary_2 = dCRF.inference(rgb_img, unary_2.numpy())

            # Unary witout background attn

            unary_u0_2_he = torch.cat((normed_cam_bg_1, Fg_unary_2_he), dim=0)

            unary_2_he[:, region_inside_bboxes_1] = torch.softmax(unary_2_he[:, region_inside_bboxes_1], dim=0)
            refined_unary = dCRF.inference(rgb_img, unary_2_he.numpy())

            unary_u0_2_he[:, region_inside_bboxes_1] = torch.softmax(unary_u0_2_he[:, region_inside_bboxes_1], dim=0)
            refined_unary_u0 = dCRF.inference(rgb_img, unary_u0_2_he.numpy())

            # (Out of bboxes) reset Fg scores to zero
            for idx_cls, uni_cls in enumerate(gt_labels, 1):
                mask = np.zeros((img_H, img_W))
                for wmin, hmin, wmax, hmax, _ in bboxes[bboxes[:, 4] == uni_cls]:
                    mask[hmin:hmax, wmin:wmax] = 1
                refined_unary[idx_cls] *= mask
                refined_unary_u0[idx_cls] *= mask

            # Y_crf and Y_crf_u0
            tmp_mask = refined_unary.argmax(0)
            tmp_mask_u0 = refined_unary_u0.argmax(0)
            Y_crf = np.zeros_like(tmp_mask, dtype=np.uint8)
            Y_crf_u0 = np.zeros_like(tmp_mask_u0, dtype=np.uint8)
            for idx_cls, uni_cls in enumerate(gt_labels, 1):
                Y_crf[tmp_mask == idx_cls] = uni_cls
                Y_crf_u0[tmp_mask_u0 == idx_cls] = uni_cls
            Y_crf[tmp_mask == 0] = 0
            Y_crf_u0[tmp_mask_u0 == 0] = 0

            #Y_crf_u0 = Y_crf_u0 * 255

            # Y_ret
            tmp_Y_crf = torch.from_numpy(Y_crf_u0)  # (H,W)
            gt_labels_with_Bg = [0] + gt_labels.tolist()
            corr_maps = []
            for uni_cls in gt_labels_with_Bg:
                indices = tmp_Y_crf == uni_cls
                # indices_test = indices.numpy()
                # indices_test_1 = f_1[..., indices].detach().cpu().numpy()
                if indices.sum():
                    normed_p = F.normalize(f_1[..., indices].mean(dim=-1))  # (1,dims)
                    corr = F.relu((normed_f_1 * normed_p[..., None, None]).sum(dim=1))  # (1,H,W)
                else:
                    normed_w = F.normalize(WEIGHTS[uni_cls][None])
                    corr = F.relu((normed_f_1 * normed_w).sum(dim=1))  # (1,H,W)
                corr_maps.append(corr)
            corr_maps = torch.cat(corr_maps)  # (1+len(gt_labels),H,W)
            corr_maps_test = corr_maps.detach().cpu().numpy()

            # (Out of bboxes) reset Fg correlations to zero
            for idx_cls, uni_cls in enumerate(gt_labels_with_Bg):
                if uni_cls == 0:
                    corr_maps[idx_cls, ~region_inside_bboxes_1] = 1
                else:
                    mask = torch.zeros(img_H, img_W).type_as(corr_maps)
                    for wmin, hmin, wmax, hmax, _ in bboxes[bboxes[:, 4] == uni_cls]:
                        mask[hmin:hmax, wmin:wmax] = 1
                    corr_maps[idx_cls] *= mask

            tmp_mask = corr_maps.argmax(0).detach().cpu().numpy()
            Y_ret = np.zeros_like(tmp_mask, dtype=np.uint8)
            for idx_cls, uni_cls in enumerate(gt_labels, 1):
                Y_ret[tmp_mask == idx_cls] = uni_cls
            Y_ret[tmp_mask == 0] = 0
            if cfg.SAVE_PSEUDO_LABLES:
                for pseudo, save_path in zip([Y_crf, Y_ret, Y_crf_u0, Y_fg_he, Y_fg_1, Y_fg_2,
                                              cam_fg1, cam_fg2, cam_fg_he, cam_bg1, cam_bg2,
                                              cam_fg1_bg1, cam_fg2_bg2], save_paths):
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