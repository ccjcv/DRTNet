import torch
import torch.nn as nn
import torch.nn.functional as F

# from .Layers import VGG16
from models import resnet50

def pad_for_grid(mask, grid_size):
    Pad_H = grid_size - mask.shape[2] % grid_size
    Pad_W = grid_size - mask.shape[3] % grid_size
    if Pad_H == grid_size:
        Pad_H = 0
    if Pad_W == grid_size:
        Pad_W = 0
    if Pad_H % 2 == 0:
        if Pad_W % 2 == 0:
            out = F.pad(mask, [Pad_W // 2, Pad_W // 2, Pad_H // 2, Pad_H // 2], value=0)
        else:
            out = F.pad(mask, [0, Pad_W, Pad_H // 2, Pad_H // 2], value=0)
    else:
        if Pad_W % 2 == 0:
            out = F.pad(mask, [Pad_W // 2, Pad_W // 2, 0, Pad_H], value=0)
        else:
            out = F.pad(mask, [0, Pad_W, 0, Pad_H], value=0)
    return out


class Labeler(nn.Module):
    def __init__(self, num_classes, roi_size, grid_size):
        super().__init__()
        #这部分是两路resnet50
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)

        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        # self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.classifier = nn.Conv2d(1024, num_classes, 1, bias=False)
        self.resnet50_2 = resnet50.resnet50(pretrained=True, use_amm=True, strides=(2, 2, 2, 1))
        self.stage2_1 = nn.Sequential(self.resnet50_2.conv1, self.resnet50_2.bn1, self.resnet50_2.relu,
                                      self.resnet50_2.maxpool,
                                      self.resnet50_2.layer1)

        self.stage2_2 = nn.Sequential(self.resnet50_2.layer2)
        self.stage2_3 = nn.Sequential(self.resnet50_2.layer3)
        # self.stage2_4 = nn.Sequential(self.resnet50_2.layer4)

        self.classifier2 = nn.Conv2d(1024, num_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3])
        self.backbone2 = nn.ModuleList([self.stage2_1, self.stage2_2, self.stage2_3])
        self.newly_added = nn.ModuleList([self.classifier, self.classifier2])

        # self.backbone = VGG16(dilation=1)
        # self.classifier = nn.Conv2d(1024, num_classes, 1, bias=False)

        self.OH, self.OW = roi_size
        self.GS = grid_size
        self.from_scratch_layers = [self.classifier]#这是啥意思

    def get_features(self, x):
        #有两个特征
        x_ori = x.clone()
        # # branch1
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        # x = self.stage4(x).detach()

        # # branch2
        x2 = self.stage2_1(x_ori)
        x2 = self.stage2_2(x2)
        x2 = self.stage2_3(x2)
        # x2 = self.stage2_4(x2)

        return x, x2#返回两个特征
        # return self.backbone(x)

    def weighted_avg_pool_2d(self, input, weight):
        filtered = input * weight
        _, dims, input_H, input_W = filtered.shape
        stride_H = input_H // self.OH
        stride_W = input_W // self.OW
        if stride_H == 0:
            stride_H += 1
            pad_H = self.OH - input_H
            filtered = F.pad(filtered, [0, 0, 0, pad_H], mode='replicate')
            weight = F.pad(weight, [0, 0, 0, pad_H], mode='replicate')
        if stride_W == 0:
            stride_W += 1
            pad_W = self.OW - input_W
            filtered = F.pad(filtered, [pad_W, 0, 0, 0], mode='replicate')
            weight = F.pad(weight, [pad_W, 0, 0, 0], mode='replicate')
        ks_H = input_H - (self.OH - 1) * stride_H
        ks_W = input_W - (self.OW - 1) * stride_W
        if ks_H <= 0:
            ks_H = 1
        if ks_W <= 0:
            ks_W = 1
        kernel = torch.ones((dims, 1, ks_H, ks_W)).type_as(filtered)
        numer = F.conv2d(filtered, kernel, stride=(stride_H, stride_W), groups=dims)
        denom = F.conv2d(weight, kernel[0][None], stride=(stride_H, stride_W)) + 1e-12
        return numer / denom

    def gen_grid(self, box_coord, width, height):
        wmin, hmin, wmax, hmax = box_coord[:4]
        grid_x = torch.linspace(wmin, wmax, width).view(1, 1, width, 1)
        grid_y = torch.linspace(hmin, hmax, height).view(1, height, 1, 1)
        grid_x = grid_x.expand(1, height, width, 1)
        grid_y = grid_y.expand(1, height, width, 1)
        grid = torch.cat((grid_x, grid_y), dim=-1)
        return grid

    def BAP(self, features, bboxes, batchID_of_box, bg_protos, valid_cellIDs, ind_valid_bg_mask, GAP=False):
        batch_size, _, fH, fW = features.shape
        norm_H, norm_W = (fH - 1) / 2, (fW - 1) / 2
        widths = bboxes[:, [0, 2]] * norm_W + norm_W
        heights = bboxes[:, [1, 3]] * norm_H + norm_H
        widths = (widths[:, 1].ceil() - widths[:, 0].floor()).int()
        heights = (heights[:, 1].ceil() - heights[:, 0].floor()).int()
        fg_protos = []
        for batch_id in range(batch_size):
            feature_map = features[batch_id][None]  # (1,dims,fH,fW)
            indices = batchID_of_box == batch_id
            for coord, width, height in zip(bboxes[indices], widths[indices], heights[indices]):
                grid = self.gen_grid(coord, width, height).type_as(feature_map)
                roi = F.grid_sample(feature_map, grid)  # (1,dims,BH,BW)
                GAP_attn = torch.ones(1, 1, *roi.shape[-2:]).type_as(roi)
                ID_list = valid_cellIDs[batch_id]
                if GAP:
                    fg_by_GAP = self.weighted_avg_pool_2d(roi, GAP_attn)  # (1,256,OH,OW)
                    fg_protos.append(fg_by_GAP)
                else:
                    if ind_valid_bg_mask[batch_id] and len(ID_list):
                        normed_roi = F.normalize(roi, dim=1)
                        valid_bg_p = bg_protos[batch_id, ID_list]  # (N,GS**2,dims,1,1)->(len(ID_list),dims,1,1)
                        normed_bg_p = F.normalize(valid_bg_p, dim=1)
                        bg_attns = F.relu(torch.sum(normed_roi * normed_bg_p, dim=1, keepdim=True))
                        bg_attn = torch.mean(bg_attns, dim=0, keepdim=True)
                        fg_attn = 1 - bg_attn
                        fg_by_BAP = self.weighted_avg_pool_2d(roi, fg_attn)  # (1,256,OH,OW)
                        fg_protos.append(fg_by_BAP)
                    else:
                        fg_by_GAP = self.weighted_avg_pool_2d(roi, GAP_attn)  # (1,256,OH,OW)
                        fg_protos.append(fg_by_GAP)
        fg_protos = torch.cat(fg_protos, dim=0)
        return fg_protos

    def get_grid_bg_and_IDs(self, padded_mask, grid_size):
        batch_size, _, padded_H, padded_W = padded_mask.shape
        cell_H, cell_W = padded_H // grid_size, padded_W // grid_size
        grid_bg = padded_mask.unfold(2, cell_H, cell_H).unfold(3, cell_W, cell_W)
        grid_bg = torch.sum(grid_bg, dim=(4, 5))  # (N,1,GS,GS,cH,cW) --> (N,1,GS,GS)
        grid_bg = grid_bg.view(-1, 1, 1, 1)  # (N * GS**2,1,1,1)
        valid_gridIDs = [idx for idx, cell in enumerate(grid_bg) if cell > 0]
        grid_bg = grid_bg.view(batch_size, -1, 1, 1, 1)  # (N,GS**2,1,1,1)
        return grid_bg, valid_gridIDs

    def get_bg_prototypes(self, padded_features, padded_mask, denom_grids, grid_size):
        batch_size, dims, padded_H, padded_W = padded_features.shape
        cell_H, cell_W = padded_H // grid_size, padded_W // grid_size
        bg_features = (padded_mask * padded_features).unfold(2, cell_H, cell_H).unfold(3, cell_W, cell_W)
        bg_protos = torch.sum(bg_features, dim=(4, 5))  # (N,dims,GS,GS,cH,cW) --> (N,dims,GS,GS)
        bg_protos = bg_protos.view(batch_size, dims, -1).permute(0, 2, 1)
        bg_protos = bg_protos.contiguous().view(batch_size, -1, dims, 1, 1)
        bg_protos = bg_protos / (denom_grids + 1e-12)  # (N,GS**2,dims,1,1)
        return bg_protos

    def forward(self, img, bboxes, batchID_of_box, bg_mask, ind_valid_bg_mask, GAP=False):
        '''
        img               : (N,3,H,W) float32
        bboxes            : (K,5) float32
        batchID_of_box    : (K,) int64
        bg_mask           : (N,1,H,W) float32
        ind_valid_bg_mask : (N,) uint8
        '''
        bboxes_CAM = bboxes.clone()
        features, features_2 = self.get_features(img)  # (N,256,105,105)  321实际跑出来(8,1024,41,41)
        # 512 (8,1024,64,64)

        batch_size, dims, fH, fW = features.shape
        ##########################################################
        padded_mask = pad_for_grid(F.interpolate(bg_mask, (fH, fW)), self.GS)  # 将M变为特征大小
        grid_bg, valid_gridIDs = self.get_grid_bg_and_IDs(padded_mask, self.GS)
        valid_cellIDs = []
        for grids in grid_bg:
            valid_cellIDs.append([idx for idx, cell in enumerate(grids) if cell > 0])
        ##########################################################
        padded_features = pad_for_grid(features, self.GS)
        bg_protos = self.get_bg_prototypes(padded_features, padded_mask, grid_bg, self.GS)
        fg_protos = self.BAP(features, bboxes, batchID_of_box, bg_protos, valid_cellIDs, ind_valid_bg_mask, GAP)
        ##########################################################
        num_fgs = fg_protos.shape[0]
        fg_protos = fg_protos.view(num_fgs, dims, -1).permute(0, 2, 1).contiguous().view(-1, dims, 1,
                                                                                         1)  # (num_fgs,dims,OH,OW) --> (num_fgs*OH*OW,dims,1,1)
        bg_protos = bg_protos.contiguous().view(-1, dims, 1, 1)[valid_gridIDs]  # (len(valid_gridIDs),dims,1,1)
        protos = torch.cat((fg_protos, bg_protos), dim=0)
        out = self.classifier(protos)

        #产生CAMs_1
        # bboxes_CAM = bboxes[0]  # (1,K,5) --> (K,5) bounding boxes
        # bg_mask_CAM = bg_mask[None]  # (1,H,W) --> (1,1,H,W) background mask


        img_H_CAM, img_W_CAM = features.shape[-2:]
        norm_H_CAM, norm_W_CAM = (img_H_CAM - 1) / 2, (img_W_CAM - 1) / 2
        bboxes_CAM[:, [0, 2]] = bboxes_CAM[:, [0, 2]] * norm_W_CAM + norm_W_CAM
        bboxes_CAM[:, [1, 3]] = bboxes_CAM[:, [1, 3]] * norm_H_CAM + norm_H_CAM
        bboxes_CAM = bboxes_CAM.long()
        gt_labels_CAM = bboxes_CAM[:, 4].unique()

        # # features_CAM = F.interpolate(features, img.shape[-2:], mode='bilinear', align_corners=True)
        # features_CAM = features
        # padded_features_CAM = padded_features  # 将f划分为N*N个规则网格
        # padded_bg_mask_CAM = padded_mask  # 将掩码划分为N*N个规则网络
        # grid_bg_CAM, valid_gridIDs_CAM = self.get_grid_bg_and_IDs(padded_bg_mask_CAM, self.GS)  # 每个网格和对应ID
        # bg_protos_CAM = self.get_bg_prototypes(padded_features_CAM, padded_bg_mask_CAM, grid_bg_CAM, self.GS)
        # bg_protos_CAM = bg_protos_CAM[0, valid_gridIDs_CAM]  # (1,GS**2,dims,1,1) --> (len(valid_gridIDs),dims,1,1)
        # normed_bg_p_CAM = F.normalize(bg_protos_CAM)  # 全是背景的查询qj
        # normed_f_CAM = F.normalize(features_CAM)

        # # Background attention maps (u0)
        # # 计算边界框内的特征与查询qj之间的余弦相似度,结节中心区域的余弦相似度有一部分为1。
        # bg_attns_CAM = F.relu(torch.sum(normed_bg_p_CAM * normed_f_CAM, dim=1))
        # # 就是公式中的attention map A
        # bg_attn_CAM = torch.mean(bg_attns_CAM, dim=0, keepdim=True)  # (len(valid_gridIDs),H,W) --> (1,H,W)
        # bg_attn_CAM[bg_attn_CAM < 0.99 * bg_attn_CAM.max()] = 0
        # Bg_unary_CAM = torch.clone(bg_mask[0])  # (1,H,W)
        # region_inside_bboxes_CAM = Bg_unary_CAM[0] == 0  # (H,W)
        # Bg_unary_CAM[:, region_inside_bboxes_CAM] = bg_attn_CAM[:, region_inside_bboxes_CAM].detach().cpu()

        # # CAMS for foreground classes (uc)
        # WEIGHTS = torch.clone(self.classifier.weight.data)
        # Fg_unary_CAM = []
        # for uni_cls_CAM in gt_labels_CAM:
        #     w_c = WEIGHTS[uni_cls_CAM][None]  # 前景类权重
        #     raw_cam = F.relu(torch.sum(w_c * features, dim=1))  # (1,H,W)
        #     # raw_cam = raw_cam.cpu().detach().numpy()
        #     normed_cam = torch.zeros_like(raw_cam)
        #     for wmin, hmin, wmax, hmax, _ in bboxes_CAM[bboxes_CAM[:, 4] == uni_cls_CAM]:
        #
        #         if hmax == 0 or hmax == hmin:
        #             hmax = hmin + 1
        #         if wmax == 0 or wmax == wmin:
        #             wmax = wmin + 1
        #
        #         # print(hmin, hmax, wmin, wmax)
        #         # print(raw_cam.size())
        #         denom = raw_cam[:, hmin:hmax, wmin:wmax].max() + 1e-12
        #         normed_cam[:, hmin:hmax, wmin:wmax] = raw_cam[:, hmin:hmax, wmin:wmax] / denom
        #     Fg_unary_CAM += [normed_cam]
        # Fg_unary_CAM = torch.cat(Fg_unary_CAM, dim=0).detach()

        # CAMS for foreground classes (uc) wu kuang
        WEIGHTS = torch.clone(self.classifier.weight.data)
        Fg_unary_CAM = []
        for uni_cls_CAM in gt_labels_CAM:
            w_c = WEIGHTS[uni_cls_CAM][None]  # 前景类权重
            raw_cam = F.relu(torch.sum(w_c * features, dim=1))  # (1,H,W)
            # raw_cam = raw_cam.cpu().detach().numpy()
            normed_cam = torch.zeros_like(raw_cam)
            for wmin, hmin, wmax, hmax, _ in bboxes_CAM[bboxes_CAM[:, 4] == uni_cls_CAM]:

                # if hmax == 0 or hmax == hmin:
                #     hmax = hmin + 1
                # if wmax == 0 or wmax == wmin:
                #     wmax = wmin + 1

                # print(hmin, hmax, wmin, wmax)
                # print(raw_cam.size())
                denom = raw_cam[:, :, :].max() + 1e-12
                normed_cam[:, :, :] = raw_cam[:, :, :] / denom
            Fg_unary_CAM += [normed_cam]
        Fg_unary_CAM = torch.cat(Fg_unary_CAM, dim=0).detach()

        # CAMS for background classes (ub)
        # w_c_bg_CAM = WEIGHTS[0][None]  # 背景类权重
        # raw_cam_bg_CAM = F.relu(torch.sum(w_c_bg_CAM * features, dim=1))  # (1,H,W)
        # normed_cam_bg_CAM = raw_cam_bg_CAM.clone().detach()
        # for uni_cls_CAM in gt_labels_CAM:
        #     for wmin, hmin, wmax, hmax, _ in bboxes_CAM[bboxes_CAM[:, 4] == uni_cls_CAM]:
        #
        #         if hmax == 0 or hmax == hmin:
        #             hmax = hmin + 1
        #         if wmax == 0 or wmax == wmin:
        #             wmax = wmin + 1
        #         normed_cam_bg_CAM[:, hmin:hmax, wmin:wmax] = 0
        # normed_cam_bg_CAM = (normed_cam_bg_CAM / normed_cam_bg_CAM.max()).detach()
        #
        # # # Final unary by contacinating foreground and background unaries
        # # unary_CAM = torch.cat((Bg_unary_CAM, Fg_unary_CAM), dim=0)
        # # # unary_CAM[:, region_inside_bboxes_CAM] = torch.softmax(unary_CAM[:, region_inside_bboxes_CAM], dim=0)
        # # # refined_unary = dCRF.inference(rgb_img, unary.numpy())
        #
        # # Unary witout background attn
        # unary_u0_CAM = torch.cat((normed_cam_bg_CAM, Fg_unary_CAM), dim=0)
        # # unary_u0_CAM[:, region_inside_bboxes_CAM] = torch.softmax(unary_u0_CAM[:, region_inside_bboxes_CAM], dim=0)
        # # refined_unary_u0 = dCRF.inference(rgb_img, unary_u0.numpy())
        #
        # # unary_u0_CAM = torch.softmax(unary_u0_CAM, dim=0)

        #支路2
        batch_size_2, dims_2, fH_2, fW_2 = features_2.shape
        ##########################################################
        padded_mask_2 = pad_for_grid(F.interpolate(bg_mask, (fH_2, fW_2)), self.GS)  # 将M变为特征大小
        grid_bg_2, valid_gridIDs_2 = self.get_grid_bg_and_IDs(padded_mask_2, self.GS)
        valid_cellIDs_2 = []
        for grids_2 in grid_bg_2:
            valid_cellIDs_2.append([idx for idx, cell in enumerate(grids_2) if cell > 0])
        ##########################################################
        padded_features_2 = pad_for_grid(features_2, self.GS)
        bg_protos_2 = self.get_bg_prototypes(padded_features_2, padded_mask_2, grid_bg_2, self.GS)
        fg_protos_2 = self.BAP(features_2, bboxes, batchID_of_box, bg_protos_2, valid_cellIDs_2, ind_valid_bg_mask, GAP)
        ##########################################################
        num_fgs_2 = fg_protos_2.shape[0]
        fg_protos_2 = fg_protos_2.view(num_fgs_2, dims_2, -1).permute(0, 2, 1).contiguous().view(-1, dims_2, 1,
                                                                                         1)  # (num_fgs,dims,OH,OW) --> (num_fgs*OH*OW,dims,1,1)
        bg_protos_2 = bg_protos_2.contiguous().view(-1, dims_2, 1, 1)[valid_gridIDs_2]  # (len(valid_gridIDs),dims,1,1)
        protos_2 = torch.cat((fg_protos_2, bg_protos_2), dim=0)
        out_2 = self.classifier2(protos_2)

        # 产生CAMs_2
        # bboxes_CAM_2 = bboxes[0]  # (1,K,5) --> (K,5) bounding boxes
        # bg_mask_CAM_2 = bg_mask[None]  # (1,H,W) --> (1,1,H,W) background mask
        bboxes_CAM_2 = bboxes.clone()

        img_H_CAM_2, img_W_CAM_2 = features_2.shape[-2:]
        norm_H_CAM_2, norm_W_CAM_2 = (img_H_CAM_2 - 1) / 2, (img_W_CAM_2 - 1) / 2
        bboxes_CAM_2[:, [0, 2]] = bboxes_CAM_2[:, [0, 2]] * norm_W_CAM_2 + norm_W_CAM_2
        bboxes_CAM_2[:, [1, 3]] = bboxes_CAM_2[:, [1, 3]] * norm_H_CAM_2 + norm_H_CAM_2
        bboxes_CAM_2 = bboxes_CAM_2.long()
        gt_labels_CAM_2 = bboxes_CAM_2[:, 4].unique()

        # # features_CAM_2 = F.interpolate(features_2, img.shape[-2:], mode='bilinear', align_corners=True)
        # features_CAM_2 = features_2
        # padded_features_CAM_2 = padded_features_2  # 将f划分为N*N个规则网格
        # padded_bg_mask_CAM_2 = padded_mask_2  # 将掩码划分为N*N个规则网络
        # grid_bg_CAM_2, valid_gridIDs_CAM_2 = self.get_grid_bg_and_IDs(padded_bg_mask_CAM_2, self.GS)  # 每个网格和对应ID
        # bg_protos_CAM_2 = self.get_bg_prototypes(padded_features_CAM_2, padded_bg_mask_CAM_2, grid_bg_CAM_2, self.GS)
        # bg_protos_CAM_2 = bg_protos_CAM_2[0, valid_gridIDs_CAM_2]  # (1,GS**2,dims,1,1) --> (len(valid_gridIDs),dims,1,1)
        # normed_bg_p_CAM_2 = F.normalize(bg_protos_CAM_2)  # 全是背景的查询qj
        # normed_f_CAM_2 = F.normalize(features_CAM_2)

        # # Background attention maps (u0)
        # # 计算边界框内的特征与查询qj之间的余弦相似度,结节中心区域的余弦相似度有一部分为1。
        # bg_attns_CAM_2 = F.relu(torch.sum(normed_bg_p_CAM_2 * normed_f_CAM_2, dim=1))
        # # 就是公式中的attention map A
        # bg_attn_CAM_2 = torch.mean(bg_attns_CAM_2, dim=0, keepdim=True)  # (len(valid_gridIDs),H,W) --> (1,H,W)
        # bg_attn_CAM_2[bg_attn_CAM_2 < 0.99 * bg_attn_CAM_2.max()] = 0
        # Bg_unary_CAM_2 = torch.clone(bg_mask[0])  # (1,H,W)
        # region_inside_bboxes_CAM_2 = Bg_unary_CAM_2[0] == 0  # (H,W)
        # Bg_unary_CAM_2[:, region_inside_bboxes_CAM_2] = bg_attn_CAM_2[:, region_inside_bboxes_CAM_2].detach().cpu()

        # # CAMS for foreground classes (uc)
        # WEIGHTS_2 = torch.clone(self.classifier2.weight.data)
        # Fg_unary_CAM_2 = []
        # for uni_cls_CAM_2 in gt_labels_CAM_2:
        #     w_c_2 = WEIGHTS_2[uni_cls_CAM_2][None]  # 前景类权重
        #     raw_cam_2 = F.relu(torch.sum(w_c_2 * features_2, dim=1))  # (1,H,W)
        #     normed_cam_2 = torch.zeros_like(raw_cam_2)
        #     for wmin, hmin, wmax, hmax, _ in bboxes_CAM_2[bboxes_CAM_2[:, 4] == uni_cls_CAM_2]:
        #
        #         if hmax == 0 or hmax == hmin:
        #             hmax = hmin + 1
        #         if wmax == 0 or wmax == wmin:
        #             wmax = wmin + 1
        #         denom_2 = raw_cam_2[:, hmin:hmax, wmin:wmax].max() + 1e-12
        #         normed_cam_2[:, hmin:hmax, wmin:wmax] = raw_cam_2[:, hmin:hmax, wmin:wmax] / denom_2
        #     Fg_unary_CAM_2 += [normed_cam_2]
        # Fg_unary_CAM_2 = torch.cat(Fg_unary_CAM_2, dim=0).detach()

        # CAMS for foreground classes (uc), wu kuang
        WEIGHTS_2 = torch.clone(self.classifier2.weight.data)
        Fg_unary_CAM_2 = []
        for uni_cls_CAM_2 in gt_labels_CAM_2:
            w_c_2 = WEIGHTS_2[uni_cls_CAM_2][None]  # 前景类权重
            raw_cam_2 = F.relu(torch.sum(w_c_2 * features_2, dim=1))  # (1,H,W)
            normed_cam_2 = torch.zeros_like(raw_cam_2)
            for wmin, hmin, wmax, hmax, _ in bboxes_CAM_2[bboxes_CAM_2[:, 4] == uni_cls_CAM_2]:

                # if hmax == 0 or hmax == hmin:
                #     hmax = hmin + 1
                # if wmax == 0 or wmax == wmin:
                #     wmax = wmin + 1
                denom_2 = raw_cam_2[:, :, :].max() + 1e-12
                normed_cam_2[:, :, :] = raw_cam_2[:, :, :] / denom_2
            Fg_unary_CAM_2 += [normed_cam_2]
        Fg_unary_CAM_2 = torch.cat(Fg_unary_CAM_2, dim=0).detach()

        # CAMS for background classes (ub)
        # w_c_bg_CAM_2 = WEIGHTS_2[0][None]  # 背景类权重
        # raw_cam_bg_CAM_2 = F.relu(torch.sum(w_c_bg_CAM_2 * features_2, dim=1))  # (1,H,W)
        # normed_cam_bg_CAM_2 = raw_cam_bg_CAM_2.clone().detach()
        # for uni_cls_CAM_2 in gt_labels_CAM_2:
        #     for wmin, hmin, wmax, hmax, _ in bboxes_CAM_2[bboxes_CAM_2[:, 4] == uni_cls_CAM_2]:
        #         if hmax == 0 or hmax == hmin:
        #             hmax = hmin + 1
        #         if wmax == 0 or wmax == wmin:
        #             wmax = wmin + 1
        #
        #         normed_cam_bg_CAM_2[:, hmin:hmax, wmin:wmax] = 0
        # normed_cam_bg_CAM_2 = (normed_cam_bg_CAM_2 / normed_cam_bg_CAM_2.max()).detach()
        #
        # # # Final unary by contacinating foreground and background unaries
        # # unary_CAM_2 = torch.cat((Bg_unary_CAM_2, Fg_unary_CAM_2), dim=0)
        # # # unary_CAM[:, region_inside_bboxes_CAM] = torch.softmax(unary_CAM[:, region_inside_bboxes_CAM], dim=0)
        # # # refined_unary = dCRF.inference(rgb_img, unary.numpy())
        #
        # # Unary witout background attn
        # unary_u0_CAM_2 = torch.cat((normed_cam_bg_CAM_2, Fg_unary_CAM_2), dim=0)
        # # unary_u0_CAM[:, region_inside_bboxes_CAM] = torch.softmax(unary_u0_CAM[:, region_inside_bboxes_CAM], dim=0)
        # # refined_unary_u0 = dCRF.inference(rgb_img, unary_u0.numpy())

        # unary_u0_CAM_2 = torch.softmax(unary_u0_CAM_2, dim=0)
        Fg_unary_CAM = torch.softmax(Fg_unary_CAM, dim=0)
        Fg_unary_CAM_2 = torch.softmax(Fg_unary_CAM_2, dim=0)
        cam = Fg_unary_CAM
        cam2 = Fg_unary_CAM_2
        return out, out_2, cam, cam2

    def get_params(self, do_init=True):
        '''
        This function is borrowed from AffinitNet. It returns (pret_weight, pret_bias, scratch_weight, scratch_bias).
        Please, also see the paper (Learning Pixel-level Semantic Affinity with Image-level Supervision, CVPR 2018), and codes (https://github.com/jiwoon-ahn/psa/tree/master/network).
        '''
        params = ([], [], [], [])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m in self.from_scratch_layers:
                    if do_init:
                        nn.init.normal_(m.weight, std=0.01)
                    params[2].append(m.weight)
                else:
                    params[0].append(m.weight)
                if m.bias is not None:
                    if m in self.from_scratch_layers:
                        if do_init:
                            nn.init.constant_(m.bias, 0)
                        params[3].append(m.bias)
                    else:
                        params[1].append(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                if m in self.from_scratch_layers:
                    if do_init:
                        nn.init.constant_(m.weight, 1)
                    params[2].append(m.weight)
                else:
                    params[0].append(m.weight)
                if m.bias is not None:
                    if m in self.from_scratch_layers:
                        if do_init:
                            nn.init.constant_(m.bias, 0)
                        params[3].append(m.bias)
                    else:
                        params[1].append(m.bias)
        return params