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
    state_dict = torch.load(model_stage_1.name)
    print(state_dict.keys())
    print("1")

    # model.load_state_dict(state_dict)
    #
    # WEIGHTS = torch.clone(model.classifier.weight.data)
    # WEIGHTS_2 = torch.clone(model.classifier2.weight.data)
    # model.eval()





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