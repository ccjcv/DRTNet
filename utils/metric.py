# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
from medpy.metric import binary
try:
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils
except ModuleNotFoundError:
    pass

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls_2 = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    dsc = 2 * np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))
    dsc = dict(zip(range(n_class), dsc))


    return {
        "Pixel Accuracy": acc_cls,
        "Mean Accuracy": acc_cls_2,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
        "dsc": dsc
    }

import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def PACU(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def MACU(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def MIOU(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU
    def CLS_IOU(self):#TP / (TP + FP + FN)
        CLS_IOU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        return CLS_IOU
    def dsc(self):#2TP / (2TP + FP + FN)
        dsc = 2 * np.diag(self.confusion_matrix) /(
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0))
        return dsc
    def FIOU(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

class COCO_Evaluator():
    def __init__(self,GROUND_TRUTH_DIR,IMG_DIR,img_ids=None):
        '''
        GROUND_TRUTH_DIR : Directory containing ground truth dataset. Must contain directories- annotations and val2017
        IMG_DIR : Directory containing generated labels
        img_ids : IDs of images. Default=None. If None all image IDs are considered.
        '''
        self.GROUND_TRUTH_DIR=GROUND_TRUTH_DIR
        self.IMG_DIR=IMG_DIR
        self.ann_path = os.path.join(GROUND_TRUTH_DIR,'annotations/instances_val2017.json')
        self.annotations = COCO(self.ann_path)
        self.cat_ids = annotations.getCatIds()
        if img_ids is None:
            self.img_ids = []
            for cat in self.cat_ids:
                self.img_ids.extend(annotations.getImgIds(catIds=cat))
        else:
            self.img_ids=img_ids
        self.img_ids = list(set(self.img_ids))
        print(f"Number of images: {len(self.img_ids)}")
        self.filter_img()
        print(f"Number of RGB images: {len(self.img_ids)}")
        self.cat_id_map=self.map_pixels_to_classIDs()

    def filter_img(self):
        # ignore grayscale images
        DIR=os.path.join(self.GROUND_TRUTH_DIR,"val2017")
        gray_scale_ids = []
        for id in self.img_ids:
            file_name = self.annotations.loadImgs(ids=id)[0]['file_name']
            img = np.asarray(Image.open(os.path.join(self.IMG_DIR,file_name)))
            if img.ndim == 2:
                gray_scale_ids.append(id)  
        self.img_ids = set(self.img_ids)
        all_gray_scale_ids = set(gray_scale_ids)
        self.img_ids = list(self.img_ids - all_gray_scale_ids)
    
    def map_pixels_to_classIDs(self):
        cat_id_map = [-1]*91
        for i in range(0,80):
            cat_id_map[self.cat_ids[i]] = i+1
        return cat_id_map
    
    def get_res(self):
        results = []
        for img_id in self.img_ids:
            img_file_name = self.annotations.loadImgs(img_id)[0]['file_name']
            res_img = np.asarray(Image.open(os.path.join(self.IMG_DIR,img_file_name[:-4]+'.png')))
  
            for id in range(1,81):
                #perform reverse mapping
                class_mask = res_img == id
                orig_cat_id = self.cat_id_map.index(id)
                cat_name = self.annotations.loadCats(orig_cat_id)[0]['name']
    
                if np.sum(class_mask!=0):
                    result = { 
                        "image_id": img_id,
                        "category_id": orig_cat_id,
                        "score": 1,
                        "segmentation": maskUtils.encode(np.asfortranarray(class_mask))
                        }
                    results.append(result)
        return results  