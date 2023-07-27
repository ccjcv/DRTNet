import os
import numpy as np
from PIL import Image
import argparse

try:
    from pycocotools.coco import COCO
    import imagesize
except ModuleNotFoundError:
    pass

def main(data_root):
  root_dir = data_root
  directory = os.path.join(root_dir,"train2017")
  ann_path = os.path.join(root_dir,'annotations/instances_train2017.json')
  annotations = COCO(ann_path)
  cat_ids = annotations.getCatIds()
  img_ids = []
  for cat in cat_ids:
      img_ids.extend(annotations.getImgIds(catIds=cat))      
  img_ids = list(set(img_ids))

  output_dir = os.path.join(root_dir,"BgMaskfromBoxes")
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  j=0        
  for img_id in img_ids:
          
      ann_ids = annotations.getAnnIds(imgIds=img_id)
      coco_annotation = annotations.loadAnns(ann_ids)

      # number of objects in the image
      num_objs = len(coco_annotation) 
      path = annotations.loadImgs(img_id)[0]['file_name']
      img_path =os.path.join(directory,path)
      img_size =imagesize.get(img_path)
      mask = np.ones((img_size[0],img_size[1]),dtype=np.uint8)
      
      # Bounding boxes for objects
      # In coco format, bbox = [xmin, ymin, width, height]
      # In numpy, the input should be [xmin, ymin, xmax, ymax]

      for i in range(num_objs):
        temp=coco_annotation[i]['bbox']
        xmin = int(temp[0]) 
        ymin = int(temp[1])
        xmax = xmin + int(temp[2])
        ymax = ymin + int(temp[3])
        mask[ymin:ymax,xmin:xmax] = 0
      
      im = Image.fromarray(mask)
      output_name = path[:-4] + ".png"
      output_path = os.path.join(output_dir,output_name)
      im.save(output_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    data_root = args.data_root
    main(data_root)