import os
import numpy as np
from PIL import Image
import xml.etree.cElementTree as et
import argparse

def main(data_root):
    root_dir = data_root
    annot_dir = os.path.join(root_dir,"Annotations")
    output_dir = os.path.join(root_dir,"BgMaskfromBoxes")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for filename in os.listdir(annot_dir):
        file_annot = os.path.join(annot_dir,filename)
        tree=et.parse(file_annot)
        root=tree.getroot()
        bounding_boxes = []
        img_size = []
        for size in root.iter('size'):
            for width in size.iter('width'):
                w = int(width.text)
                img_size.append(w)
            for height in root.iter('height'):
                h = int(height.text)
                img_size.append(h)
        #flipped dimensions here
        mask = np.ones((img_size[1],img_size[0]),dtype=np.uint8)
        for bndbox in root.iter('bndbox'):
            for xmin in bndbox.iter('xmin'):
                x_min = int(float(xmin.text))
            for xmax in bndbox.iter('xmax'):
                x_max = int(float(xmax.text))
            for ymin in bndbox.iter('ymin'):
                y_min = int(float(ymin.text))
            for ymax in bndbox.iter('ymax'):
                y_max = int(float(ymax.text))
            mask[y_min:y_max,x_min:x_max] = 0
            
        im = Image.fromarray(mask)
        output_name = filename[:-4] + ".png"
        output_path = os.path.join(output_dir,output_name)
        print(output_name)
        im.save(output_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    data_root = args.data_root
    main(data_root)