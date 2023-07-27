#!/bin/bash

DATASETS_ROOT=$1
SEGMENTATION_AUG_ROOT=$DATASETS_ROOT/VOCdevkit/VOC2012
IMAGESET_ROOT=$DATASETS_ROOT/VOCdevkit/VOC2012/ImageSets

echo '1. Downloading the VOC dataset'
wget -nc -P $DATASETS_ROOT http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf $DATASETS_ROOT/VOCtrainval_11-May-2012.tar -C $DATASETS_ROOT

echo '2. Downloading augmentated training data as SegmentationClassAug'
wget -nc -P $DATASETS_ROOT 'https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip'
unzip $DATASETS_ROOT/SegmentationClassAug.zip -d $SEGMENTATION_AUG_ROOT

echo '3. Downloading official image sets as ImageSets/SegmentationAug'
wget -nc -P $DATASETS_ROOT https://github.com/kazuto1011/deeplab-pytorch/files/2945588/list.zip
unzip $DATASETS_ROOT/list.zip -d $IMAGESET_ROOT
mv $IMAGESET_ROOT/list $IMAGESET_ROOT/SegmentationAug

echo 'Removing the redundant files'
rm $DATASETS_ROOT/VOCtrainval_11-May-2012.tar
rm $DATASETS_ROOT/SegmentationClassAug.zip
rm $DATASETS_ROOT/list.zip

echo 'Pascal VOC 2012 dataset downloaded and setup. Process finished!'