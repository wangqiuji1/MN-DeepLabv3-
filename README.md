# MN-DeepLabv3
This is a code focusing on high-precision corrosion degree nondestructive segmentation method with virtual and real synthetic data labeled by unsupervised learning

# Requirements environment
PyTorch >=1.7.1 is recommended for the use of this code.

# Dataset preparation
This code was trained using the dataset in the VOC format. The tag image files were placed in SegmentationClass under the VOCdevkit/VOC2007 folder. The origin image files were placed in JPEGImages under the VOCdevkit/VOC2007 folder.
voc_annotation.py was run to get 2007_train.txt and 2007_val.txt for training.

# Training
Run train.py to start training. 

# Testing
Modify model_path in the deeplab.py file. The model_path corresponds to the weight file.
Run predict.py.
