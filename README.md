# DK_HRS

PyTorch Implementation of paper "A Domain Knowledge Powered Hybrid Regularization Strategy for Semi-supervised Breast Cancer Diagnosis"

1、Environment

Cuda 10.1 + python 3.6 + pytorch 1.4.0

2、Dataset

The dataset used in our paper mainly consists of 4390 ultrasound images, readers can find some samples in direcory 'uda_data', 
and the samples of data path during training process can be found in "labeled_images_20.pth" and "labeled_images_80.pth" 

3、Train & Test

Train (an example under 20% of labeled data setting):

bash train_b8mu30_dk_vat_28.sh

Test:

bash val_dk_vat.sh