# PFAN
 Code for CVPR-2019 paper "Progressive Feature Alignment for Unsupervised Domain Adaptation",We will release a journal version code which further improves the reported results in our paper.We will keep updating this code.

Prerequisites:
    Python2/Python3
    Tensorflow 1.10
    Numpy
    
Dataset:
You need to download the domain_adaptation_images dataset for test.

Training:
    1.run 'train.py' to get the prototype vector
    2.run 'pseudo.py' to get the new train dataset
    3.execute 1&2 alternatively and iteratively


Citation:
If you use this code for your research, please consider citing:
@InProceedings{PFAN_2019_CVPR,
author = {Chen, Chaoqi and Xie, Weiping and Huang, Wenbing and Rong, Yu and Ding, Xinghao and Huang, Yue and Xu, Tingyang and Huang, Junzhou},
title = {Progressive Feature Alignment for Unsupervised Domain Adaptation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2019}
}

Contact:
If you have any problem about our code, feel free to contact Xiewp67@stu.xmu.edu.cn.
