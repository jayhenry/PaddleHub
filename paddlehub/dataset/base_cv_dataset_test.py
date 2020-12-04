#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
 Author: zhaopenghao
 Create Time: 2019/12/18 下午12:23
"""
from paddlehub.dataset.base_cv_dataset import ObjectDetectionDataset


ds = ObjectDetectionDataset()
ds.base_path = '/Users/zhaopenghao/Downloads/coco_10'
ds.train_image_dir = 'val'
ds.train_list_file = 'annotations/val.json'

itr = ds.train_data(False)
train_egs = ds.get_train_examples()
label_dict = ds.label_dict()

print(label_dict)
print("number:", len(train_egs))

for x in itr:
    print(x)
    break