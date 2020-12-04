#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
 Author: zhaopenghao
 Create Time: 2019/12/18 下午2:24
"""
import matplotlib.pyplot as plt
import numpy as np
from paddlehub.dataset.base_cv_dataset import ObjectDetectionDataset
from paddlehub.reader.cv_reader import ObjectDetectionReader

# define dataset
ds = ObjectDetectionDataset()
ds.base_path = '/Users/zhaopenghao/Downloads/coco_10'
ds.train_image_dir = 'val'
ds.train_list_file = 'annotations/val.json'
ds.validate_image_dir = 'val'
ds.validate_list_file = 'annotations/val.json'

# define batch reader
obreader = ObjectDetectionReader(1,1, dataset=ds, model_type='yolo')
breader = obreader.data_generator(2, phase='train')

all_batches = []
for b in breader():
    all_batches.append(b)

train_egs = obreader.get_train_examples()

print("train egs", len(train_egs))
print("batch num", len(all_batches))

b1 = all_batches[1]
print("b1[0]", b1[0])
print("b1[0] im_shape", b1[0][3])
# print("b1[0]['gt_class']:", b1[0][2])
# image = b1[0][0]
# print("image shape and type:", image.shape, type(image))
# image2 = np.transpose(image, [1,2,0])
# print("image2 shape and type:", image2.shape, type(image2))
# plt.imshow(image2)
# plt.show()

# import pdb; pdb.set_trace()

