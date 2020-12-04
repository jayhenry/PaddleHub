#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
 Author: zhaopenghao
 Create Time: 2019/12/18 上午11:27
"""
import json
import random
import numpy as np
import os
import subprocess
from collections import defaultdict

input_dir = '/home/local1/data/coco2017/'
img = '{}/val2017'.format(input_dir)
anno = "{}/annotations/instances_val2017.format.json".format(input_dir)

m = 10
output_dir = '/home/local3/zhaopenghao/data/detect_data/coco_10'
out_img = '{}/val'.format(output_dir)
out_anno_dir = '{}/annotations'.format(output_dir)
out_anno = '{}/annotations/val.json'.format(output_dir)
os.system("mkdir -p {}".format(out_img))
os.system("mkdir -p {}".format(out_anno_dir))

d = json.load(open(anno))

sampled = {
    'info': d['info'],
    'licenses': d['licenses'],
    'images': [],
    'annotations': [],
    'categories': d['categories']
}

images = d['images']
annotations = d['annotations']

image_id2image = {}
for image in images:
    image_id = image['id']
    image_id2image[image_id] = image

image_id2anno = defaultdict(list)
for annotation in annotations:
    image_id = annotation['image_id']
    image_id2anno[image_id].append(annotation)

print('images:', len(images))
print('annotations:', len(annotations))
n = len(images)
idx = np.arange(n)
np.random.shuffle(idx)

samplea = []
samplei = []
for i, iidx in enumerate(idx):
    if i >= m:
        break
    image = images[iidx]
    image_id = image['id']
    annos = image_id2anno[image_id]
    image_fname = image['file_name']
    samplea.extend(annos)
    samplei.append(image)
    src_img = '{}/{}'.format(img, image_fname)
    dest_img = '{}/{}'.format(out_img, image_fname)
    subprocess.check_call("cp {} {}".format(src_img, dest_img), shell=True)


sampled['images'] = sorted(samplei, key=lambda x: x['id'])
sampled['annotations'] = sorted(samplea, key=lambda x: x['id'])

with open(out_anno, 'w') as f:
    json.dump(sampled, f, indent=4)

print("sample done")

