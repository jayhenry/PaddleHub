#coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import paddlehub as hub
from paddlehub.common.downloader import default_downloader
from ..contrib.ppdet.data.source import build_source
from ..common import detection_config as dconf


class ImageClassificationDataset(object):
    def __init__(self):
        self.base_path = None
        self.train_list_file = None
        self.test_list_file = None
        self.validate_list_file = None
        self.label_list_file = None
        self.num_labels = 0
        self.label_list = []

        self.train_examples = []
        self.dev_examples = []
        self.test_examples = []

    def _download_dataset(self, dataset_path, url):
        if not os.path.exists(dataset_path):
            result, tips, dataset_path = default_downloader.download_file_and_uncompress(
                url=url,
                save_path=hub.common.dir.DATA_HOME,
                print_progress=True,
                replace=True)
            if not result:
                print(tips)
                exit()
        return dataset_path

    def _parse_data(self, data_path, shuffle=False, phase=None):
        data = []
        with open(data_path, "r") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line = line.strip()
                items = line.split(" ")
                if len(items) > 2:
                    image_path = " ".join(items[0:-1])
                else:
                    image_path = items[0]
                if not os.path.isabs(image_path):
                    if self.base_path is not None:
                        image_path = os.path.join(self.base_path, image_path)
                label = items[-1]
                data.append((image_path, items[-1]))

        if phase == 'train':
            self.train_examples = data
        elif phase == 'dev':
            self.dev_examples = data
        elif phase == 'test':
            self.test_examples = data

        if shuffle:
            np.random.shuffle(data)

        def _base_reader():
            for item in data:
                yield item

        return _base_reader()

    def label_dict(self):
        if not self.label_list:
            with open(os.path.join(self.base_path, self.label_list_file),
                      "r") as file:
                self.label_list = file.read().split("\n")
        return {index: key for index, key in enumerate(self.label_list)}

    def train_data(self, shuffle=True):
        train_data_path = os.path.join(self.base_path, self.train_list_file)
        return self._parse_data(train_data_path, shuffle, phase='train')

    def test_data(self, shuffle=False):
        test_data_path = os.path.join(self.base_path, self.test_list_file)
        return self._parse_data(test_data_path, shuffle, phase='dev')

    def validate_data(self, shuffle=False):
        validate_data_path = os.path.join(self.base_path,
                                          self.validate_list_file)
        return self._parse_data(validate_data_path, shuffle, phase='test')

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples


class ObjectDetectionDataset(ImageClassificationDataset):
    def __init__(self, base_path, train_image_dir, train_list_file, validate_image_dir, validate_list_file,
                 test_image_dir, test_list_file, model_type='ssd'):
        super(ObjectDetectionDataset, self).__init__()
        self.base_path = base_path
        self.train_image_dir = train_image_dir
        self.train_list_file = train_list_file
        self.validate_image_dir = validate_image_dir
        self.validate_list_file = validate_list_file
        self.test_image_dir = test_image_dir
        self.test_list_file = test_list_file
        self.model_type = model_type
        self._dsc = None
        self.cid2cname = None
        self.label_dict()  # refresh cid2cname and num_labels
        assert self.cid2cname is not None
        assert self.num_labels > 0

    def label_dict(self):
        if self.cid2cname is not None:
            return self.cid2cname
        # get label info from train data json
        _ = self.train_data()
        return self.cid2cname

    def _parse_data(self, data_path, image_dir, shuffle=False, phase=None):
        with_background = dconf.conf[self.model_type]['with_background']
        mixup_epoch = -1
        if phase == 'train':
            mixup_epoch = dconf.conf[self.model_type].get('mixup_epoch', -1)
        file_conf = {
            'ANNO_FILE': data_path,
            'IMAGE_DIR': image_dir,
            # 'USE_DEFAULT_LABEL': feed.dataset.use_default_label,
            'IS_SHUFFLE': shuffle,
            'SAMPLES': -1,
            'WITH_BACKGROUND': with_background,
            'MIXUP_EPOCH': mixup_epoch,
            'TYPE': 'RoiDbSource',
        }
        sc_conf = {'data_cf': file_conf, 'cname2cid': None}
        data_source = build_source(sc_conf)
        self._dsc = data_source
        data_source.reset()
        data = data_source._roidb
        if self.cid2cname is None:
            cname2cid = data_source.cname2cid
            cid2cname = {v: k for k, v in cname2cid.items()}
            self.cid2cname = cid2cname
            if with_background:
                self.num_labels = len(cid2cname) + 1
            else:
                self.num_labels = len(cid2cname)

        if phase == 'train':
            self.train_examples = data
        elif phase == 'dev':
            self.dev_examples = data
        elif phase == 'test':
            self.test_examples = data
        return data_source

    def train_data(self, shuffle=True):
        train_data_path = os.path.join(self.base_path, self.train_list_file)
        train_image_dir = os.path.join(self.base_path, self.train_image_dir)
        return self._parse_data(
            train_data_path, train_image_dir, shuffle, phase='train')

    def test_data(self, shuffle=False):
        test_data_path = os.path.join(self.base_path, self.test_list_file)
        test_image_dir = os.path.join(self.base_path, self.test_image_dir)
        return self._parse_data(
            test_data_path, test_image_dir, shuffle, phase='dev')

    def validate_data(self, shuffle=False):
        validate_data_path = os.path.join(self.base_path,
                                          self.validate_list_file)
        validate_image_dir = os.path.join(self.base_path,
                                          self.validate_image_dir)
        return self._parse_data(
            validate_data_path, validate_image_dir, shuffle, phase='test')
