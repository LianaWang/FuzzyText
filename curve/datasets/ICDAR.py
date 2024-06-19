from .ArT import ArTDataset
import numpy as np
import os
import mmcv
from multiprocessing import Pool
from mmdet.datasets import DATASETS


@DATASETS.register_module()
class ICDAR(ArTDataset):

    def read_ann_info(self, img_id):
        filename = os.path.join(self.img_prefix, 'Annotations', img_id + '.gt')
        texts = open(filename).readlines()
        num_objs = len(texts)
        boxes = []
        labels = []
        centers = []
        hard_flg = np.zeros((num_objs), dtype=np.int32)

        for i in range(num_objs):
            text = texts[i].strip(' \n')
            _hard, *box = [int(i) for i in text.split(' ')]
            box = np.array(box).reshape([-1, 2])
            center = np.mean(box, axis=0)
            hard_flg[i] = _hard
            centers.append(center)
            boxes.append(box.reshape(-1))
            labels.append(1)
        return boxes, labels, centers, boxes, None, hard_flg
