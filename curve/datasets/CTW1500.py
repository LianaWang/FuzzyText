import os
from multiprocessing import Pool

import mmcv
import numpy as np
from shapely.geometry import Polygon

from mmdet.datasets import DATASETS
from .ArT import ArTDataset


@DATASETS.register_module()
class CTW1500(ArTDataset):

    def _text_to_bboxes(self, text):
        """ 32d : [xmin, ymin, xmax, ymax, offset_x, offset_y ....]
        :param text:
        :return: polygon of 14 pts: 28d
        """
        box = text.split(',')  #
        box = np.array(box, dtype=np.float32)
        box[4::2] += box[0]
        box[5::2] += box[1]
        box = box[4:]
        return box

    def read_ann_info(self, img_id, TRAIN=True):
        filename = os.path.join(self.img_prefix, 'Annotations', img_id + '.txt')
        texts = open(filename).readlines()
        num_objs = len(texts)
        boxes = []
        points = []
        labels = []
        centers = []
        hard_flg = np.zeros((num_objs), dtype=np.int32)

        for i in range(num_objs):
            box = self._text_to_bboxes(texts[i])
            pts = box.reshape((-1, 2))
            center = np.mean(box, 0)
            centers.append(center)
            boxes.append(box)
            points.append(pts)
            labels.append(1)
        return boxes, labels, centers, points, None, hard_flg
