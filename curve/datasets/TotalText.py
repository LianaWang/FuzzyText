import os
from multiprocessing import Pool

import mmcv
import numpy as np
from shapely.geometry import Polygon

from mmdet.datasets import DATASETS
from .ArT import ArTDataset
from .curve_utils import expand_twelve


@DATASETS.register_module()
class TotalText(ArTDataset):

    def _text_to_bboxes(self, text):
        x = text[0][5:-2].split()
        x = np.array(x, dtype=np.float32)
        y = text[1][6:-2].split()
        y = np.array(y, dtype=np.float32)
        box = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
        return box

    def read_ann_info(self, img_id, TRAIN=True):
        filename = os.path.join(self.img_prefix, 'Annotations', img_id + '.txt')
        texts = open(filename).readlines()
        num_objs = len(texts)
        points = []
        boxes = []
        labels = []
        centers = []
        hard_flag = np.zeros((num_objs), dtype=np.int32)

        for i in range(num_objs):
            text = texts[i].split(',')
            pts = self._text_to_bboxes(text)
            hard = int(text[-1][20:-3] == '#')
            twelve_pts = expand_twelve(pts) if TRAIN else pts
            center = np.mean(twelve_pts, 0)

            points.append(pts)
            boxes.append(twelve_pts.reshape(-1))
            centers.append(center)
            labels.append(1)
            hard_flag[i] = hard

        return boxes, labels, centers, points, None, hard_flag
