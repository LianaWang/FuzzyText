import json
import os.path as osp
from multiprocessing import Pool
import os
import mmcv
import cv2
import numpy as np
from shapely.geometry import Polygon

from mmdet.datasets import CustomDataset
from mmdet.datasets import DATASETS
from .curve_utils import expand_twelve, sample_contour, cheby_fit, poly_fit, fourier_fit, rotate_cheby_fit

from .eval_utils import eval_polygons
from mmcv.utils import print_log
from mmdet.utils import get_root_logger

@DATASETS.register_module()
class ArTDataset(CustomDataset):
    CLASSES = ('text')

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 debug_mode=False,
                 cache_root=None,
                 encoding='cheby',
                 degree=22,
                 sample_pts=360):
        self.data_root = data_root
        self.cache_root = cache_root
        self.img_prefix = img_prefix
        self.degree = degree
        self.sample_pts = sample_pts
        self.load_dict = None
        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
        assert encoding in ['cheby', 'poly', 'fourier', 'none']
        self.encoding = encoding
        self.debug = debug_mode
        # super init
        super(ArTDataset, self).__init__(ann_file, pipeline, classes, data_root,
                                         img_prefix, seg_prefix,
                                         proposal_file, test_mode, False)

    def load_annotations(self, ann_file):
        img_ids = mmcv.list_from_file(self.ann_file)
        self.img_ids = img_ids
        if self.debug:
            print('Skipping Loading Anoonations')
            return [] * len(img_ids)
        dir_name = osp.join(self.img_prefix, self.cache_root) #'Cache_%dx%d'%(scale[0], scale[1])
        if not osp.exists(dir_name):
            os.makedirs(dir_name)
        pool = Pool(8)
        img_infos = pool.map(self._load_annotations, img_ids)
        pool.close()
        pool.join()
        print("\nload success with %d samples in load_annotations" % len(img_infos))
        return img_infos


    def _load_annotations(self, img_id):
        dir_name = osp.join(self.img_prefix, self.cache_root)
        ann_path = '{}/{}.npy'.format(dir_name, img_id)
        try:
            info_dict = np.load(ann_path, allow_pickle=True).item()
        except Exception as err:
            if osp.exists(ann_path):
                print(err)
            filename = 'JPGImages/{}.jpg'.format(img_id)
            im_path = osp.join(self.img_prefix, filename)
            height, width, _ = mmcv.imread(im_path).shape
            info_dict = dict(id=img_id, filename=filename, width=width, height=height)
            if not self.test_mode:
                ann = self.compute_ann_info(img_id)
                info_dict.update({"ann": ann})
            np.save(ann_path, info_dict)
            print('.', end='', flush=True)
        return info_dict

    def read_ann_info(self, img_id, TRAIN=True):
        """
        boxes are list of 1d coordinates, labels are all ones, centers are center points of polygons
        points are original pts of [k, 2]
        trans, hard_flag ...
        :param img_id: 
        :return: boxes, labels, centers, points, trans, hard_flg
        """
        if self.load_dict is None:
            anno_path = osp.join(self.img_prefix, 'train_labels.json')
            with open(anno_path, 'r') as f:
                self.load_dict = json.load(f)
        ano = self.load_dict['gt_%d' % (int(img_id))]
        boxes = []
        labels = []
        centers = []
        points = []
        trans = []
        num_objs = len(ano)
        hard_flg = np.zeros((num_objs), dtype=np.int32)

        for ix in range(num_objs):
            pts = np.array(ano[ix]['points'])
            twelve_pts = expand_twelve(pts) if TRAIN else pts
            center = np.mean(twelve_pts, 0)
            points.append(pts)
            centers.append(np.array(center))
            boxes.append(twelve_pts.reshape(-1))
            labels.append(1)
            trans.append(len(ano[ix]['transcription']))
            hard_flg[ix] = 1 if ano[ix]['illegibility'] else 0
        return boxes, labels, centers, points, trans, hard_flg

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        if ann_info is None: # remove shits
            return None
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def compute_ann_info(self, img_id):
        boxes, labels, centers, points, _, hard_flg = self.read_ann_info(img_id)
        
        filename = 'Masks/{}.png'.format(img_id)
        filename_gap = 'Masks_gap/{}.png'.format(img_id)
        filename_contour = 'Masks_contour/{}.png'.format(img_id)

        idx_ignore = np.where(hard_flg == 1)[0]
        idx_easy = np.where(hard_flg == 0)[0]
        boxes_easy = np.array(boxes)[idx_easy, :]
        boxes_ignore = np.array(boxes)[idx_ignore, :]
        centers_easy = np.array(centers)[idx_easy, :]
        centers_ignore = np.array(centers)[idx_ignore, :]
        labels = np.array(labels)[idx_easy]
        try:
            bboxes = np.hstack((boxes_easy, centers_easy))
        except:
            print(boxes_easy.shape, centers_easy.shape)
        bboxes_ignore = np.hstack((boxes_ignore, centers_ignore))
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels=labels.astype(np.int64),
            mask = filename,
            mask_gap = filename_gap,
            mask_contour = filename_contour,
        )
        return ann


    def get_gt_points(self):
        gt_points = []
        for img_info in self.data_infos:
            img_id = img_info['id']
            _,_,_,pts,_,hards = self.read_ann_info(img_id, TRAIN=False)
            samples = [(pts[idx], hards[idx]) for idx in range(len(hards))]
            gt_points.append(samples)
        return gt_points

    def evaluate(self, outputs, metric='mAP', logger=None, **kwargs):
        output_dict = {}
        thr_lists = kwargs.get('proposal_thr_list', [])
        print_log(f'Evaluating {metric}\n', logger)
        gt_list = []
        gt_points = self.get_gt_points()
        for idx in range(len(gt_points)):
            gt_polygons = [[Polygon(pts), hard] for (pts, hard) in gt_points[idx]]
            gt_list.append(gt_polygons)

        for thr in thr_lists:
            pred_list = []
            for idx in range(len(outputs)):
                pred_polygons = []
                for boxes in outputs[idx]:
                    pts = boxes[:-1].reshape((-1, 2)) # coords
                    score = float(boxes[-1])
                    if score < thr:
                        continue
                    poly = Polygon(pts)
                    poly = poly if poly.is_valid else poly.buffer(0)
                    pred_polygons.append([poly, score])
                pred_list.append(pred_polygons)
            eval_output = eval_polygons(pred_list, gt_list) # eval function in eval_utils.py
            msg = eval_output.pop('output_string')
            print_log(f'Evaluation By Thr {thr:.3f}: '+msg, logger)
            for key, val in eval_output.items():
                if "f_measure" in key:
                    output_dict[f'{key}({thr})'] = val
        return output_dict
