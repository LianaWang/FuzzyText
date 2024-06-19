import os.path as osp
import warnings

import cv2
import copy
import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.datasets.builder import PIPELINES

__all__ = ['LoadCurveAnnotations']

@PIPELINES.register_module()
class LoadCurveAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 skip_img_without_anno=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.skip_img_without_anno = skip_img_without_anno

    def _load_bboxes(self, results):
        ann_info = copy.deepcopy(results['ann_info']) # avoid being modified by last epoch
        # print('anno_info', ann_info.keys())
        results['gt_bboxes'] = ann_info['bboxes']
        results['bbox_fields'].extend(['gt_bboxes'])
        if 'bboxes_ignore' in ann_info.keys():
            results['gt_bboxes_ignore'] = ann_info['bboxes_ignore']
            results['bbox_fields'].extend(['gt_bboxes_ignore'])

        if 'cheby' in ann_info.keys():
            results['gt_coefs'] = ann_info['cheby']
            results['bbox_fields'].extend(['gt_coefs'])
        elif 'fourier' in ann_info.keys():
            results['gt_coefs'] = ann_info['fourier']
            results['bbox_fields'].extend(['gt_coefs'])
        elif 'poly' in ann_info.keys():
            results['gt_coefs'] = ann_info['poly']
            results['bbox_fields'].extend(['gt_coefs'])
        elif 'rotate_cheby' in ann_info.keys():
            results['gt_coefs'] = ann_info['rotate_cheby']
            results['bbox_fields'].extend(['gt_coefs'])

        if 'skeleton' in ann_info.keys():
            results['gt_skeleton'] = ann_info['skeleton']            
            results['bbox_fields'].extend(['gt_skeleton'])
        if len(results['gt_bboxes']) == 0 and self.skip_img_without_anno:
            file_path = osp.join(results['img_prefix'],
                                 results['img_info']['filename'])
            warnings.warn(
                'Skip the image "{}" that has no valid gt bbox'.format(
                    file_path))
            return None  
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
#         h, w = results['img_info']['height'], results['img_info']['width']
        mask_path = osp.join(results['img_prefix'], results['ann_info']['mask'])
        mask_gap_path = osp.join(results['img_prefix'], results['ann_info']['mask_gap'])
        # mask_contour_path = osp.join(results['img_prefix'], results['ann_info']['mask_contour'])
        # mask_dilate_path = mask_contour_path.replace("Masks_contour", "Masks_dilate")
        # read mask
        grayImage = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, gt_mask = cv2.threshold(grayImage,127,1,cv2.THRESH_BINARY)
        grayImage_gap = cv2.imread(mask_gap_path, 0)
        _, gt_mask_gap = cv2.threshold(grayImage_gap,127,1,cv2.THRESH_BINARY)
        # gt_mask_contour = cv2.imread(mask_contour_path, cv2.IMREAD_GRAYSCALE)
        gt_mask_contour = grayImage_gap
#         grayImage_dilate = cv2.imread(mask_dilate_path, cv2.IMREAD_GRAYSCALE)
#         _, gt_mask_dilate = cv2.threshold(grayImage_dilate,127,1,cv2.THRESH_BINARY)
        results['gt_mask'] = gt_mask
        results['mask_fields'].append('gt_mask')
        results['gt_mask_gap'] = gt_mask_gap
        results['mask_fields'].append('gt_mask_gap')
        results['gt_mask_contour'] = gt_mask_contour
        results['mask_fields'].append('gt_mask_contour')
#         results['gt_mask_dilate'] = gt_mask_dilate
#         results['mask_fields'].append('gt_mask_contour')

        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str
