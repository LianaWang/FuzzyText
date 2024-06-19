import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from shapely.geometry import Point
from scipy import misc, ndimage
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init, build_norm_layer
from mmcv.runner import force_fp32, auto_fp16
from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss


@HEADS.register_module()
class SingleMaskHead(nn.Module):

    def __init__(self,
                 num_classes=2,
                 in_channels=256,
                 out_channels=256,
                 norm_cfg=None,
                 conv_cfg=None, # dict(type='DCNv2')
                 att_cfg=None,
                 up2x_num=0,
                 loss_mask=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_gap=dict(type='CrossEntropyLoss', loss_weight=2.0),
                 loss_contour=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 ):
        super(SingleMaskHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.att_cfg = att_cfg
        self.up2x_num = up2x_num
        self.loss_mask = build_loss(loss_mask)
        self.loss_gap = None if loss_gap is None else build_loss(loss_gap)
        self.loss_contour = None if loss_contour is None else build_loss(loss_contour)
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_fuse = ConvModule(self.in_channels, self.out_channels, 3, padding=1,
                                    conv_cfg=None, norm_cfg=self.norm_cfg)
        if self.att_cfg is not None:
            self.conv_att = AttHead(**self.att_cfg)
        # mask _convs
        self.mask_convs = nn.ModuleList()
        layer_num = 3
        for i in range(layer_num):
            conv_cfg = self.conv_cfg if i==layer_num-1 else None
            conv_norm_relu = ConvModule(self.out_channels, self.out_channels, 3, padding=1,
                                    conv_cfg=conv_cfg, norm_cfg=self.norm_cfg)
            self.mask_convs.append(conv_norm_relu)
        # logits
        self.mask_logits = nn.Conv2d(self.out_channels, 1 if self.loss_mask.use_sigmoid else 2, 1)
        # gap _convs
        if self.loss_gap:
            self.gap_convs = nn.ModuleList()
            for i in range(layer_num):
                conv_cfg = self.conv_cfg if i==layer_num-1 else None
                conv_norm_relu = ConvModule(self.out_channels, self.out_channels, 3, padding=1,
                                        conv_cfg=conv_cfg, norm_cfg=self.norm_cfg)
                self.gap_convs.append(conv_norm_relu)
            # logits
            self.gap_logits = nn.Conv2d(self.out_channels, 1 if self.loss_gap.use_sigmoid else 2, 1)
            normal_init(self.gap_logits, std=0.01)

        if self.loss_contour:
            self.contour_convs = nn.ModuleList()
            for i in range(layer_num):
                conv_cfg = self.conv_cfg if i==layer_num-1 else None
                conv_norm_relu = ConvModule(self.out_channels, self.out_channels, 3, padding=1,
                                        conv_cfg=conv_cfg, norm_cfg=self.norm_cfg)
                self.contour_convs.append(conv_norm_relu)
            self.contour_logits = nn.Conv2d(self.out_channels, 1, 1)
            normal_init(self.contour_logits, std=0.01)
            # gap sdt
            self.gapsdt_convs = nn.ModuleList()
            for i in range(layer_num):
                conv_cfg = self.conv_cfg if i==layer_num-1 else None
                conv_norm_relu = ConvModule(self.out_channels, self.out_channels, 3, padding=1,
                                        conv_cfg=conv_cfg, norm_cfg=self.norm_cfg)
                self.gapsdt_convs.append(conv_norm_relu)
            self.gapsdt_logits = nn.Conv2d(self.out_channels, 1, 1)
            normal_init(self.gapsdt_logits, std=0.01)

    def init_weights(self):
        """Initialize weights of the head."""
#        for m in self.mask_convs:
#            if isinstance(m.conv, nn.Conv2d):
#                normal_init(m.conv, std=0.01)
#        for m in self.gap_convs:
#            if isinstance(m.conv, nn.Conv2d):
#                normal_init(m.conv, std=0.01)
        normal_init(self.mask_logits, std=0.01)

    def forward_train(self,
                      x,
                      x_shape,
                      img_metas,
                      gt_masks,
                      gt_masks_gap,
                      gt_mask_contour,
                      gt_bboxes):
        outs = self.forward(x)
        loss_inputs = outs + (x_shape, img_metas, gt_masks, gt_masks_gap, gt_mask_contour, gt_bboxes)
        losses = self.loss(*loss_inputs)
        return losses
    
#     def forward(self, feats):
#         return multi_apply(self.forward_single, feats)


    @force_fp32(apply_to=('mask_pred', 'gap_pred', 'gapsdt_pred', 'contour_pred', ))
    def forward(self, x):
        x = self.conv_fuse(x)
        if self.att_cfg is not None:
            x = self.conv_att(x)
        # masks
        mask_feat = x
        for layer in self.mask_convs:
            mask_feat = layer(mask_feat)
        # mask_feat = torch.cat([mask_feat, gap_feat], dim=1)
        mask_pred = self.mask_logits(mask_feat)

        # gap
        gap_pred = None
        if self.loss_gap:
            gap_feat = x
            for layer in self.gap_convs:
                gap_feat = layer(gap_feat)
            # gap_feat = torch.cat([mask_feat, gap_feat], dim=1)
            gap_pred = self.gap_logits(gap_feat)

        # sdt
        contour_pred = None
        gapsdt_pred = None
        if self.loss_contour:
            contour_feat = x
            for layer in self.contour_convs:
                contour_feat = layer(contour_feat)
            contour_pred = self.contour_logits(contour_feat)

            gapsdt_feat = x
            for layer in self.gapsdt_convs:
                gapsdt_feat = layer(gapsdt_feat)
            gapsdt_pred = self.gapsdt_logits(gapsdt_feat)

        return mask_pred, gap_pred, contour_pred, gapsdt_pred

    
    @force_fp32(apply_to=('mask_pred', 'gap_pred', 'contour_pred', 'gapsdt_pred'))
    def loss(self, mask_pred, gap_pred, contour_pred, gapsdt_pred,
             x_shape, img_metas, gt_masks, gt_masks_gap, gt_contour, gt_bboxes):
        N = len(gt_masks)
        _, Hx, Wx = x_shape
        pad_dims = [(0, Wx-gt_mask.size(1), 0, Hx-gt_mask.size(0)) for gt_mask in gt_masks]
        gt_masks = torch.stack([F.pad(gt_masks[i].data, pad_dims[i]) for i in range(N)]).to(mask_pred.device)
        gt_masks_gap = torch.stack([F.pad(gt_masks_gap[i].data, pad_dims[i]) for i in range(N)]).to(mask_pred.device)
        # gt_contour = torch.stack([F.pad(gt_contour[i].data, pad_dims[i]) for i in range(N)]).to(mask_pred.device)
        H, W = mask_pred.shape[-2:]
        gt_mask_i = F.interpolate(gt_masks.unsqueeze(1), size=(H, W)).long()
        gt_mask_gap_i = F.interpolate(gt_masks_gap.unsqueeze(1), size=(H, W)).long()
        # gt_contour_i = F.interpolate(gt_contour.unsqueeze(1), size=(H, W)).float()/255.0
        loss_dict = dict()
        if self.loss_contour:
            # uncertainty
            sdts = []
            for i, gt_bbox in enumerate(gt_bboxes):
                gt_bbox = gt_bbox[:, :-2] * (H/Hx)
                gt_bbox = gt_bbox.reshape(gt_bbox.size(0), -1, 1, 2) # N * P * 1 * 2
                _points = np.array(gt_bbox.cpu()).astype(np.int32)
                _contours = cv2.drawContours(np.zeros(gt_mask_i[i].squeeze(0).size()), _points, -1, 1) #list(gt_bbox)
                dt = ndimage.distance_transform_edt(_contours == 0)
                sdt = dt.copy() #phi_0
                thres = 10.0
                sdt[sdt > thres] = thres  # truncate
                sdt[gt_mask_i[i].squeeze(0).cpu() <= 0] *= -1.0
                sdt = sdt / thres
                sdts.append(sdt)
            # gap sdt
            gapsdts = []
            kernel = np.ones((3,3), np.uint8)
            for i in range(N):
                gt_gap_i = np.array(gt_mask_gap_i[i, ...].squeeze(0).cpu()).astype(np.uint8)
                # contours, _ = cv2.findContours(gt_gap_i.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                # contour_map = cv2.drawContours(gt_gap_i.astype("uint8")*0, contours, -1, 255)
                if gt_gap_i.sum()>0:
                    contour_map = cv2.dilate(gt_gap_i, kernel, iterations=1) - gt_gap_i
                    dt = ndimage.distance_transform_edt(contour_map == 0)
                    sdt = dt.copy() #phi_0
                    thres = 10.0
                    sdt[sdt > thres] = thres  # truncate
                    sdt[gt_gap_i <= 0] *= -1.0
                    sdt = sdt / thres
                else:
                    sdt = np.ones(gt_gap_i.shape) * -1.0
                gapsdts.append(sdt)
            gt_sdts = torch.tensor(sdts).view(mask_pred.shape).to(mask_pred.device)
            gt_gapsdts = torch.tensor(gapsdts).view(mask_pred.shape).to(mask_pred.device)
            loss_dict['loss_sdt'] = self.loss_contour(contour_pred, gt_sdts)
            loss_dict['loss_gapsdt'] = self.loss_contour(gapsdt_pred, gt_gapsdts)
        if self.loss_gap:
            loss_dict['loss_gap'] = self.loss_gap(gap_pred, gt_mask_gap_i, gt_gapsdts)
        # loss mask
        loss_dict['loss_mask'] = self.loss_mask(mask_pred, gt_mask_i, gt_sdts)
        return loss_dict
    

    def find_cc(self, final, min_area):
        label_num, labels = cv2.connectedComponents(final, connectivity=4)
        for label_idx in range(1, label_num):
            if np.sum(labels == label_idx) < min_area:
                labels[labels == label_idx] = 0
            # neighbors
        return labels


    def get_bboxes(self, label, score, scale):
#         labels_num = np.max(label) + 1
        labels = np.array(torch.tensor(label, dtype=torch.int64).unique())
        bboxes = []
#         for i in range(1, label_num):
        for i in labels[1:]:
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]
            score_i = torch.mean(score[label == i])
            # rect = cv2.minAreaRect(points)
            binary = np.zeros(label.shape, dtype='uint8')
            binary[label == i] = 1

            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         epsilon = 0.1 * cv2.arcLength(contour, True)
    #         bbox = cv2.approxPolyDP(contour, epsilon, True)
            if len(contours)==0:
                continue
            bbox = contours[0]
            if bbox.shape[0] <= 2:
                continue
            bbox = bbox / scale
            bbox = bbox.astype('int32')
            bboxes.append(np.hstack([bbox.reshape(-1), score_i]))
        return bboxes
    
    def post_processing(self, mask, gap, sdt, gapsdt, min_area, scale, thres=0.5):
        # refine
        r = 0.1
        mask = mask*(1 + r*sdt)
        gap = gap*(1 + r*gapsdt)
        mask = mask.clamp(min=0, max=1)
        gap = gap.clamp(min=0, max=1)
        final = mask*(1.0 - gap)
        # thres
        binary_final = np.array((torch.sign(final - thres) + 1) / 2).astype(np.uint8)

        labels = self.find_cc(binary_final, min_area)
        bboxes = self.get_bboxes(labels, final, scale)
        return final, bboxes
    
    
    @force_fp32(apply_to=('mask_pred', 'gap_pred',))
    def simple_test_mask(self, x, img_metas, input_shape):
        # import pdb; pdb.set_trace()
        N, C, H, W = input_shape # img_metas[i]['pad_shape']
        model_outs = self.forward(x) #4,C,H,W
        mask_pred, gap_pred = model_outs[:2]
        outs = []
        final_list = []
        mask_list = []
        gap_list = []
        sdt_list = []
        gapsdt_list = []
        final_list = []

        for i in range(N):
            h, w = img_metas[i]['img_shape'][:2]
            _h, _w = img_metas[i]['ori_shape'][:2]
            scale = img_metas[i]['scale_factor'][0]
            if self.loss_mask.use_sigmoid:
                masks_i = mask_pred[i].sigmoid()
            else:
                masks_i = F.softmax(mask_pred[i], dim=0)[1:, ...]
            masks_i = F.interpolate(masks_i.unsqueeze(0), size=(H, W))  #input_shape 1*1*H*W
            masks_i = masks_i[:, :, :h, :w].cpu().squeeze(0).squeeze(0)    #img_shape
            sdts_i = torch.zeros_like(masks_i)
            gapsdt_i = torch.zeros_like(masks_i)
            if self.loss_contour:
                sdts_i = model_outs[2][i]
                sdts_i = F.interpolate(sdts_i.unsqueeze(0), size=(H, W))  #input_shape 1*1*H*W
                sdts_i = sdts_i[:, :, :h, :w].cpu().squeeze(0).squeeze(0)    #img_shape
                gapsdt_i = model_outs[3][i]
                gapsdt_i = F.interpolate(gapsdt_i.unsqueeze(0), size=(H, W))  #input_shape 1*1*H*W
                gapsdt_i = gapsdt_i[:, :, :h, :w].cpu().squeeze(0).squeeze(0)    #img_shape

            # gaps
            gaps_i = torch.zeros_like(masks_i)
            if self.loss_gap:
                gaps_i = gap_pred[i].sigmoid() if self.loss_gap.use_sigmoid else F.softmax(gap_pred[i], dim=0)[1:, ...]
                gaps_i = F.interpolate(gaps_i.unsqueeze(0), size=(H, W))  #input_shape 1*1*H*W
                gaps_i = gaps_i[:, :, :h, :w].cpu().squeeze(0).squeeze(0)   #img_shape
#             import pdb; pdb.set_trace()
            final_i, bboxes = self.post_processing(masks_i, gaps_i, sdts_i, gapsdt_i, min_area=h*w*0.0005, scale=scale, thres=0.5)
            outs.append(bboxes)
            mask_list.append(masks_i)
            gap_list.append(gaps_i)
            sdt_list.append(sdts_i)
            gapsdt_list.append(gapsdt_i)
            final_list.append(final_i)
        return outs, mask_list, gap_list, sdt_list, gapsdt_list, final_list
