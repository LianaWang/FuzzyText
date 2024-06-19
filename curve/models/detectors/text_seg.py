import mmcv
import torch
import torch.nn.functional as F
from mmdet.core import bbox_mapping
from mmdet.models import DETECTORS
from mmdet.models import build_backbone, build_head, build_neck
from mmdet.models import BaseDetector


@DETECTORS.register_module()
class Text_Seg(BaseDetector):
    """Implementation of Region Proposal Network."""

    def __init__(self,
                 backbone,
                 neck,
                 mask_head,
                 stuff_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None):
        super(Text_Seg, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck) if neck is not None else None
        self.mask_head = build_head(mask_head)
        self.stuff_head = build_head(stuff_head) if stuff_head is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(Text_Seg, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.neck is not None:
            self.neck.init_weights()
        self.mask_head.init_weights()

    def extract_feat(self, img):
        """Extract features.

        Args:
            img (torch.Tensor): Image tensor with shape (n, c, h ,w).

        Returns:
            list[torch.Tensor]: Multi-level features that may have
                different resolutions.
        """
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        if self.stuff_head is not None:
            x = self.stuff_head(x)
            
        return x

    def forward_dummy(self, img1, img2):
        """Dummy forward function."""
        x1 = self.extract_feat(img1)
        x2 = self.extract_feat(img2)
        outputs = self.mask_head(x1, x2)
        return outputs

    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_mask,
                      gt_mask_gap,
                      gt_mask_contour,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      gt_labels=None,
                      ):
        x = self.extract_feat(img)
        losses = self.mask_head.forward_train(x, img[0].size(), img_metas, gt_mask, gt_mask_gap, gt_mask_contour, gt_bboxes)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        box_list,_, _, _, _, _ = self.mask_head.simple_test_mask(x, img_metas, img.shape)
        return box_list

    def aug_test(self, imgs, img_metas, rescale=False):
        pass

    def show_result(self, data, result, dataset=None, top_k=20):
        """Show RPN proposals on the image.

        Although we assume batch size is 1, this method supports arbitrary
        batch size.
        """
        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            mmcv.imshow_bboxes(img_show, result, top_k=top_k)
