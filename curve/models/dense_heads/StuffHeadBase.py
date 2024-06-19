import torch
import numpy as np
from torch import nn
from curve.models.utils import upsample_like
from mmdet.models.builder import build_loss


class StuffHeadBase(nn.Module):
    def __init__(self,
                 num_classes,
                 loss_stuff=dict(
                     type='CrossEntropyLoss', use_mask=False, loss_weight=1.0),
                 loss_name='stuff'):
        super(StuffHeadBase, self).__init__()
        if loss_stuff is not None:
            self.loss_stuff_fn = build_loss(loss_stuff)
        self.num_classes = num_classes
        self.eps = np.spacing(1)
        self.loss_name=loss_name


    @property
    def with_stuff_loss(self):
        return hasattr(self, 'loss_stuff_fn') and self.loss_stuff_fn is not None

    def loss(self, logits, gt_semantic_seg):
        logits = upsample_like(logits, gt_semantic_seg)
        logits = logits.permute((0, 2, 3, 1))
        # hard code here, minus one
        not_ignore = (gt_semantic_seg > 0)
        gt_semantic_seg_bias = torch.where(not_ignore, gt_semantic_seg - 1, torch.zeros_like(gt_semantic_seg))
        not_ignore = not_ignore.float()
        avg_factor = torch.sum(not_ignore) + self.eps

        # shit, has to convert to long
        gt_semantic_seg_bias = gt_semantic_seg_bias.long()

        loss_stuff = self.loss_stuff_fn(
            logits.reshape(-1, self.num_classes),  # => [NxHxW, C]
            gt_semantic_seg_bias.reshape(-1),
            weight=not_ignore.reshape(-1),
            avg_factor=avg_factor,
        )
        return {f'loss_{self.loss_name}': loss_stuff}

    def init_weights(self):
        return NotImplementedError

    def initialize(self):
        return self.init_weights()
