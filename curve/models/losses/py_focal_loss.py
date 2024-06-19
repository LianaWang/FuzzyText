
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import LOSSES
from mmdet.models.losses import weight_reduce_loss


# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          contour=None,
                          weight=None,
                          running_pos=-1.0,
                          running_neg=-1.0,
                          gamma=2.0,
                          alpha=0.25,
                          orig_logits=True,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid() if orig_logits else pred
    labels = target.long().clone()
    valid_inds = (labels >= 0).float()
    target = target.type_as(pred).view(pred.shape)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weights = pt.pow(gamma).detach()
    if contour is not None:
        lamb = 0.5
        weights *= (1.0 + lamb*contour.abs())
    alpha_t = 1.0 # default do not use alpha
#     beta_t = (contour+1.0).clamp(0.0)
    if alpha>0: # original balance
        alpha_t = (alpha * target + (1 - alpha) * (1 - target))
    elif alpha<0: # normalize online
        N = target.size(0)
        HW = float(target.view(N, -1).shape[-1])
        valid_num = valid_inds.view(N, -1).sum(1).mean()
        pos_weight = torch.where(labels==1, weights, weights*0.0+1e-9)
        pos_weight = float(torch.sum(pos_weight.view(N, -1), dim=-1).mean())
        neg_weight = torch.where(labels==0, weights, weights*0.0+1e-9)
        neg_weight = float(torch.sum(neg_weight.view(N, -1), dim=-1).mean())
        # update running pos and neg
        running_pos = running_pos*0.99 + pos_weight*0.01 if running_pos > 0 else pos_weight
        running_neg = running_neg*0.99 + neg_weight*0.01 if running_neg > 0 else neg_weight
        # compute scale
        alpha_p = (0.5 * valid_num / running_pos).clamp(min=0, max=100.0)
#         alpha_n = alpha_p * running_pos / running_neg # 1:1 balance
        alpha_n = (0.5 * valid_num / running_neg).clamp(min=0, max=100.0)
        alpha_t = alpha_p * target + alpha_n * (1 - target)
        alpha_t *= (HW / valid_num.clamp(min=1)).clamp(max=100.0)
    focal_weight =  alpha_t * weights #* beta_t
    loss = F.binary_cross_entropy(
        pred_sigmoid, target, reduction='none') * focal_weight
    loss = loss * valid_inds
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss, running_pos, running_neg

@LOSSES.register_module()
class PyFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 orig_logits=True,
                 use_guided=False,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(PyFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.orig_logits = orig_logits
        self.use_guided = use_guided
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.running_pos = -1.0
        self.running_neg = -1.0

    def forward(self,
                pred,
                target,
                contour,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss, running_pos, running_neg = py_sigmoid_focal_loss(
                pred,
                target,
                contour=contour if self.use_guided else None,
                weight=weight,
                running_pos=self.running_pos,
                running_neg=self.running_neg,
                gamma=self.gamma,
                alpha=self.alpha,
                orig_logits=self.orig_logits,
                reduction=reduction,
                avg_factor=avg_factor)
            self.running_pos = running_pos
            self.running_neg = running_neg
            loss_cls = self.loss_weight * loss
        else:
            raise NotImplementedError
        return loss_cls
