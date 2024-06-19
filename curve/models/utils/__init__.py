from torch.nn.functional import interpolate
import torch
from numpy import cumsum

__all__ = ['upsample_like', 'crop_like',
           'get_params', 'one_hot', '_split_mask', 'gradient_sobel', 'Heaviside', 'Dirac']

def upsample_like(logits, z):
    def _upsample_like(logits, z):
        if len(z.shape) == 4:
            _, _, h, w = z.shape
        else:  # take last two dimensions
            h, w = z.shape[-2:]

        _, _, lh, lw = logits.shape
        if (h != lh or w != lw):
            logits = interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        return logits

    if len(logits.shape) == 3:
        logits = logits[:, None, :, :]
        logits = _upsample_like(logits, z)
        return logits[:, 0, :, :]
    else:
        return _upsample_like(logits, z)


def crop_like(tensor, ref):
    h, w = ref.shape[-2:]
    ndim = len(tensor.shape)
    if ndim == 3:
        return tensor[:, :h, :w]
    elif ndim == 4:
        return tensor[:, :, :h, :w]
    else:
        raise ValueError('Invalid input dimension {}'.format(ndim))


def _split_mask(mask, _splits):
    # split mask_pred into lists
    _splits = cumsum([0] + _splits)
    masks = []
    for i in range(len(_splits) - 1):
        masks.append(mask[_splits[i]: _splits[i + 1]])
    return masks


def one_hot(tensor, num_class):
    ndim = len(tensor.shape)
    assert ndim in [2, 3]
    if ndim == 3:  # [b, h, w]
        b, h, w = tensor.shape
        # [b, class, h, w]
        hot = torch.arange(0, num_class)[None, :, None, None].repeat([b, 1, h, w])
        hot = hot.to(tensor.device)
        hot = hot == tensor[:, None]
    else:  # [h, w]
        h, w = tensor.shape
        hot = torch.arange(0, num_class)[:, None, None].repeat([1, h, w])
        hot = hot.to(tensor.device)
        hot = hot == tensor[None]
    return hot


def get_params(model, prefixs, suffixes, exclude=None):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    for name, module in model.named_modules():
        for prefix in prefixs:
            if name == prefix:
                for n, p in module.named_parameters():
                    n = '.'.join([name, n])
                    if type(exclude) == list and n in exclude:
                        continue
                    if type(exclude) == str and exclude in n:
                        continue

                    for suffix in suffixes:
                        if (n.split('.')[-1].startswith(suffix) or n.endswith(suffix)) and p.requires_grad:
                            # print(n, end=', ')
                            yield p
                break
