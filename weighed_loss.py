import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms


class Lambda(nn.Module):
    """Wraps a callable in an :class:`nn.Module` without registering it."""

    def __init__(self, func):
        super().__init__()
        object.__setattr__(self, 'forward', func)

    def extra_repr(self):
        return getattr(self.forward, '__name__', type(self.forward).__name__) + '()'


class WeightedLoss(nn.ModuleList):
    """A weighted combination of multiple loss functions."""

    def __init__(self, losses, weights, verbose=False):
        super().__init__()
        for loss in losses:
            self.append(loss if isinstance(loss, nn.Module) else Lambda(loss))
        self.weights = weights
        self.verbose = verbose

    def _print_losses(self, losses):
        for i, loss in enumerate(losses):
            print(f'({i}) {type(self[i]).__name__}: {loss.item()}')

    def forward(self, *args, **kwargs):
        losses = []
        for loss, weight in zip(self, self.weights):
            losses.append(loss(*args, **kwargs) * weight)
        if self.verbose:
            self._print_losses(losses)
        return sum(losses)


class TVLoss(nn.Module):
    """Total variation loss (Lp penalty on image gradient magnitude).
    """

    def __init__(self, p, reduction='mean', eps=1e-8):
        super().__init__()
        if p not in {1, 2}:
            raise ValueError('p must be 1 or 2')
        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.p = p
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target=None):
        input = F.pad(input, (0, 1, 0, 1), 'replicate')
        x_diff = input[..., :-1, :-1] - input[..., :-1, 1:]
        y_diff = input[..., :-1, :-1] - input[..., 1:, :-1]
        diff = x_diff**2 + y_diff**2
        if self.p == 1:
            diff = (diff + self.eps).mean(dim=1, keepdims=True).sqrt()
        if self.reduction == 'mean':
            return diff.mean()
        if self.reduction == 'sum':
            return diff.sum()
        return diff


class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.
    """

    models = {'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, model='vgg19', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = self.models[model](pretrained=True).features[:layer+1]
        self.model.eval()
        self.model.requires_grad_(False)

    def get_features(self, input):
        return self.model(self.normalize(input))
        # return self.model(input)

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)
