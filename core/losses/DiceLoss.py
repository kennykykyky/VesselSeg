# -*- coding: utf-8 -*-
"""
Modules for loss computation.

__author__ = "Xinzhe Luo"
__version__ = 0.1

"""

import torch.nn as nn
# import torch.nn.functional as F
import torch


class Dice(nn.Module):
    """
    The Dice coefficient computed between probabilistic predictions and the ground truth.

    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true, dims_sum, return_before_divide=False):
        assert y_pred.size() == y_true.size(), "The prediction and ground truth must be of the same size!"
        numerator = 2 * torch.sum(y_true * y_pred, dim=dims_sum)
        denominator = torch.sum(y_true + y_pred, dim=dims_sum)
        if return_before_divide:
            return numerator, denominator
        else:
            dice = (numerator + self.eps) / (denominator + self.eps)
            return dice
