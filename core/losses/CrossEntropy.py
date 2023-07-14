# -*- coding: utf-8 -*-
"""
Modules for loss computation.

__author__ = "Xinzhe Luo"
__version__ = 0.1

"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class CrossEntropyLoss(nn.Module):
    """
    The cross-entropy loss computed between logits and one-hot labels.

    """
    def __init__(self, class_weight=None, ignore_index=None, **kwargs):
        """
        Initialize the cross-entropy loss.

        :param class_weight: a manual rescaling weight given to each class
        :param ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
        """
        super(CrossEntropyLoss, self).__init__()
        self.class_weight = class_weight
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """

        :param logits: tensor of shape [batch, num_classes, *vol_shape]
        :param labels: tensor of shape [batch, num_classes, *vol_shape]
        :return:
        """
        assert logits.size() == labels.size(), "The logits and labels must be of the same size!"

        if self.ignore_index is not None:
            logits = logits[:, list(range(self.ignore_index)) + list(range(self.ignore_index + 1, logits.size()[1])), ]
            labels = labels[:, list(range(self.ignore_index)) + list(range(self.ignore_index + 1, labels.size()[1])), ]


        if self.class_weight is not None:
            weight = torch.tensor(self.class_weight, dtype=torch.float32)
            assert weight.dim() == logits.size()[1]
            weight = weight.view(1, -1, *[1]*(logits.dim() - 2))
        else:
            weight = 1.

        log_prob = F.log_softmax(logits, dim=1)
        loss = - torch.mean(torch.sum(weight * labels * log_prob, dim=1))

        return loss
