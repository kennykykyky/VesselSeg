import torch
import warnings
from torch import Tensor, nn
from monai.apps.detection.utils.box_coder import BoxCoder

class BoxLoss(nn.Module):
    def __init__(self, spatial_dims):
        super(BoxLoss, self).__init__()
        self.target_box_key = "boxes"
        self.spatial_dims = spatial_dims

        self.set_box_regression_loss(
            torch.nn.SmoothL1Loss(beta=1.0 / 9, reduction="mean"), encode_gt=True, decode_pred=False
        )  # box regression loss
        self.box_coder = BoxCoder(weights=(1.0,) * 2 * self.spatial_dims)

    def set_box_regression_loss(self, box_loss: nn.Module, encode_gt: bool, decode_pred: bool) -> None:
        """
        Using for training. Set loss for box regression.

        Args:
            box_loss: loss module for box regression
            encode_gt: if True, will encode ground truth boxes to target box regression
                before computing the losses. Should be True for L1 loss and False for GIoU loss.
            decode_pred: if True, will decode predicted box regression into predicted boxes
                before computing losses. Should be False for L1 loss and True for GIoU loss.

        Example:
            .. code-block:: python

                detector.set_box_regression_loss(
                    torch.nn.SmoothL1Loss(beta=1.0 / 9, reduction="mean"),
                    encode_gt = True, decode_pred = False
                )
                detector.set_box_regression_loss(
                    monai.losses.giou_loss.BoxGIoULoss(reduction="mean"),
                    encode_gt = False, decode_pred = True
                )
        """
        self.box_loss_func = box_loss
        self.encode_gt = encode_gt
        self.decode_pred = decode_pred

    def forward(
        self,
        box_regression: Tensor,
        targets: list[dict[str, Tensor]],
        anchors: list[Tensor],
        matched_idxs: list[Tensor],
    ) -> Tensor:
        """
        Compute box regression losses.

        Args:
            box_regression: box regression results, sized (B, sum(HWA), 2*self.spatial_dims)
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            anchors: a list of Tensor. Each Tensor represents anchors for each image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.
            matched_idxs: a list of matched index. each element is sized (sum(HWA),) or  (sum(HWDA),)

        Return:
            box regression losses.
        """
        total_box_regression_list = []
        total_target_regression_list = []

        for targets_per_image, box_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(
            targets, box_regression, anchors, matched_idxs
        ):
            # for each image, get training samples
            decode_box_regression_per_image, matched_gt_boxes_per_image = self.get_box_train_sample_per_image(
                box_regression_per_image, targets_per_image, anchors_per_image, matched_idxs_per_image
            )
            total_box_regression_list.append(decode_box_regression_per_image)
            total_target_regression_list.append(matched_gt_boxes_per_image)

        total_box_regression = torch.cat(total_box_regression_list, dim=0)
        total_target_regression = torch.cat(total_target_regression_list, dim=0)

        if total_box_regression.shape[0] == 0:
            # if there is no training sample.
            losses = torch.tensor(0.0)
            return losses

        losses = self.box_loss_func(total_box_regression, total_target_regression).to(total_box_regression.dtype)

        return losses

    def get_box_train_sample_per_image(
        self,
        box_regression_per_image: Tensor,
        targets_per_image: dict[str, Tensor],
        anchors_per_image: Tensor,
        matched_idxs_per_image: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Get samples from one image for box regression losses computation.

        Args:
            box_regression_per_image: box regression result for one image, (sum(HWA), 2*self.spatial_dims)
            targets_per_image: a dict with at least two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            anchors_per_image: anchors of one image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.
            matched_idxs_per_image: matched index, sized (sum(HWA),) or  (sum(HWDA),)

        Return:
            paired predicted and GT samples from one image for box regression losses computation
        """

        if torch.isnan(box_regression_per_image).any() or torch.isinf(box_regression_per_image).any():
            if torch.is_grad_enabled():
                raise ValueError("NaN or Inf in predicted box regression.")
            else:
                warnings.warn("NaN or Inf in predicted box regression.")

        foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
        num_gt_box = targets_per_image[self.target_box_key].shape[0]

        # if no GT box, return empty arrays
        if num_gt_box == 0:
            return box_regression_per_image[0:0, :], box_regression_per_image[0:0, :]

        # select only the foreground boxes
        # matched GT boxes for foreground anchors
        matched_gt_boxes_per_image = targets_per_image[self.target_box_key][
            matched_idxs_per_image[foreground_idxs_per_image]
        ].to(box_regression_per_image.device)
        # predicted box regression for foreground anchors
        box_regression_per_image = box_regression_per_image[foreground_idxs_per_image, :]
        # foreground anchors
        anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

        # encode GT boxes or decode predicted box regression before computing losses
        matched_gt_boxes_per_image_ = matched_gt_boxes_per_image
        box_regression_per_image_ = box_regression_per_image
        if self.encode_gt:
            matched_gt_boxes_per_image_ = self.box_coder.encode_single(matched_gt_boxes_per_image_, anchors_per_image)
        if self.decode_pred:
            box_regression_per_image_ = self.box_coder.decode_single(box_regression_per_image_, anchors_per_image)

        return box_regression_per_image_, matched_gt_boxes_per_image_