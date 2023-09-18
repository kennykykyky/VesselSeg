from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.apps.detection.utils.anchor_utils import AnchorGenerator
from monai.apps.detection.utils.ATSS_matcher import ATSSMatcher
from monai.apps.detection.utils.box_coder import BoxCoder
from monai.apps.detection.utils.box_selector import BoxSelector
from monai.apps.detection.utils.detector_utils import check_training_targets, preprocess_images
from monai.apps.detection.utils.hard_negative_sampler import HardNegativeSampler
from monai.apps.detection.utils.predict_utils import ensure_dict_value_to_list_, predict_with_inferer
from monai.data.box_utils import box_iou
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import resnet
from monai.utils import BlendMode, PytorchPadMode, ensure_tuple_rep, optional_import

from core.losses.boxLoss import BoxLoss

BalancedPositiveNegativeSampler, _ = optional_import(
    "torchvision.models.detection._utils", name="BalancedPositiveNegativeSampler"
)
Matcher, _ = optional_import("torchvision.models.detection._utils", name="Matcher")


class RetinaNetDetectorModel(nn.Module):
    """
    Retinanet detector, expandable to other one stage anchor based box detectors in the future.
    An example of construction can found in the source code of
    :func:`~monai.apps.detection.networks.retinanet_detector.retinanet_resnet50_fpn_detector` .

    The input to the model is expected to be a list of tensors, each of shape (C, H, W) or  (C, H, W, D),
    one for each image, and should be in 0-1 range. Different images can have different sizes.
    Or it can also be a Tensor sized (B, C, H, W) or  (B, C, H, W, D). In this case, all images have same size.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

    - boxes (``FloatTensor[N, 4]`` or ``FloatTensor[N, 6]``): the ground-truth boxes in ``StandardMode``, i.e.,
      ``[xmin, ymin, xmax, ymax]`` or ``[xmin, ymin, zmin, xmax, ymax, zmax]`` format,
      with ``0 <= xmin < xmax <= H``, ``0 <= ymin < ymax <= W``, ``0 <= zmin < zmax <= D``.

    The model returns a Dict[str, Tensor] during training, containing the regression losses.
    When saving the model, only self.network contains trainable parameters and needs to be saved.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:

    - boxes (``FloatTensor[N, 4]`` or ``FloatTensor[N, 6]``): the predicted boxes in ``StandardMode``, i.e.,
      ``[xmin, ymin, xmax, ymax]`` or ``[xmin, ymin, zmin, xmax, ymax, zmax]`` format,
      with ``0 <= xmin < xmax <= H``, ``0 <= ymin < ymax <= W``, ``0 <= zmin < zmax <= D``.
    - labels_scores (Tensor[N]): the scores for each prediction

    Args:
        network: a network that takes an image Tensor sized (B, C, H, W) or (B, C, H, W, D) as input
            and outputs a dictionary Dict[str, List[Tensor]] or Dict[str, Tensor].
        anchor_generator: anchor generator.
        box_overlap_metric: func that compute overlap between two sets of boxes, default is Intersection over Union (IoU).
        debug: whether to print out internal parameters, used for debugging and parameter tuning.

    Notes:

        Input argument ``network`` can be a monai.apps.detection.networks.retinanet_network.RetinaNet(*) object,
        but any network that meets the following rules is a valid input ``network``.

        1. It should have attributes including spatial_dims, box_reg_key, num_anchors, size_divisible.

            - spatial_dims (int) is the spatial dimension of the network, we support both 2D and 3D.
            - size_divisible (int or Sequence[int]) is the expectation on the input image shape.
              The network needs the input spatial_size to be divisible by size_divisible, length should be 2 or 3.
            - box_reg_key (str) is the key to represent box regression in the output dict.
            - num_anchors (int) is the number of anchor shapes at each location. it should equal to
              ``self.anchor_generator.num_anchors_per_location()[0]``.

            If network does not have these attributes, user needs to provide them for the detector.

        2. Its input should be an image Tensor sized (B, C, H, W) or (B, C, H, W, D).

        3. About its output ``head_outputs``, it should be either a list of tensors or a dictionary of str: List[Tensor]:

            - If it is a dictionary, it needs to have at least one key:
              ``network.box_reg_key``, representing predicted box regression maps.
              ``head_outputs[network.box_reg_key]`` should be List[Tensor] or Tensor. Each Tensor represents
              box regression map at one resolution level,
              sized (B, 2*spatial_dims*num_anchors, H_i, W_i)or (B, 2*spatial_dims*num_anchors, H_i, W_i, D_i).
              ``len(head_outputs[network.cls_key]) == len(head_outputs[network.box_reg_key])``.
            - If it is a list of N tensors, the N tensors should be the predicted box regression maps.

    Example:

        .. code-block:: python

            # define a naive network
            import torch
            class NaiveNet(torch.nn.Module):
                def __init__(self, spatial_dims: int):
                    super().__init__()
                    self.spatial_dims = spatial_dims
                    self.size_divisible = 2
                    self.box_reg_key = "box_reg"
                    self.num_anchors = 1
                def forward(self, images: torch.Tensor):
                    spatial_size = images.shape[-self.spatial_dims:]
                    out_spatial_size = tuple(s//self.size_divisible for s in spatial_size)  # half size of input
                    out_box_reg_shape = (images.shape[0],2*self.spatial_dims*self.num_anchors) + out_spatial_size
                    return {self.box_reg_key: [torch.randn(out_box_reg_shape)]}

            # create a RetinaNetDetector detector
            spatial_dims = 3
            anchor_generator = monai.apps.detection.utils.anchor_utils.AnchorGeneratorWithAnchorShape(
                feature_map_scales=(1, ), base_anchor_shapes=((8,) * spatial_dims)
            )
            net = NaiveNet(spatial_dims)
            detector = RetinaNetDetector(net, anchor_generator)

            # only detector.network may contain trainable parameters.
            optimizer = torch.optim.SGD(
                detector.network.parameters(),
                1e-3,
                momentum=0.9,
                weight_decay=3e-5,
                nesterov=True,
            )
            torch.save(detector.network.state_dict(), 'model.pt')  # save model
            detector.network.load_state_dict(torch.load('model.pt'))  # load model
    """

    def __init__(
        self,
        network: nn.Module,
        anchor_generator: AnchorGenerator,
        box_overlap_metric: Callable = box_iou,
        spatial_dims: int | None = None,  # used only when network.spatial_dims does not exist
        size_divisible: Sequence[int] | int = 1,  # used only when network.size_divisible does not exist
        box_reg_key: str = "box_regression",  # used only when network.box_reg_key does not exist
        debug: bool = False,
    ):
        super().__init__()

        self.network = network
        # network attribute
        self.spatial_dims = self.get_attribute_from_network("spatial_dims", default_value=spatial_dims)

        self.size_divisible = self.get_attribute_from_network("size_divisible", default_value=size_divisible)
        self.size_divisible = ensure_tuple_rep(self.size_divisible, self.spatial_dims)
        # keys for the network output
        self.box_reg_key = self.get_attribute_from_network("box_reg_key", default_value=box_reg_key)

        # check if anchor_generator matches with network
        self.anchor_generator = anchor_generator
        self.num_anchors_per_loc = self.anchor_generator.num_anchors_per_location()[0]
        network_num_anchors = self.get_attribute_from_network("num_anchors", default_value=self.num_anchors_per_loc)
        if self.num_anchors_per_loc != network_num_anchors:
            raise ValueError(
                f"Number of feature map channels ({network_num_anchors}) "
                f"should match with number of anchors at each location ({self.num_anchors_per_loc})."
            )
        # if new coming input images has same shape with
        # self.previous_image_shape, there is no need to generate new anchors.
        self.anchors: list[Tensor] | None = None
        self.previous_image_shape: Any | None = None

        self.box_overlap_metric = box_overlap_metric
        self.debug = debug

        # default setting for training
        self.fg_bg_sampler: Any | None = None

        # default setting for both training and inference
        # can be updated by self.set_box_coder_weights(*)
        self.box_coder = BoxCoder(weights=(1.0,) * 2 * self.spatial_dims)

        # box loss function
        self.box_loss = BoxLoss(self.box_coder)

        # default keys in the ground truth targets and predicted boxes,
        # can be updated by self.set_target_keys(*)
        self.target_box_key = "boxes"

        # default setting for inference,
        # can be updated by self.set_sliding_window_inferer(*)
        self.inferer: SlidingWindowInferer | None = None
        # can be updated by self.set_box_selector_parameters(*),
        self.box_selector = BoxSelector(
            box_overlap_metric=self.box_overlap_metric,
            score_thresh=0.05,
            topk_candidates_per_level=1000,
            nms_thresh=0.5,
            detections_per_img=300,
            apply_sigmoid=True,
        )

    def get_attribute_from_network(self, attr_name, default_value=None):
        if hasattr(self.network, attr_name):
            return getattr(self.network, attr_name)
        elif default_value is not None:
            return default_value
        else:
            raise ValueError(f"network does not have attribute {attr_name}, please provide it in the detector.")

    def set_box_coder_weights(self, weights: tuple[float]) -> None:
        """
        Set the weights for box coder.

        Args:
            weights: a list/tuple with length of 2*self.spatial_dims

        """
        if len(weights) != 2 * self.spatial_dims:
            raise ValueError(f"len(weights) should be {2 * self.spatial_dims}, got weights={weights}.")
        self.box_coder = BoxCoder(weights=weights)

    def set_regular_matcher(
        self, fg_iou_thresh: float, bg_iou_thresh: float, allow_low_quality_matches: bool = True
    ) -> None:
        """
        Using for training. Set torchvision matcher that matches anchors with ground truth boxes.

        Args:
            fg_iou_thresh: foreground IoU threshold for Matcher, considered as matched if IoU > fg_iou_thresh
            bg_iou_thresh: background IoU threshold for Matcher, considered as not matched if IoU < bg_iou_thresh
            allow_low_quality_matches: if True, produce additional matches
                for predictions that have only low-quality match candidates.
        """
        if fg_iou_thresh < bg_iou_thresh:
            raise ValueError(
                "Require fg_iou_thresh >= bg_iou_thresh. "
                f"Got fg_iou_thresh={fg_iou_thresh}, bg_iou_thresh={bg_iou_thresh}."
            )
        self.proposal_matcher = Matcher(
            fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=allow_low_quality_matches
        )

    def set_atss_matcher(self, num_candidates: int = 4, center_in_gt: bool = False) -> None:
        """
        Using for training. Set ATSS matcher that matches anchors with ground truth boxes

        Args:
            num_candidates: number of positions to select candidates from.
                Smaller value will result in a higher matcher threshold and less matched candidates.
            center_in_gt: If False (default), matched anchor center points do not need
                to lie withing the ground truth box. Recommend False for small objects.
                If True, will result in a strict matcher and less matched candidates.
        """
        self.proposal_matcher = ATSSMatcher(num_candidates, self.box_overlap_metric, center_in_gt, debug=self.debug)

    def set_hard_negative_sampler(
        self, batch_size_per_image: int, positive_fraction: float, min_neg: int = 1, pool_size: float = 10
    ) -> None:
        """
        Using for training. Set hard negative sampler that samples part of the anchors for training.

        HardNegativeSampler is used to suppress false positive rate in classification tasks.
        During training, it select negative samples with high prediction scores.

        Args:
            batch_size_per_image: number of elements to be selected per image
            positive_fraction: percentage of positive elements in the selected samples
            min_neg: minimum number of negative samples to select if possible.
            pool_size: when we need ``num_neg`` hard negative samples, they will be randomly selected from
                ``num_neg * pool_size`` negative samples with the highest prediction scores.
                Larger ``pool_size`` gives more randomness, yet selects negative samples that are less 'hard',
                i.e., negative samples with lower prediction scores.
        """
        self.fg_bg_sampler = HardNegativeSampler(
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
            min_neg=min_neg,
            pool_size=pool_size,
        )

    def set_balanced_sampler(self, batch_size_per_image: int, positive_fraction: float) -> None:
        """
        Using for training. Set torchvision balanced sampler that samples part of the anchors for training.

        Args:
            batch_size_per_image: number of elements to be selected per image
            positive_fraction: percentage of positive elements per batch

        """
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image=batch_size_per_image, positive_fraction=positive_fraction
        )

    def set_sliding_window_inferer(
        self,
        roi_size: Sequence[int] | int,
        sw_batch_size: int = 1,
        overlap: float = 0.5,
        mode: BlendMode | str = BlendMode.CONSTANT,
        sigma_scale: Sequence[float] | float = 0.125,
        padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: torch.device | str | None = None,
        device: torch.device | str | None = None,
        progress: bool = False,
        cache_roi_weight_map: bool = False,
    ) -> None:
        """
        Define sliding window inferer and store it to self.inferer.
        """
        self.inferer = SlidingWindowInferer(
            roi_size,
            sw_batch_size,
            overlap,
            mode,
            sigma_scale,
            padding_mode,
            cval,
            sw_device,
            device,
            progress,
            cache_roi_weight_map,
        )

    def set_box_selector_parameters(
        self,
        score_thresh: float = 0.05,
        topk_candidates_per_level: int = 1000,
        nms_thresh: float = 0.5,
        detections_per_img: int = 300,
        apply_sigmoid: bool = True,
    ) -> None:
        """
        Using for inference. Set the parameters that are used for box selection during inference.
        The box selection is performed with the following steps:

        #. For each level, discard boxes with scores less than self.score_thresh.
        #. For each level, keep boxes with top self.topk_candidates_per_level scores.
        #. For the whole image, perform non-maximum suppression (NMS) on boxes, with overlapping threshold nms_thresh.
        #. For the whole image, keep boxes with top self.detections_per_img scores.

        Args:
            score_thresh: no box with scores less than score_thresh will be kept
            topk_candidates_per_level: max number of boxes to keep for each level
            nms_thresh: box overlapping threshold for NMS
            detections_per_img: max number of boxes to keep for each image
        """

        self.box_selector = BoxSelector(
            box_overlap_metric=self.box_overlap_metric,
            apply_sigmoid=apply_sigmoid,
            score_thresh=score_thresh,
            topk_candidates_per_level=topk_candidates_per_level,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
        )

    def forward(
        self,
        input_images: list[Tensor] | Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        use_inferer: bool = False,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """
        Returns a dict of losses during training, or a list predicted dict of boxes and labels during inference.

        Args:
            input_images: The input to the model is expected to be a list of tensors, each of shape (C, H, W) or  (C, H, W, D),
                one for each image, and should be in 0-1 range. Different images can have different sizes.
                Or it can also be a Tensor sized (B, C, H, W) or  (B, C, H, W, D). In this case, all images have same size.
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image (optional).
            use_inferer: whether to use self.inferer, a sliding window inferer, to do the inference.
                If False, will simply forward the network.
                If True, will use self.inferer, and requires
                ``self.set_sliding_window_inferer(*args)`` to have been called before.

        Return:
            If training mode, will return a dict with at least two keys,
            including self.cls_key and self.box_reg_key, representing classification loss and box regression loss.

            If evaluation mode, will return a list of detection results.
            Each element corresponds to an images in ``input_images``, is a dict with at least three keys,
            including self.target_box_key, self.target_label_key, self.pred_score_key,
            representing predicted boxes, classification labels, and classification scores.

        """
        # 1. Check if input arguments are valid
        if self.training:
            targets = check_training_targets(
                input_images, targets, self.spatial_dims, self.target_label_key, self.target_box_key
            )
            self._check_detector_training_components()

        # 2. Pad list of images to a single Tensor `images` with spatial size divisible by self.size_divisible.
        # image_sizes stores the original spatial_size of each image before padding.
        images, image_sizes = preprocess_images(input_images, self.spatial_dims, self.size_divisible)

        # 3. Generate network outputs. Use inferer only in evaluation mode.
        if self.training or (not use_inferer):
            head_outputs = self.network(images)
            if isinstance(head_outputs, (tuple, list)):
                tmp_dict = {}
                tmp_dict[self.box_reg_key] = head_outputs[len(head_outputs) // 2 :]
                head_outputs = tmp_dict
            else:
                # ensure head_outputs is Dict[str, List[Tensor]]
                ensure_dict_value_to_list_(head_outputs)
        else:
            if self.inferer is None:
                raise ValueError(
                    "`self.inferer` is not defined." "Please refer to function self.set_sliding_window_inferer(*)."
                )
            head_outputs = predict_with_inferer(
                images, self.network, keys=[self.box_reg_key], inferer=self.inferer
            )

        # 4. Generate anchors and store it in self.anchors: List[Tensor]
        self.generate_anchors(images, head_outputs)
        # num_anchor_locs_per_level: List[int], list of HW or HWD for each level
        # num_anchor_locs_per_level = [x.shape[2:].numel() for x in head_outputs[self.cls_key]]

        # 5. Reshape and concatenate head_outputs values from List[Tensor] to Tensor
        # head_outputs, originally being Dict[str, List[Tensor]], will be reshaped to Dict[str, Tensor]

        # reshape to Tensor sized(B, sum(HWA), 2* self.spatial_dims) for self.box_reg_key
        # A = self.num_anchors_per_loc
        head_outputs[self.box_reg_key] = self._reshape_maps(head_outputs[self.box_reg_key])

        # 6(1). If during training, return losses
        if self.training:
            losses = self.loss_box(head_outputs, targets, self.anchors, num_anchor_locs_per_level)  # type: ignore
            return losses

        # 6(2). If during inference, return detection results
        detections = self.postprocess_detections(
            head_outputs, self.anchors, image_sizes, num_anchor_locs_per_level  # type: ignore
        )
        return detections

    def _check_detector_training_components(self):
        """
        Check if self.proposal_matcher and self.fg_bg_sampler have been set for training.
        """
        if not hasattr(self, "proposal_matcher"):
            raise AttributeError(
                "Matcher is not set. Please refer to self.set_regular_matcher(*) or self.set_atss_matcher(*)."
            )
        if self.fg_bg_sampler is None and self.debug:
            warnings.warn(
                "No balanced sampler is used. Negative samples are likely to "
                "be much more than positive samples. Please set balanced samplers with self.set_balanced_sampler(*) "
                "or self.set_hard_negative_sampler(*), "
                "or set classification loss function as Focal loss with self.set_cls_loss(*)"
            )

    def generate_anchors(self, images: Tensor, head_outputs: dict[str, list[Tensor]]) -> None:
        """
        Generate anchors and store it in self.anchors: List[Tensor].
        We generate anchors only when there is no stored anchors,
        or the new coming images has different shape with self.previous_image_shape

        Args:
            images: input images, a (B, C, H, W) or (B, C, H, W, D) Tensor.
            head_outputs: head_outputs. ``head_output_reshape[self.cls_key]`` is a Tensor
              sized (B, sum(HW(D)A), self.num_classes). ``head_output_reshape[self.box_reg_key]`` is a Tensor
              sized (B, sum(HW(D)A), 2*self.spatial_dims)
        """
        if (self.anchors is None) or (self.previous_image_shape != images.shape):
            self.anchors = self.anchor_generator(images, head_outputs[self.cls_key])  # List[Tensor], len = batchsize
            self.previous_image_shape = images.shape

    def _reshape_maps(self, result_maps: list[Tensor]) -> Tensor:
        """
        Concat network output map list to a single Tensor.
        This function is used in both training and inference.

        Args:
            result_maps: a list of Tensor, each Tensor is a (B, num_channel*A, H, W) or (B, num_channel*A, H, W, D) map.
                A = self.num_anchors_per_loc

        Return:
            reshaped and concatenated result, sized (B, sum(HWA), num_channel) or (B, sum(HWDA), num_channel)
        """
        all_reshaped_result_map = []

        for result_map in result_maps:
            batch_size = result_map.shape[0]
            num_channel = result_map.shape[1] // self.num_anchors_per_loc
            spatial_size = result_map.shape[-self.spatial_dims :]

            # reshaped_result_map will become (B, A, num_channel, H, W) or (B, A, num_channel, H, W, D)
            # A = self.num_anchors_per_loc
            view_shape = (batch_size, -1, num_channel) + spatial_size
            reshaped_result_map = result_map.view(view_shape)

            # permute output to (B, H, W, A, num_channel) or (B, H, W, D, A, num_channel)
            if self.spatial_dims == 2:
                reshaped_result_map = reshaped_result_map.permute(0, 3, 4, 1, 2)
            elif self.spatial_dims == 3:
                reshaped_result_map = reshaped_result_map.permute(0, 3, 4, 5, 1, 2)
            else:
                ValueError("Images can only be 2D or 3D.")

            # reshaped_result_map will become (B, HWA, num_channel) or (B, HWDA, num_channel)
            reshaped_result_map = reshaped_result_map.reshape(batch_size, -1, num_channel)

            if torch.isnan(reshaped_result_map).any() or torch.isinf(reshaped_result_map).any():
                if torch.is_grad_enabled():
                    raise ValueError("Concatenated result is NaN or Inf.")
                else:
                    warnings.warn("Concatenated result is NaN or Inf.")

            all_reshaped_result_map.append(reshaped_result_map)

        return torch.cat(all_reshaped_result_map, dim=1)

    def postprocess_detections(
        self,
        head_outputs_reshape: dict[str, Tensor],
        anchors: list[Tensor],
        image_sizes: list[list[int]],
        num_anchor_locs_per_level: Sequence[int],
        need_sigmoid: bool = True,
    ) -> list[dict[str, Tensor]]:
        """
        Postprocessing to generate detection result from classification logits and box regression.
        Use self.box_selector to select the final output boxes for each image.

        Args:
            head_outputs_reshape: reshaped head_outputs. ``head_output_reshape[self.cls_key]`` is a Tensor
              sized (B, sum(HW(D)A), self.num_classes). ``head_output_reshape[self.box_reg_key]`` is a Tensor
              sized (B, sum(HW(D)A), 2*self.spatial_dims)
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            anchors: a list of Tensor. Each Tensor represents anchors for each image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.

        Return:
            a list of dict, each dict corresponds to detection result on image.
        """

        # recover level sizes, HWA or HWDA for each level
        num_anchors_per_level = [
            num_anchor_locs * self.num_anchors_per_loc for num_anchor_locs in num_anchor_locs_per_level
        ]

        # split outputs per level
        split_head_outputs: dict[str, list[Tensor]] = {}
        for k in head_outputs_reshape:
            split_head_outputs[k] = list(head_outputs_reshape[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]  # List[List[Tensor]]

        box_regression = split_head_outputs[self.box_reg_key]  # List[Tensor], each sized (B, HWA, 2*spatial_dims)

        num_images = len(image_sizes)  # B

        detections: list[dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [
                br[index] for br in box_regression
            ]  # List[Tensor], each sized (HWA, 2*spatial_dims)
            logits_per_image = [cl[index] for cl in class_logits]  # List[Tensor], each sized (HWA, self.num_classes)
            anchors_per_image, img_spatial_size = split_anchors[index], image_sizes[index]
            # decode box regression into boxes
            boxes_per_image = [
                self.box_coder.decode_single(b.to(torch.float32), a).to(compute_dtype)
                for b, a in zip(box_regression_per_image, anchors_per_image)
            ]  # List[Tensor], each sized (HWA, 2*spatial_dims)

            selected_boxes, selected_scores, selected_labels = self.box_selector.select_boxes_per_image(
                boxes_per_image, logits_per_image, img_spatial_size
            )

            detections.append(
                {
                    self.target_box_key: selected_boxes,  # Tensor, sized (N, 2*spatial_dims)
                    self.pred_score_key: selected_scores,  # Tensor, sized (N, )
                    self.target_label_key: selected_labels,  # Tensor, sized (N, )
                }
            )

        return detections

    def loss_box(
        self,
        head_outputs_reshape: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        anchors: list[Tensor],
        num_anchor_locs_per_level: Sequence[int],
    ) -> dict[str, Tensor]:
        """
        Compute losses.

        Args:
            head_outputs_reshape: reshaped head_outputs. ``head_output_reshape[self.cls_key]`` is a Tensor
              sized (B, sum(HW(D)A), self.num_classes). ``head_output_reshape[self.box_reg_key]`` is a Tensor
              sized (B, sum(HW(D)A), 2*self.spatial_dims)
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            anchors: a list of Tensor. Each Tensor represents anchors for each image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.

        Return:
            a dict of several kinds of losses.
        """
        matched_idxs = self.compute_anchor_matched_idxs(anchors, targets, num_anchor_locs_per_level)
        losses_box_regression = self.box_loss(
            head_outputs_reshape[self.box_reg_key], targets, anchors, matched_idxs
        )
        return losses_box_regression

    def compute_anchor_matched_idxs(
        self, anchors: list[Tensor], targets: list[dict[str, Tensor]], num_anchor_locs_per_level: Sequence[int]
    ) -> list[Tensor]:
        """
        Compute the matched indices between anchors and ground truth (gt) boxes in targets.
        output[k][i] represents the matched gt index for anchor[i] in image k.
        Suppose there are M gt boxes for image k. The range of it output[k][i] value is [-2, -1, 0, ..., M-1].
        [0, M - 1] indicates this anchor is matched with a gt box,
        while a negative value indicating that it is not matched.

        Args:
            anchors: a list of Tensor. Each Tensor represents anchors for each image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            num_anchor_locs_per_level: each element represents HW or HWD at this level.


        Return:
            a list of matched index `matched_idxs_per_image` (Tensor[int64]), Tensor sized (sum(HWA),) or (sum(HWDA),).
            Suppose there are M gt boxes. `matched_idxs_per_image[i]` is a matched gt index in [0, M - 1]
            or a negative value indicating that anchor i could not be matched.
            BELOW_LOW_THRESHOLD = -1, BETWEEN_THRESHOLDS = -2
        """
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # anchors_per_image: Tensor, targets_per_image: Dice[str, Tensor]
            if targets_per_image[self.target_box_key].numel() == 0:
                # if no GT boxes
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            # matched_idxs_per_image (Tensor[int64]): Tensor sized (sum(HWA),) or (sum(HWDA),)
            # Suppose there are M gt boxes. matched_idxs_per_image[i] is a matched gt index in [0, M - 1]
            # or a negative value indicating that anchor i could not be matched.
            # BELOW_LOW_THRESHOLD = -1, BETWEEN_THRESHOLDS = -2
            if isinstance(self.proposal_matcher, Matcher):
                # if torchvision matcher
                match_quality_matrix = self.box_overlap_metric(
                    targets_per_image[self.target_box_key].to(anchors_per_image.device), anchors_per_image
                )
                matched_idxs_per_image = self.proposal_matcher(match_quality_matrix)
            elif isinstance(self.proposal_matcher, ATSSMatcher):
                # if monai ATSS matcher
                match_quality_matrix, matched_idxs_per_image = self.proposal_matcher(
                    targets_per_image[self.target_box_key].to(anchors_per_image.device),
                    anchors_per_image,
                    num_anchor_locs_per_level,
                    self.num_anchors_per_loc,
                )
            else:
                raise NotImplementedError(
                    "Currently support torchvision Matcher and monai ATSS matcher. Other types of matcher not supported. "
                    "Please override self.compute_anchor_matched_idxs(*) for your own matcher."
                )

            if self.debug:
                print(f"Max box overlap between anchors and gt boxes: {torch.max(match_quality_matrix,dim=1)[0]}.")

            if torch.max(matched_idxs_per_image) < 0:
                warnings.warn(
                    f"No anchor is matched with GT boxes. Please adjust matcher setting, anchor setting,"
                    " or the network setting to change zoom scale between network output and input images."
                    f"GT boxes are {targets_per_image[self.target_box_key]}."
                )

            matched_idxs.append(matched_idxs_per_image)
        return matched_idxs


def retinanet_resnet50_fpn_detector(
    num_classes: int,
    anchor_generator: AnchorGenerator,
    returned_layers: Sequence[int] = (1, 2, 3),
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> RetinaNetDetectorModel:
    """
    Returns a RetinaNet detector using a ResNet-50 as backbone, which can be pretrained
    from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`
    _.

    Args:
        num_classes: number of output classes of the model (excluding the background).
        anchor_generator: AnchorGenerator,
        returned_layers: returned layers to extract feature maps. Each returned layer should be in the range [1,4].
            len(returned_layers)+1 will be the number of extracted feature maps.
            There is an extra maxpooling layer LastLevelMaxPool() appended.
        pretrained: If True, returns a backbone pre-trained on 23 medical datasets
        progress: If True, displays a progress bar of the download to stderr

    Return:
        A RetinaNetDetector object with resnet50 as backbone

    Example:

        .. code-block:: python

            # define a naive network
            resnet_param = {
                "pretrained": False,
                "spatial_dims": 3,
                "n_input_channels": 2,
                "num_classes": 3,
                "conv1_t_size": 7,
                "conv1_t_stride": (2, 2, 2)
            }
            returned_layers = [1]
            anchor_generator = monai.apps.detection.utils.anchor_utils.AnchorGeneratorWithAnchorShape(
                feature_map_scales=(1, 2), base_anchor_shapes=((8,) * resnet_param["spatial_dims"])
            )
            detector = retinanet_resnet50_fpn_detector(
                **resnet_param, anchor_generator=anchor_generator, returned_layers=returned_layers
            )
    """

    backbone = resnet.resnet50(pretrained, progress, **kwargs)
    spatial_dims = len(backbone.conv1.stride)
    # number of output feature maps is len(returned_layers)+1
    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=spatial_dims,
        pretrained_backbone=pretrained,
        trainable_backbone_layers=None,
        returned_layers=returned_layers,
    )
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    size_divisible = [s * 2 * 2 ** max(returned_layers) for s in feature_extractor.body.conv1.stride]
    network = RetinaNet(
        spatial_dims=spatial_dims,
        num_classes=num_classes,
        num_anchors=num_anchors,
        feature_extractor=feature_extractor,
        size_divisible=size_divisible,
    )
    return RetinaNetDetectorModel(network, anchor_generator)