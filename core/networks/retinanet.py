from monai.networks.nets import resnet
import torch

conv1_t_size = [max(7, 2 * s + 1) for s in args.conv1_t_stride]
backbone = resnet.ResNet(
    block=resnet.ResNetBottleneck,
    layers=[3, 4, 6, 3],
    block_inplanes=resnet.get_inplanes(),
    n_input_channels=args.n_input_channels,
    conv1_t_stride=args.conv1_t_stride,
    conv1_t_size=conv1_t_size,
)
feature_extractor = resnet_fpn_feature_extractor(
    backbone=backbone,
    spatial_dims=args.spatial_dims,
    pretrained_backbone=False,
    trainable_backbone_layers=None,
    returned_layers=args.returned_layers,
)
num_anchors = anchor_generator.num_anchors_per_location()[0]
size_divisible = [s * 2 * 2 ** max(args.returned_layers) for s in feature_extractor.body.conv1.stride]

class RetinaNet