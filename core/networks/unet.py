# from core.register.SpatialTransformer import SpatialTransformer
# from core.register.VectorIntegration import VectorIntegration

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import pdb

class ConvBlk(nn.Module):

    def __init__(self, cfg, in_channels, out_channels, n_layers, info):
        super().__init__()
        self.cfg = cfg
        self.info = info
        self.n_layers = n_layers
        # [B, C, ...]
        self.output: torch.Tensor = None

        if cfg.dataset.dim == 2:
            Conv = nn.Conv2d
            Norm = nn.BatchNorm2d
            Dropout = nn.Dropout2d
        elif cfg.dataset.dim == 3:
            Conv = nn.Conv3d
            Norm = nn.BatchNorm3d
            Dropout = nn.Dropout3d
        else:
            raise NotImplementedError
        Activation = nn.LeakyReLU

        kernel_size = cfg.net[info['parent']].conv_blk.kernel_size
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0
        else:
            raise ValueError

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(
                Conv(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels,
                     kernel_size=kernel_size, padding=padding, bias=False))
            self.norms.append(Norm(out_channels, momentum=cfg.net.momentum_bns))

        if cfg.net.residual:
            if in_channels != out_channels:
                self.conv_res = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
                self.norm_res = Norm(out_channels, momentum=cfg.net.momentum_bns)

        self.activation = Activation(inplace=True)
        self.dropout = Dropout(cfg.net.dropout)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        # x: [B, C, ...]
        if self.cfg.net.residual:
            y = x.clone()
            if hasattr(self, 'conv_res'):
                y = nn.Sequential(self.conv_res, self.norm_res)(y)
                y = self.activation(y)

        for i in range(self.n_layers):
            x = self.convs[i](x)
            x = self.norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        if self.cfg.net.residual:
            x = y + x

        self.output = x
        return x


class DeconvBlk(nn.Module):

    def __init__(self, cfg, in_channels, out_channels):
        super().__init__()
        self.cfg = cfg

        if cfg.dataset.dim == 2:
            Deconv = nn.ConvTranspose2d
            Conv = nn.Conv2d
            Norm = nn.BatchNorm2d
            Dropout = nn.Dropout2d
            mode_interp = 'bilinear'
        elif cfg.dataset.dim == 3:
            Deconv = nn.ConvTranspose3d
            Conv = nn.Conv3d
            Norm = nn.BatchNorm3d
            Dropout = nn.Dropout3d
            mode_interp = 'trilinear'
        else:
            raise NotImplementedError
        Activation = nn.LeakyReLU

        if cfg.net.decoder.deconv_blk.mode == 'deconv':
            self.deconv = Deconv(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
            self.activation = Activation(inplace=True)
            self.dropout = Dropout(cfg.net.dropout)
            self.norm = Norm(out_channels, momentum=cfg.net.momentum_bns)
            nn.init.xavier_uniform_(self.deconv.weight, gain=nn.init.calculate_gain('leaky_relu'))
        elif cfg.net.decoder.deconv_blk.mode == 'resize_conv':
            kernel_size = cfg.net.decoder.deconv_blk.resize_conv.kernel_size
            if kernel_size == 3:
                padding = 1
            elif kernel_size == 1:
                padding = 0
            else:
                raise ValueError
            self.deconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode_interp, align_corners=False),
                Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )
            nn.init.xavier_uniform_(self.deconv[1].weight, gain=nn.init.calculate_gain('leaky_relu'))
        else:
            raise NotImplementedError

    def forward(self, x):
        # x: [B, C, ...]
        x = self.deconv(x)
        if self.cfg.net.decoder.deconv_blk.mode == 'deconv':
            x = self.norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        elif self.cfg.net.decoder.deconv_blk.mode != 'resize_conv':
            raise NotImplementedError

        return x


class AttentionBlk(nn.Module):

    def __init__(self, cfg, F_g, F_l, F_int):
        super().__init__()
        self.cfg = cfg
        assert cfg.dataset.dim == 2
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(1),
                                 nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class RecurrentBlk(nn.Module):

    def __init__(self, ch_out, t=2):
        super().__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                                  nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNNBlk(nn.Module):

    def __init__(self, ch_in, ch_out, t=2):
        super().__init__()
        self.RCNN = nn.Sequential(RecurrentBlk(ch_out, t=t), RecurrentBlk(ch_out, t=t))
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        x = x + x1
        self.output = x
        return x


class up_conv(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                                nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class Encoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.dataset.dim == 2:
            Pool = nn.MaxPool2d
        elif cfg.dataset.dim == 3:
            Pool = nn.MaxPool3d
        else:
            raise NotImplementedError

        self.conv_blks = nn.ModuleList()
        self.pool = Pool(2, 2)
        for i in range(cfg.net.n_levels):
            info = {'parent': 'encoder', 'level': i}

            if i == 0:
                in_channels_convblk = cfg.dataset.n_channels_img * len(cfg.dataset.mods)
                out_channels_convblk = cfg.net.n_channels_init
            else:
                if cfg.net.double_channels:
                    in_channels_convblk = cfg.net.n_channels_init * 2**i // 2
                    out_channels_convblk = in_channels_convblk * 2
                else:
                    in_channels_convblk = out_channels_convblk = cfg.net.n_channels_init

            if self.cfg.net.recurrent:
                self.conv_blks.append(RRCNNBlk(cfg, in_channels_convblk, out_channels_convblk))
            else:
                self.conv_blks.append(
                    ConvBlk(cfg, in_channels_convblk, out_channels_convblk, n_layers=cfg.net.encoder.conv_blk.n_layers,
                            info=info))

    def forward(self, x):
        for i in range(self.cfg.net.n_levels):
            x = self.conv_blks[i](x)
            if i != self.cfg.net.n_levels - 1:
                x = self.pool(x)


class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.outputs: dict[str, torch.Tensor] = {
            'seg': None, # [B, C, ...], logits
        }

        if cfg.dataset.dim == 2:
            Conv = nn.Conv2d
        elif cfg.dataset.dim == 3:
            Conv = nn.Conv3d
        else:
            raise NotImplementedError

        self.conv_blks = nn.ModuleDict()
        self.deconv_blks = nn.ModuleDict()
        if cfg.net.attention:
            self.attention_blks = nn.ModuleDict()

        level2in_channels_convblk = {}
        level2out_channels_convblk = {}
        for i in reversed(range(cfg.net.n_levels)):
            if cfg.net.double_channels:
                if cfg.net.decoder.merge_skip == 'sum':
                    in_channels_convblk = cfg.net.n_channels_init * 2**i
                elif cfg.net.decoder.merge_skip == 'concat':
                    if i == cfg.net.n_levels - 1:
                        in_channels_convblk = cfg.net.n_channels_init * 2**i
                    else:
                        in_channels_convblk = cfg.net.n_channels_init * 2**(i + 1)
                else:
                    raise NotImplementedError
                out_channels_convblk = cfg.net.n_channels_init * 2**(i - 1)
            else:
                if cfg.net.decoder.merge_skip == 'sum':
                    in_channels_convblk = cfg.net.n_channels_init
                elif cfg.net.decoder.merge_skip == 'concat':
                    if i == cfg.net.n_levels - 1:
                        in_channels_convblk = cfg.net.n_channels_init
                    else:
                        in_channels_convblk = cfg.net.n_channels_init * 2
                else:
                    raise NotImplementedError
                out_channels_convblk = cfg.net.n_channels_init
            level2in_channels_convblk[i] = int(in_channels_convblk)
            level2out_channels_convblk[i] = int(out_channels_convblk)

        for i in reversed(range(cfg.net.n_levels)):
            info = {'parent': 'decoder', 'level': i}
            if cfg.net.recurrent:
                self.conv_blks[str(i)] = RRCNNBlk(cfg, level2in_channels_convblk[i], level2out_channels_convblk[i])
            else:
                self.conv_blks[str(i)] = ConvBlk(cfg, level2in_channels_convblk[i], level2out_channels_convblk[i],
                                                 n_layers=cfg.net.decoder.conv_blk.n_layers, info=info)
            if i != 0:
                if cfg.net.decoder.merge_skip == 'sum':
                    out_channels_deconvblk = level2in_channels_convblk[i - 1]
                elif cfg.net.decoder.merge_skip == 'concat':
                    out_channels_deconvblk = level2in_channels_convblk[i - 1] // 2
                else:
                    raise NotImplementedError
                self.deconv_blks[str(i)] = DeconvBlk(cfg, in_channels=level2out_channels_convblk[i],
                                                     out_channels=out_channels_deconvblk)

            if i != cfg.net.n_levels - 1:
                if cfg.net.attention:
                    if cfg.net.decoder.merge_skip == 'sum':
                        in_channels_gl = level2in_channels_convblk[i]
                    elif cfg.net.decoder.merge_skip == 'concat':
                        in_channels_gl = level2in_channels_convblk[i] // 2
                    else:
                        raise NotImplementedError
                    self.attention_blks[str(i)] = AttentionBlk(cfg, F_g=in_channels_gl, F_l=in_channels_gl,
                                                               F_int=in_channels_gl // 2)

        kernel_size = cfg.net.decoder.out_conv.kernel_size
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0
        else:
            raise ValueError
        self.out_conv_seg = Conv(in_channels=level2out_channels_convblk[0], out_channels=1, kernel_size=kernel_size,
                                 stride=1, padding=padding)

    def forward(self):
        for i in reversed(range(self.cfg.net.n_levels)):
            skip = self.cfg.var.obj_model.net.encoder.conv_blks[i].output
            if i == self.cfg.net.n_levels - 1:
                x = skip
            else:
                if self.cfg.net.attention:
                    skip = self.attention_blks[str(i)](g=x, x=skip)
                if self.cfg.net.decoder.merge_skip == 'sum':
                    x = x + skip
                elif self.cfg.net.decoder.merge_skip == 'concat':
                    x = torch.concat([x, skip], dim=1) # [B, 2xC, ...]
                else:
                    raise NotImplementedError
            x = self.conv_blks[str(i)](x)
            if i != 0:
                x = self.deconv_blks[str(i)](x)
        x_seg = self.out_conv_seg(x) # [B, 1, ...]
        self.outputs['seg'] = x_seg


class UNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, x):
        self.encoder(x)
        self.decoder()
        return None


class R2U_Net(nn.Module):

    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()
        self.decoder = nn.Sequential()
        self.decoder.outputs = {}

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNNBlk(ch_in=img_ch, ch_out=16, t=t)

        self.RRCNN2 = RRCNNBlk(ch_in=16, ch_out=32, t=t)

        self.RRCNN3 = RRCNNBlk(ch_in=32, ch_out=64, t=t)

        self.RRCNN4 = RRCNNBlk(ch_in=64, ch_out=128, t=t)

        self.RRCNN5 = RRCNNBlk(ch_in=128, ch_out=256, t=t)

        self.Up5 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN5 = RRCNNBlk(ch_in=256, ch_out=128, t=t)

        self.Up4 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN4 = RRCNNBlk(ch_in=128, ch_out=64, t=t)

        self.Up3 = up_conv(ch_in=64, ch_out=32)
        self.Up_RRCNN3 = RRCNNBlk(ch_in=64, ch_out=32, t=t)

        self.Up2 = up_conv(ch_in=32, ch_out=16)
        self.Up_RRCNN2 = RRCNNBlk(ch_in=32, ch_out=16, t=t)

        self.Conv_1x1 = nn.Conv2d(16, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        self.decoder.outputs['seg'] = d1
        return None


from torchvision.models import DenseNet
from torchvision.models.densenet import _Transition, _load_state_dict
import torch.nn.functional as F
from collections import OrderedDict


class _DenseUNetEncoder(DenseNet):

    def __init__(self, skip_connections, growth_rate, block_config, num_init_features, bn_size, drop_rate, downsample):
        super(_DenseUNetEncoder, self).__init__(growth_rate, block_config, num_init_features, bn_size, drop_rate)

        self.skip_connections = skip_connections

        # remove last norm, classifier
        features = OrderedDict(list(self.features.named_children())[:-1])
        delattr(self, 'classifier')
        if not downsample:
            features['conv0'].stride = 1
            del features['pool0']
        self.features = nn.Sequential(features)

        for module in self.features.modules():
            if isinstance(module, nn.AvgPool2d):
                module.register_forward_hook(lambda _, input, output: self.skip_connections.append(input[0]))

    def forward(self, x):
        return self.features(x)


class _DenseUNetDecoder(DenseNet):

    def __init__(self, skip_connections, growth_rate, block_config, num_init_features, bn_size, drop_rate, upsample):
        super(_DenseUNetDecoder, self).__init__(growth_rate, block_config, num_init_features, bn_size, drop_rate)

        self.skip_connections = skip_connections
        self.upsample = upsample

        # remove conv0, norm0, relu0, pool0, last denseblock, last norm, classifier
        features = list(self.features.named_children())[4:-2]
        delattr(self, 'classifier')

        num_features = num_init_features
        num_features_list = []
        for i, num_layers in enumerate(block_config):
            num_input_features = num_features + num_layers * growth_rate
            num_output_features = num_features // 2
            num_features_list.append((num_input_features, num_output_features))
            num_features = num_input_features // 2

        for i in range(len(features)):
            name, module = features[i]
            if isinstance(module, _Transition):
                num_input_features, num_output_features = num_features_list.pop(1)
                features[i] = (name, _TransitionUp(num_input_features, num_output_features, skip_connections))

        features.reverse()

        self.features = nn.Sequential(OrderedDict(features))

        num_input_features, _ = num_features_list.pop(0)

        if upsample:
            self.features.add_module('upsample0', nn.Upsample(scale_factor=4, mode='bilinear'))
        self.features.add_module('norm0', nn.BatchNorm2d(num_input_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('conv0',
                                 nn.Conv2d(num_input_features, num_init_features, kernel_size=1, stride=1, bias=False))
        self.features.add_module('norm1', nn.BatchNorm2d(num_init_features))

    def forward(self, x):
        return self.features(x)


class _Concatenate(nn.Module):

    def __init__(self, skip_connections):
        super(_Concatenate, self).__init__()
        self.skip_connections = skip_connections

    def forward(self, x):
        return torch.cat([x, self.skip_connections.pop()], 1)


class _TransitionUp(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, skip_connections):
        super(_TransitionUp, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1',
                        nn.Conv2d(num_input_features, num_output_features * 2, kernel_size=1, stride=1, bias=False))

        self.add_module('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
        self.add_module('cat', _Concatenate(skip_connections))
        self.add_module('norm2', nn.BatchNorm2d(num_output_features * 4))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2',
                        nn.Conv2d(num_output_features * 4, num_output_features, kernel_size=1, stride=1, bias=False))


class DenseUNet(nn.Module):

    def __init__(self, n_classes=1, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=16, bn_size=4,
                 drop_rate=0, downsample=True, pretrained_encoder_uri=None, progress=None):
        super(DenseUNet, self).__init__()
        self.skip_connections = []
        self.encoder = _DenseUNetEncoder(self.skip_connections, growth_rate, block_config, num_init_features, bn_size,
                                         drop_rate, downsample)
        self.decoder = _DenseUNetDecoder(self.skip_connections, growth_rate, block_config, num_init_features, bn_size,
                                         drop_rate, downsample)
        self.classifier = nn.Conv2d(num_init_features, n_classes, kernel_size=1, stride=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.encoder._load_state_dict = self.encoder.load_state_dict
        self.encoder.load_state_dict = lambda state_dict: self.encoder._load_state_dict(state_dict, strict=False)
        if pretrained_encoder_uri:
            _load_state_dict(self.encoder, str(pretrained_encoder_uri), progress)
        self.encoder.load_state_dict = lambda state_dict: self.encoder._load_state_dict(state_dict, strict=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        y = self.classifier(x)
        return self.softmax(y)


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


#Nested Unet


class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(NestedUNet, self).__init__()
        self.decoder = nn.Sequential()
        self.decoder.outputs = {}

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        self.decoder.outputs['seg'] = output
        return output
