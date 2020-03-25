import math
from pathlib import Path

import torch
import torch.nn as nn

# Source: https://github.com/tonylins/pytorch-mobilenet-v2


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, omit_stride=False,
                 no_res_connect=False, dropout=0., bn_momentum=0.1,
                 batchnorm=None):
        super().__init__()
        self.out_channels = oup
        self.stride = stride
        self.omit_stride = omit_stride
        self.use_res_connect = not no_res_connect and\
            self.stride == 1 and inp == oup
        self.dropout = dropout
        actual_stride = self.stride if not self.omit_stride else 1
        if batchnorm is None:
            def batchnorm(num_features):
                return nn.BatchNorm2d(num_features, momentum=bn_momentum)

        assert actual_stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        if expand_ratio == 1:
            modules = [
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, actual_stride, 1,
                          groups=hidden_dim, bias=False),
                batchnorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                batchnorm(oup),
            ]
            if self.dropout > 0:
                modules.append(nn.Dropout2d(self.dropout))
            self.conv = nn.Sequential(*modules)
        else:
            modules = [
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                batchnorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, actual_stride, 1,
                          groups=hidden_dim, bias=False),
                batchnorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                batchnorm(oup),
            ]
            if self.dropout > 0:
                modules.insert(3, nn.Dropout2d(self.dropout))
            self.conv = nn.Sequential(*modules)
            self._initialize_weights()

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNetV2(nn.Module):
    def __init__(self, widen_factor=1., pretrained=True,
                 last_channel=None, input_channel=32):
        super().__init__()
        self.widen_factor = widen_factor
        self.pretrained = pretrained
        self.last_channel = last_channel
        self.input_channel = input_channel

        block = InvertedResidual
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(self.input_channel * widen_factor)
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * widen_factor)
            for i in range(n):
                if i == 0:
                    self.features.append(block(
                        input_channel, output_channel, s, expand_ratio=t,
                        omit_stride=True))
                else:
                    self.features.append(block(
                        input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        if self.last_channel is not None:
            output_channel = int(self.last_channel * widen_factor)\
                if widen_factor > 1.0 else self.last_channel
            self.features.append(conv_1x1_bn(input_channel, output_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.out_channels = output_channel
        self.feat_1x_channels = int(
            interverted_residual_setting[-1][1] * widen_factor)
        self.feat_2x_channels = int(
            interverted_residual_setting[-2][1] * widen_factor)
        self.feat_4x_channels = int(
            interverted_residual_setting[-4][1] * widen_factor)
        self.feat_8x_channels = int(
            interverted_residual_setting[-5][1] * widen_factor)

        if self.pretrained:
            state_dict = torch.load(
                Path(__file__).resolve().parent / 'weights/mobilenet_v2.pth.tar')
            self.load_state_dict(state_dict, strict=False)
        else:
            self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        feat_2x, feat_4x, feat_8x = None, None, None
        for idx, module in enumerate(self.features._modules.values()):
            x = module(x)
            if idx == 7:
                feat_4x = x.clone()
            elif idx == 14:
                feat_2x = x.clone()
            if idx > 0 and hasattr(module, 'stride') and module.stride != 1:
                x = x[..., ::2, ::2]

        return x, feat_2x, feat_4x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
