# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from numbers import Integral

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.vision.ops import DeformConv2D
from paddle.nn.initializer import Constant, Uniform
from paddle.regularizer import L2Decay

from paddle3d.apis import manager
from paddle3d.models import layers
from paddle3d.models.layers import reset_parameters
from paddle3d.utils import checkpoint

__all__ = ['ResNet']


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 is_vd_mode=False,
                 act=None,
                 frozen_norm=False,
                 lr=1.0,
                 dcn_v2=False,
                 data_format='NCHW'):
        super(ConvBNLayer, self).__init__()
        if dilation != 1 and kernel_size != 3:
            raise RuntimeError("When the dilation isn't 1," \
                "the kernel_size should be 3.")
        self.act = act
        self.dcn_v2 = dcn_v2
        self.is_vd_mode = is_vd_mode
        if self.is_vd_mode:
            self._pool2d_avg = nn.AvgPool2D(
                kernel_size=2,
                stride=2,
                padding=0,
                ceil_mode=True,
                data_format=data_format)
        if self.dcn_v2:
            self.offset_channel = 2 * kernel_size ** 2
            self.mask_channel = kernel_size ** 2
            self.conv = DeformConv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                dilation=1,
                groups=groups,
                weight_attr=ParamAttr(learning_rate=lr),
                bias_attr=False)

            self.conv_offset = nn.Conv2D(
                in_channels=in_channels,
                out_channels=3 * kernel_size ** 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                weight_attr=ParamAttr(initializer=Constant(0.)),
                bias_attr=ParamAttr(initializer=Constant(0.)))

        else:
            self._conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2 \
                    if dilation == 1 else dilation,
                dilation=dilation,
                groups=groups,
                bias_attr=False,
                data_format=data_format)

        self._batch_norm = nn.BatchNorm2D(out_channels, data_format=data_format)
        # # freeze norm parameters
        # norm_params = self._batch_norm.parameters()
        # if frozen_norm:
        #     for param in norm_params:
        #         param.stop_gradient = True

        if act:
            self._act = nn.ReLU()

    def forward(self, inputs):
        if not self.dcn_v2:
            if self.is_vd_mode:
                inputs = self._pool2d_avg(inputs)
            y = self._conv(inputs)
            y = self._batch_norm(y)
            if self.act:
                y = self._act(y)
        else:
            offset_mask = self.conv_offset(inputs)
            offset, mask = paddle.split(
                offset_mask,
                num_or_sections=[self.offset_channel, self.mask_channel],
                axis=1)
            mask = F.sigmoid(mask)
            out = self.conv(inputs, offset, mask=mask)
            y = self._batch_norm(out)
            if self.act:
                y = self._act(y)

        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 first_conv=False,
                 dilation=1,
                 is_vd_mode=False,
                 frozen_norm=False,
                 style='pytorch',
                 dcn_v2=False,
                 data_format='NCHW'):
        super(BottleneckBlock, self).__init__()

        self.data_format = data_format
        if style == 'pytorch':
            conv0_stride = 1
            conv1_stride = stride
        else:
            conv0_stride = stride
            conv1_stride = 1

        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=conv0_stride,
            act='relu',
            frozen_norm=frozen_norm,
            data_format=data_format)

        if first_conv and dilation != 1:
            dilation //= 2

        self.dilation = dilation

        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=conv1_stride,
            act='relu',
            dilation=dilation,
            frozen_norm=frozen_norm,
            dcn_v2=dcn_v2,
            data_format=data_format)
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            frozen_norm=frozen_norm,
            data_format=data_format)

        if if_first or stride == 1:
            is_vd_mode = False

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=stride,
                is_vd_mode=is_vd_mode,
                frozen_norm=frozen_norm,
                data_format=data_format)

        self.shortcut = shortcut
        # NOTE: Use the wrap layer for quantization training
        self.relu = nn.ReLU()

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(short, conv2)
        y = self.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilation=1,
                 shortcut=True,
                 if_first=False,
                 is_vd_mode=False,
                 frozen_norm=False,
                 dcn_v2=False,
                 data_format='NCHW'):
        super(BasicBlock, self).__init__()
        self.dcn_v2 = dcn_v2
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            act='relu',
            freeze_norm=frozen_norm,
            data_format=data_format)
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=dilation,
            act=None,
            freeze_norm=frozen_norm,
            dcn_v2=self.dcn_v2,
            data_format=data_format)

        if if_first or stride == 1:
            is_vd_mode = False
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                is_vd_mode=is_vd_mode,
                freeze_norm=frozen_norm,
                data_format=data_format)

        self.shortcut = shortcut
        self.dilation = dilation
        self.data_format = data_format
        self.relu = nn.ReLU()

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(short, conv1)
        y = self.relu(y)

        return y


@manager.BACKBONES.add_component
class ResNet(nn.Layer):
    def __init__(self,
                 layers=50,
                 output_stride=8,
                 multi_grid=(1, 1, 1),
                 return_idx=[3],
                 pretrained=None,
                 variant='b',
                 style='pytorch',
                 frozen_stages=-1,
                 frozen_norm=False,
                 dcn_v2=False,
                 norm_eval=False,
                 stage_with_dcn=(False, False, False, False),
                 preprocess=True,
                 data_format='NCHW'):
        """
        Residual Network, see https://arxiv.org/abs/1512.03385

        Args:
            variant (str): ResNet variant, supports 'a', 'b', 'c', 'd' currently
            layers (int, optional): The layers of ResNet_vd. The supported layers are (18, 34, 50, 101, 152, 200). Default: 50.
            output_stride (int, optional): The stride of output features compared to input images. It is 8 or 16. Default: 8.
            multi_grid (tuple|list, optional): The grid of stage4. Defult: (1, 1, 1).
            pretrained (str, optional): The path of pretrained model.
            style (str, optional): If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
                                   it is "caffe", the stride-two layer is the first 1x1 conv layer.
            frozen_stages (int): Stages to be frozen (stop grad and set eval mode):
                            -1 means not freezing any parameters.

        """
        super(ResNet, self).__init__()
        self.stage_with_dcn = stage_with_dcn
        self.dcn_v2 = dcn_v2
        self.variant = variant
        self.style = style
        self.frozen_norm = frozen_norm
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.data_format = data_format
        self.conv1_logit = None  # for gscnn shape stream
        self.layers = layers
        self.pre_process = preprocess
        self.norm_mean = paddle.to_tensor([0.485, 0.456, 0.406])
        self.norm_std = paddle.to_tensor([0.229, 0.224, 0.225])
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024
                        ] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        # for channels of four returned stages
        self.feat_channels = [c * 4 for c in num_filters
                              ] if layers >= 50 else num_filters

        dilation_dict = None
        if output_stride == 8:
            dilation_dict = {2: 2, 3: 4}
        elif output_stride == 16:
            dilation_dict = {3: 2}
        else:
            dilation_dict = {2:1, 3:1}

        self.return_idx = return_idx

        if variant in ['c', 'd']:
            conv_defs = [
                [3, 32, 3, 2],
                [32, 32, 3, 1],
                [32, 64, 3, 1],
            ]
        else:
            conv_defs = [[3, 64, 7, 2]]
        self.conv1 = nn.Sequential()
        for (i, conv_def) in enumerate(conv_defs):
            c_in, c_out, k, s = conv_def
            self.conv1.add_sublayer(
                str(i),
                ConvBNLayer(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=k,
                    stride=s,
                    act='relu',
                    frozen_norm=self.frozen_norm,
                    is_vd_mode=False,
                    data_format=data_format))
        self.pool2d_max = nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, data_format=data_format)

        self.stage_list = []
        if layers >= 50:
            index = 0
            for block in range(len(depth)):
                # freeze norm
                if index < frozen_stages+1:
                    frozen_norm = False
                else:
                    frozen_norm = self.frozen_norm
                dcn_v2 = self.dcn_v2 if self.stage_with_dcn[index] else None
                index = index + 1
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)

                    ###############################################################################
                    # Add dilation rate for some segmentation tasks, if dilation_dict is not None.
                    dilation_rate = dilation_dict[
                        block] if dilation_dict and block in dilation_dict else 1

                    # Actually block here is 'stage', and i is 'block' in 'stage'
                    # At the stage 4, expand the the dilation_rate if given multi_grid
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]

                    ###############################################################################
                    bottleneck_block = self.add_sublayer(
                        'layer_%d_%d' % (block, i),
                        BottleneckBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0
                            and dilation_rate == 1 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            first_conv=i == 0,
                            is_vd_mode=variant in ['c', 'd'],
                            dilation=dilation_rate,
                            style=self.style,
                            dcn_v2=dcn_v2,
                            frozen_norm=frozen_norm,
                            data_format=data_format))

                    block_list.append(bottleneck_block)
                    shortcut = True
                self.stage_list.append(block_list)
        else:
            for block in range(len(depth)):
                # freeze norm
                if index > frozen_stages:
                    frozen_norm = False
                else:
                    frozen_norm = self.frozen_norm
                dcn_v2 = self.dcn_v2 if self.stage_with_dcn[index] else None
                index = index + 1
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    dilation_rate = dilation_dict[block] \
                        if dilation_dict and block in dilation_dict else 1
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]

                    basic_block = self.add_sublayer(
                        'layer_%d_%d' % (block, i),
                        BasicBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 \
                                and dilation_rate == 1 else 1,
                            dilation=dilation_rate,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            is_vd_mode=variant in ['c', 'd'],
                            dcn_v2=dcn_v2,
                            frozen_norm=frozen_norm,
                            data_format=data_format))
                    block_list.append(basic_block)
                    shortcut = True
                self.stage_list.append(block_list)

        self.pretrained = pretrained
        self.init_weight()


    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in self.conv1.sublayers():
                if isinstance(m, nn.BatchNorm2D):
                    m.eval()
                for param in m.parameters():
                    param.stop_gradient = True

            for i in range(min(self.frozen_stages+1, 4)):
                for j in range(len(self.stage_list[i])):
                    self.stage_list[i][j].eval()
                    for p in self.stage_list[i][j].parameters():
                        p.stop_gradient = True


    def forward(self, inputs):
        if self.pre_process:
            inputs = self.preprocess(inputs)
        y = self.conv1(inputs)
        y = self.pool2d_max(y)

        # A feature list saves the output feature map of each stage.
        feat_list = []
        for idx, stage in enumerate(self.stage_list):
            for block in stage:
                y = block(y)
            if idx in self.return_idx:
                feat_list.append(y)

        return feat_list

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images [paddle.Tensor(N, 3, H, W)]: Input images
        Return
            x [paddle.Tensor(N, 3, H, W)]: Preprocessed images
        """
        x = images
        # Create a mask for padded pixels
        mask = paddle.isnan(x)

        # Match ResNet pretrained preprocessing
        x = self.normalize(x, mean=self.norm_mean, std=self.norm_std)

        # Make padded pixels = 0
        a = paddle.zeros_like(x)
        x = paddle.where(mask, a, x)

        return x

    def normalize(self, image, mean, std):
        shape = paddle.shape(image)
        if mean.shape:
            mean = mean[..., :, None]
        if std.shape:
            std = std[..., :, None]
        out = (image.reshape([shape[0], shape[1], shape[2] * shape[3]]) -
               mean) / std
        return out.reshape(shape)

    def init_weight(self):
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                reset_parameters(sublayer)

    def train(self):
        super(ResNet, self).train()
        self._freeze_stages()
        if self.norm_eval:
            for m in self.sublayers():
                if isinstance(m, nn.BatchNorm2D):
                    m.eval()