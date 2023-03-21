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
"""
This code is based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_3d/spconv_backbone.py#L69
Ths copyright of OpenPCDet is as follows:
Apache-2.0 license [see LICENSE for details].
"""

import numpy as np
import paddle
from paddle import sparse
from paddle.sparse import nn

from paddle3d.apis import manager
from paddle3d.models.layers import param_init

from .sparse_resnet import SparseBasicBlock

__all__ = ['SparseEncoder']


def make_sparse_convmodule(in_channels,
                           out_channels,
                           kernel_size,
                           indice_key,
                           stride=1,
                           padding=0,
                           conv_type='SubMConv3d',
                           norm_cfg=None,
                           order=('conv', 'norm', 'act')):
    if conv_type == 'SubMConv3d':
        conv = nn.SubmConv3D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias_attr=False)
    elif conv_type == 'SparseConv3d':
        conv = nn.Conv3D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=False)
    elif conv_type == 'inverseconv':
        raise NotImplementedError
    else:
        raise NotImplementedError

    m = paddle.nn.Sequential(
        conv,
        nn.BatchNorm(out_channels, epsilon=1e-3, momentum=1 - 0.01, bias_attr=False),
        nn.ReLU(),
    )
    return m


@manager.MIDDLE_ENCODERS.add_component
class SparseEncoder(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                       64)),
                 encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                1)),
                 block_type='conv_module'):
        super(SparseEncoder, self).__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            self.base_channels,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')

        # self.init_weight()


    # def init_weight(self):
    #     for layer in self.sublayers():
    #         if isinstance(layer, (nn.Conv3D, nn.SubmConv3D)):
    #             param_init.reset_parameters(layer)
    #         if isinstance(layer, nn.BatchNorm):
    #             param_init.constant_init(layer.weight, value=1)
    #             param_init.constant_init(layer.bias, value=0)

    def make_encoder_layers(self,
                            in_channels,
                            block_type='conv_module',
                            conv_cfg=dict(type='SubMConv3d')):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = paddle.nn.Sequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_sparse_convmodule(
                            in_channels,
                            out_channels,
                            3,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_sparse_convmodule(
                                in_channels,
                                out_channels,
                                3,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                bias_attr=False))
                else:
                    blocks_list.append(
                        make_sparse_convmodule(
                            in_channels,
                            out_channels,
                            3,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = paddle.nn.Sequential(*blocks_list)
            self.encoder_layers.add_sublayer(stage_name, stage_layers)
        return out_channels


    def forward(self, voxel_features, coors, batch_size):

        shape = [batch_size] + list(self.sparse_shape) + [self.in_channels]
        input_sp_tensor = sparse.sparse_coo_tensor(
            coors.transpose((1, 0)),
            voxel_features,
            shape=shape,
            stop_gradient=False)

        x = self.conv_input(input_sp_tensor)
        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)
        out = self.conv_out(encode_features[-1])
        out = out.to_dense()
        out = paddle.transpose(out, perm=[0, 4, 1, 2, 3])
        N, C, D, H, W = out.shape
        out = paddle.reshape(out, shape=[N, C * D, H, W])
        return out
