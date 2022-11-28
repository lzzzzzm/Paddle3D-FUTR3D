import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager

__all__ = ['RadarPointEncoder']


class RFELayer(nn.Layer):
    """Radar Feature Encoder layer.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 ):
        super(RFELayer, self).__init__()
        self.fp16_enabled = False
        self.norm = nn.BatchNorm1D(num_features=out_channels, epsilon=1e-3, momentum=0.01)
        self.linear = nn.Linear(in_channels, out_channels, bias_attr=False)

    def forward(self, inputs):
        """Forward function.
        Args:
            inputs (torch.Tensor): Points features of shape (B, M, C).
                M is the number of points in
                C is the number of channels of point features.
        Returns:
            the same shape
        """

        x = self.linear(inputs)  # [B, M, C]
        # BMC -> BCM -> BMC
        x = paddle.transpose(x, (0, 2, 1))
        x = self.norm(x)
        x = paddle.transpose(x, (0, 2, 1))
        out = F.relu(x)

        return out


@manager.RADAR_ENCODER.add_component
class RadarPointEncoder(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels
                 ):
        super(RadarPointEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        in_chn = in_channels

        layers = []
        for out_chn in out_channels:
            layer = RFELayer(in_chn, out_chn)
            layers.append(layer)
            in_chn = out_chn

        self.feat_layers = nn.Sequential(*layers)

    def forward(self, points):
        '''
        points: [B, N, C]. N: as max
        masks: [B, N, 1]
        ret:
            out: [B, N, C+1], last channel as 0-1 mask
        '''
        masks = points[:, :, -1]
        masks = paddle.unsqueeze(masks, -1)
        x = points[:, :, :-1]
        xy = points[:, :, :2]

        for feat_layer in self.feat_layers:
            x = feat_layer(x)

        out = x * masks

        out = paddle.concat((x, masks), axis=-1)

        out = paddle.concat((xy, out), axis=-1)
        return out
