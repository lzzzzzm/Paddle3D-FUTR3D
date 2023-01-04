import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute

from paddle3d.apis import manager
from paddle3d.models.layers.param_init import (constant_init,
                                               xavier_uniform_init)

from .transformer_layers import BaseTransformerLayer, TransformerLayerSequence


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = paddle.clip(x, min=0, max=1)
    x1 = paddle.clip(x, min=eps)
    x2 = paddle.clip((1 - x), min=eps)
    return paddle.log(x1 / x2)


@manager.MODELS.add_component
class FUTR3DTransformer(nn.Layer):
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 decoder=None,
                 reference_points_aug=False):
        super(FUTR3DTransformer, self).__init__()
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        self.decoder = decoder
        self.reference_points_aug = reference_points_aug
        self.embed_dims = self.decoder.embed_dims
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_init(p)

        xavier_uniform_init(self.reference_points.weight)
        constant_init(self.reference_points.bias, value=0)

    def forward(self,
                pts_feats,
                img_feats,
                rad_feats,
                query_embed,
                reg_branches=None,
                img_metas=None):
        assert query_embed is not None
        if pts_feats:
            bs = pts_feats[0].shape[0]
        else:
            bs = img_feats[0].shape[0]
        query_pos, query = paddle.split(query_embed, query_embed.shape[1] // self.embed_dims, axis=1)
        query_pos = query_pos.unsqueeze(0).tile([bs, 1, 1])
        query = query.unsqueeze(0).tile([bs, 1, 1])

        reference_points = self.reference_points(query_pos)
        if self.training and self.reference_points_aug:
            reference_points = reference_points + paddle.normal(reference_points)
        reference_points = F.sigmoid(reference_points)
        init_reference_out = reference_points

        # decoder
        query = paddle.transpose(query, (1, 0, 2))
        query_pos = paddle.transpose(query_pos, (1, 0, 2))

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            pts_feats=pts_feats,
            img_feats=img_feats,
            rad_feats=rad_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            img_metas=img_metas)
        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


@manager.MODELS.add_component
class DetrTransformerDecoderLayer(BaseTransformerLayer):
    def __init__(self,
                 attns,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LayerNorm'),
                 ffn_num_fcs=2,
                 use_recompute=False,
                 **kwargs):
        super(DetrTransformerDecoderLayer, self).__init__(
            attns=attns,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs
        )
        self.use_recompute = use_recompute
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def _forward(
            self,
            query,
            key=None,
            value=None,
            query_pos=None,
            key_pos=None,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            **kwargs
    ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(DetrTransformerDecoderLayer, self).forward(
            query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask,
            **kwargs
        )

        return x

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_recompute and self.training:
            x = recompute(
                self._forward,
                query,
                key,
                value,
                query_pos,
                key_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                **kwargs
            )
        else:
            x = self._forward(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        return x


@manager.MODELS.add_component
class FUTR3DTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=None,
                 return_intermediate=False,
                 **kwargs):

        super(FUTR3DTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            # TODO hard code
            self.post_norm = nn.LayerNorm(self.embed_dims)
        else:
            self.post_norm = None

    def forward(self,
                query,
                reference_points=None,
                reg_branches=None,
                *args,
                **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer(query,
                           reference_points=reference_points_input,
                           *args,
                           **kwargs)
            output = paddle.transpose(output, (1, 0, 2))
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 3
                new_reference_points = paddle.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                reference_points = F.sigmoid(new_reference_points)
            output = paddle.transpose(output, (1, 0, 2))
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return paddle.stack(intermediate), paddle.stack(
                intermediate_reference_points)

        return output, reference_points
