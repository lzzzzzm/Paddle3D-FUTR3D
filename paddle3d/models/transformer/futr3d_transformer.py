import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import math
import copy
import numpy as np

from paddle3d.apis import manager
from paddle3d.models.transformer.attention import MultiHeadAttention, FUTR3DCrossAtten, inverse_sigmoid
from paddle3d.models.layers.param_init import xavier_uniform_, constant_

__all__ = ["FUTR3DTransformer", 'FUTR3DTransformerDecoder', 'DetrTransformerDecoderLayer']


@manager.TRANSFORMER_LAYERS.add_component
class DetrTransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 attn_dropout=0.1,
                 use_LiDAR=False,
                 use_Cam=True,
                 use_Radar=True,
                 num_levels=4,
                 num_cams=6,
                 radar_dims=64,
                 radar_topk=30,
                 im2col_step=64,
                 pc_range=None,
                 use_dconv=True,
                 weight_dropout=0,
                 use_level_cam_embed=True,
                 num_points=1,
                 feedforward_channels=512,
                 feedforward_dropout=0.1,
                 normalize_before=False,
                 activation="relu"):
        super(DetrTransformerDecoderLayer, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.use_LiDAR = use_LiDAR
        self.use_Cam = use_Cam
        self.pc_range = pc_range
        self.use_dconv = use_dconv
        self.use_level_cam_embed = use_level_cam_embed
        self.num_points = num_points
        self.feedforward_channels = feedforward_channels
        self.feedforward_dropout = feedforward_dropout
        self.normalize_before = normalize_before

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if feedforward_dropout is None else feedforward_dropout

        self.self_attn = MultiHeadAttention(
            embed_dim=self.embed_dims,
            num_heads=self.num_heads,
            attn_drop=attn_dropout
        )

        self.cross_attn = FUTR3DCrossAtten(use_LiDAR=use_LiDAR,
                                           use_Cam=use_Cam,
                                           use_Radar=use_Radar,
                                           embed_dims=embed_dims,
                                           num_heads=num_heads,
                                           num_levels=num_levels,
                                           num_points=num_points,
                                           num_cams=num_cams,
                                           radar_dims=radar_dims,
                                           radar_topk=radar_topk,
                                           im2col_step=im2col_step,
                                           pc_range=pc_range,
                                           dropout=dropout,
                                           weight_dropout=weight_dropout,
                                           use_dconv=use_dconv,
                                           use_level_cam_embed=use_level_cam_embed)
        # ffn
        self.ffns = nn.LayerList(
            [nn.Linear(in_features=embed_dims, out_features=feedforward_channels),
             nn.Linear(in_features=feedforward_channels, out_features=embed_dims)]
        )
        # ffn dropout
        self.dropout1 = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
        # norms
        self.norms = nn.LayerList(
            [nn.LayerNorm(embed_dims) for i in range(3)]
        )
        self.activation = getattr(F, activation)
        # self._reset_parameters()

    # def _reset_parameters(self):
    #     linear_init_(self.ffn1)
    #     linear_init_(self.ffn2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                self_attn_masks=None,
                corss_attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                img_feats=None,
                pts_feats=None,
                rad_feats=None,
                img_metas=None):
        # order : self_attn->norm->corss_attn->norm->ffn->norm
        if self.normalize_before:
            query = self.norm1(query)
        temp_key = temp_value = query
        # self_attn
        # attn module has shortcut
        # self.attn
        query = self.self_attn(query=query,
                               key=temp_key,
                               value=temp_value,
                               identity=None,
                               query_pos=query_pos,
                               key_pos=query_pos,
                               attn_mask=self_attn_masks,
                               key_padding_mask=query_key_padding_mask,
                               )
        # norm
        query = self.norms[0](query)
        # corss_attn
        query = self.cross_attn(query=query,
                                key=key,
                                value=value,
                                identity=None,
                                query_pos=query_pos,
                                key_pos=key_pos,
                                attn_masks=corss_attn_masks,
                                key_padding_mask=key_padding_mask,
                                reference_points=reference_points,
                                spatial_shapes=spatial_shapes,
                                level_start_index=level_start_index,
                                img_feats=img_feats,
                                pts_feats=pts_feats,
                                rad_feats=rad_feats,
                                img_metas=img_metas
                                )
        # norm
        query = self.norms[1](query)
        # ffn
        residual = query
        query = self.ffns[1](self.dropout1(self.activation(self.ffns[0](query))))
        query = residual + self.dropout2(query)
        # norm
        query = self.norms[2](query)
        return query


@manager.TRANSFORMER_DECODER.add_component
class FUTR3DTransformerDecoder(nn.Layer):
    def __init__(self,
                 use_LiDAR=False,
                 use_Radar=True,
                 use_Cam=True,
                 num_levels=4,
                 num_layers=6,
                 return_intermediate=True,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 attn_dropout=0.1,
                 num_cams=6,
                 radar_dims=64,
                 radar_topk=30,
                 im2col_step=64,
                 use_dconv=True,
                 feedforward_channels=512,
                 feedforward_dropout=0.1,
                 normalize_before=False,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 ):
        super(FUTR3DTransformerDecoder, self).__init__()
        self.return_intermediate = return_intermediate
        self.num_layers = num_layers
        self.layers = nn.LayerList()
        for i in range(num_layers):
            self.layers.append(
                DetrTransformerDecoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    dropout=dropout,
                    attn_dropout= attn_dropout,
                    use_LiDAR= use_LiDAR,
                    use_Cam=use_Cam,
                    use_Radar= use_Radar,
                    num_levels=num_levels,
                    num_cams= num_cams,
                    radar_dims= radar_dims,
                    radar_topk= radar_topk,
                    im2col_step= im2col_step,
                    use_dconv= use_dconv,
                    feedforward_channels= feedforward_channels,
                    feedforward_dropout= feedforward_dropout,
                    normalize_before=normalize_before,
                    pc_range=pc_range
                )
            )
        self.embed_dims = self.layers[0].embed_dims

    def forward(self,
                query,
                key=None,
                pts_feats=None,
                img_feats=None,
                rad_feats=None,
                query_pos=None,
                reference_points=None,
                reg_branches=None,
                img_metas=None):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points_input,
                img_feats=img_feats,
                rad_feats=rad_feats,
                img_metas=img_metas
            )
            output = paddle.transpose(output, (1, 0, 2))
            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3
                new_reference_points = np.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                                                 ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                new_reference_points = paddle.to_tensor(reference_points, dtype='float32')
                reference_points = F.sigmoid(new_reference_points)
            output = paddle.transpose(output, (1, 0, 2))
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return paddle.stack(intermediate), paddle.stack(
                intermediate_reference_points)

        return output, reference_points


@manager.TRANSFORMER.add_component
class FUTR3DTransformer(nn.Layer):
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 transformer_decoder=None,
                 reference_points_aug=False):
        super(FUTR3DTransformer, self).__init__()
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        self.decoder = transformer_decoder
        self.reference_points_aug = reference_points_aug
        self.embed_dims = self.decoder.embed_dims
        # layer
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

        xavier_uniform_(self.reference_points.weight)
        constant_(self.reference_points.bias, 0)


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
