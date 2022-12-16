import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.models.transformer.attention.multiheadattention import  MultiHeadSelfAttention
from paddle3d.models.detection.futr3d.futr3d_utils import nan_to_num

import numpy as np
from scipy.spatial.distance import cdist
import pickle


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename

def load_variavle(filename):
   f=open(filename,'rb')
   r=pickle.load(f)
   f.close()
   return r


def convert_attention_mask(attn_mask, dtype):
    """
    Convert the attention mask to the target dtype we expect.
    Parameters:
        attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
        dtype (VarType): The target type of `attn_mask` we expect.
    Returns:
        Tensor: A Tensor with shape same as input `attn_mask`, with data type `dtype`.
    """
    return nn.layer.transformer._convert_attention_mask(attn_mask, dtype)


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



def gather(feature: paddle.Tensor, ind: paddle.Tensor):
    """Simplified version of torch.gather. Always gather based on axis 1.
    Args:
        feature: all results in 3 dimensions, such as [n, h * w, c]
        ind: positive index in 3 dimensions, such as [n, k, 1]
    Returns:
        gather feature
    """
    bs_ind = paddle.arange(ind.shape[0], dtype=ind.dtype)
    bs_ind = bs_ind.unsqueeze(1).unsqueeze(2)
    print(bs_ind.shape)
    bs_ind = bs_ind.expand([ind.shape[0], ind.shape[1], 1])
    ind = paddle.concat([bs_ind, ind], axis=-1)

    return feature.gather_nd(ind)

def pairwise_dist (A, B):
    A2 = (A ** 2).sum(axis=1).reshape((-1, 1))
    B2 = (B ** 2).sum(axis=1)
    D = A2 + B2 - 2 * A.mm(B.transpose((1, 0)))
    D = D.sqrt()
    return D


def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    reference_points = reference_points.clone()
    lidar2img = paddle.to_tensor(lidar2img, dtype='float32')
    # lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    reference_points_3d = reference_points.clone()
    reference_points = reference_points.numpy()
    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    reference_points = paddle.to_tensor(reference_points)
    B, num_query = reference_points.shape[:2]

    # reference_points (B, num_queries, 4)
    reference_points = paddle.concat((reference_points, paddle.ones_like(reference_points[..., :1])), -1)
    num_cam = lidar2img.shape[1]
    # ref_point change to (B, num_cam, num_query, 4, 1)
    # reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    reference_points = paddle.reshape(reference_points, shape=(B, 1, num_query, 4)).tile([1, num_cam, 1, 1]).unsqueeze(
        -1)
    lidar2img = paddle.reshape(lidar2img, shape=(B, num_cam, 1, 4, 4)).tile([1, 1, num_query, 1, 1])
    # lidar2img = paddle.repeat_interleave(lidar2img, (1, 1, num_query, 1, 1))
    # ref_point_cam change to (B, num_cam, num_query, 4)
    reference_points_cam = paddle.matmul(lidar2img, reference_points).squeeze(-1)
    # save_variable(lidar2img.numpy(), 'lidar2img.txt')
    # save_variable(reference_points_cam.numpy(), 'reference_points_cam.txt')

    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    # ref_point_cam change to img coordinates
    reference_points_cam = reference_points_cam[..., 0:2] / paddle.maximum(
        reference_points_cam[..., 2:3], paddle.ones_like(reference_points_cam[..., 2:3]) * eps)
    # img_metas['img_shape']=[900, 1600]
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 1:2] > -1.0)
            & (reference_points_cam[..., 1:2] < 1.0))
    # mask shape (B, 1, num_query, num_cam, 1, 1)
    mask = paddle.reshape(mask, shape=(B, num_cam, 1, num_query, 1, 1))
    mask = paddle.transpose(mask, (0, 2, 3, 1, 4, 5))

    # mask = nan_to_num(mask)
    sampled_feats = []
    num_points = 1
    # save_variable(reference_points_cam.numpy(), 'reference_points_cam.txt')
    reference_points_cam = load_variavle('reference_points_cam.txt')
    reference_points_cam = paddle.to_tensor(reference_points_cam)
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.shape
        # feat_flip = paddle.flip(feat, [-1])
        feat = paddle.reshape(feat, shape=(B * N, C, H, W))
        # ref_point_cam shape change from (B, num_cam, num_query, 2) to (B*num_cam, num_query/10, 10, 2)
        reference_points_cam_lvl = paddle.reshape(reference_points_cam, shape=(B * N, int(num_query / 10), 10, 2))
        # sample_feat shape (B*N, C, num_query/10, 10)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        # sampled_feat = sampled_feat.clone()
        # sampled_feat shape (B, C, num_query, N, num_points)
        sampled_feat = paddle.reshape(sampled_feat, shape=(B, N, C, num_query, num_points))
        sampled_feat = paddle.transpose(sampled_feat, (0, 2, 3, 1, 4))
        save_variable(sampled_feat.numpy(), 'sampled_feat.txt')
        sampled_feats.append(sampled_feat)

    sampled_feats = paddle.stack(sampled_feats, -1)
    # sampled_feats (B, C, num_query, num_cam, num_points, len(lvl_feats))
    sampled_feats = paddle.reshape(sampled_feats, shape=(B, C, num_query, num_cam, num_points, len(mlvl_feats)))
    save_variable(sampled_feats.numpy(), 'sampled_feats.txt')
    # ref_point_3d (B, N, num_query, 3)  maks (B, N, num_query, 1)
    return reference_points_3d, sampled_feats, mask


class MultiHeadAttention(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.):
        super(MultiHeadAttention, self).__init__()
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        out = self.attn(query=query, key=key, value=value)
        return identity + self.proj_drop(out)


class FUTR3DCrossAtten(nn.Layer):
    def __init__(self,
                 use_LiDAR=True,
                 use_Cam=False,
                 use_Radar=True,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=1,
                 num_cams=6,
                 radar_dims=64,
                 radar_topk=30,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 weight_dropout=0.0,
                 use_dconv=False,
                 use_level_cam_embed=False):
        super(FUTR3DCrossAtten, self).__init__()
        self.use_LiDAR = use_LiDAR
        self.use_Cam = use_Cam
        self.use_Radar = use_Radar
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_cams = num_cams
        self.radar_dims = radar_dims
        self.radar_topk = radar_topk
        self.im2col_step = im2col_step
        self.pc_range = pc_range
        self.dropout = dropout
        self.weight_dropout = weight_dropout
        self.use_dconv = use_dconv
        self.use_level_cam_embed = use_level_cam_embed
        if self.embed_dims % self.num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads

        self.weight_dropout = nn.Dropout(weight_dropout)
        self.dropout = nn.Dropout(dropout)

        self.fused_embed = 0
        if self.use_Cam:
            self.attention_weights = nn.Linear(embed_dims,
                                               num_cams * num_levels * num_points)
            self.img_output_proj = nn.Linear(embed_dims, embed_dims)
            self.fused_embed += embed_dims

        if self.use_LiDAR:
            self.pts_attention_weights = nn.Linear(embed_dims,
                                                   num_levels * num_points)
            self.pts_output_proj = nn.Linear(embed_dims, embed_dims)
            self.fused_embed += embed_dims

        if self.use_Radar:
            self.radar_dims = radar_dims
            self.radar_topk = radar_topk
            self.radar_attention_weights = nn.Linear(embed_dims, radar_topk)
            self.radar_output_proj = nn.Linear(self.radar_dims, self.radar_dims)
            self.fused_embed += radar_dims

        if self.fused_embed > embed_dims:
            self.modality_fusion_layer = nn.Sequential(
                nn.Linear(self.fused_embed, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
            )

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
        )

    def forward(self,
                query,
                key,
                value,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                img_feats=None,
                pts_feats=None,
                rad_feats=None,
                img_metas=None):
        # save_variable(query.numpy(), 'self_attn.txt')

        if key is None:
            key = query
        if value is None:
            value = key

        if identity is None:
            inp_identity = query
        if query_pos is not None:
            query = query + query_pos

        query = paddle.transpose(query, (1, 0, 2))
        bs, num_query, _ = query.shape
        if self.use_Cam:
            # (B, 1, num_query, num_cams, num_points, num_levels)
            img_attention_weights = self.attention_weights(query)
            img_attention_weights = paddle.reshape(img_attention_weights,
                                                   shape=(
                                                       bs, 1, num_query, self.num_cams, self.num_points,
                                                       self.num_levels))

            reference_points_3d, img_output, mask = feature_sampling(
                img_feats, reference_points, self.pc_range, img_metas)

            img_output = nan_to_num(img_output)
            img_attention_weights = F.sigmoid(img_attention_weights)
            img_attention_weights = self.weight_dropout(img_attention_weights)

            img_output = img_output * img_attention_weights
            img_output = paddle.sum(img_output, -1)
            img_output = paddle.sum(img_output, -1)
            img_output = paddle.sum(img_output, -1)
            img_output = paddle.transpose(img_output, (2, 0, 1))
            img_output = self.img_output_proj(img_output)

        if self.use_Radar:
            radar_feats, radar_mask = rad_feats[:, :, :-1], rad_feats[:, :, -1]
            radar_xy = radar_feats[:, :, :2]
            ref_xy = reference_points[:, :, :2]
            radar_feats = radar_feats[:, :, 2:]
            pad_xy = paddle.ones_like(radar_xy) * 1000.0
            temp_radar_mask = (1.0 - paddle.unsqueeze(radar_mask, -1)) * pad_xy
            radar_xy = radar_xy + temp_radar_mask
            # [B, num_query, M]
            ref_radar_dist = []
            for index in range(ref_xy.shape[0]):
                ref_radar_dist.append(-1.0 * cdist(ref_xy[index], radar_xy[index]))
            ref_radar_dist = paddle.to_tensor(ref_radar_dist, dtype='float32')
            test1 = ref_radar_dist
            ref_radar_dist = load_variavle('ref_radar_dist.txt')
            ref_radar_dist = paddle.to_tensor(ref_radar_dist, dtype='float32')
            test2 = ref_radar_dist
            compare = (test1 - test2).max()
            # [B, num_query, topk]
            _value, indices = paddle.topk(ref_radar_dist, self.radar_topk)
            # [B, num_query, M]
            radar_mask = radar_mask.unsqueeze(1).tile([1, num_query, 1])
            # [B, num_query, topk]
            top_mask = paddle.take_along_axis(radar_mask, indices=indices, axis=2)
            # [B, num_query, M, radar_dim]
            radar_feats = radar_feats.unsqueeze(1).tile([1, num_query, 1, 1])
            # radar_feats = paddle.tile(radar_feats, (1, num_query, 1, 1))
            radar_dim = radar_feats.shape[-1]
            # [B, num_query, topk, radar_dim]
            indices_pad = indices.unsqueeze(-1).tile([1, 1, 1, radar_dim])
            # [B, num_query, topk, radar_dim]
            radar_feats_topk = paddle.take_along_axis(radar_feats, axis=2, indices=indices_pad)

            radar_attention_weights = self.radar_attention_weights(query).reshape(
                [bs, num_query, self.radar_topk])

            # [B, num_query, topk]
            radar_attention_weights = F.sigmoid(radar_attention_weights) * top_mask
            # [B, num_query, topk, radar_dim]
            radar_out = radar_feats_topk * radar_attention_weights.unsqueeze(-1)
            # [bs, num_query, radar_dim]
            radar_out = radar_out.sum(2)

            # change to (num_query, bs, embed_dims)
            radar_out = radar_out.transpose((1, 0, 2))

            radar_out = self.radar_output_proj(radar_out)

        if self.use_Cam and self.use_Radar:
            output = paddle.concat((img_output, radar_out), axis=2).transpose((1, 0, 2))
            output = self.modality_fusion_layer(output).transpose((1, 0, 2))
        elif self.use_Cam:
            output = img_output

        reference_points_3d = reference_points.clone()
        # (num_query, bs, embed_dims)
        pos_encoder_reference_points_3d = self.pos_encoder(inverse_sigmoid(reference_points_3d)).transpose((1, 0, 2))
        output = self.dropout(output) + inp_identity + pos_encoder_reference_points_3d
        return output
