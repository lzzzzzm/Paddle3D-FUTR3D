import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import numpy as np
from scipy.spatial.distance import cdist


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
    x2 = paddle.clip((1-x), min=eps)
    return paddle.log(x1 / x2)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None, name=None):
    """
    Replaces NaN, positive infinity, and negative infinity values in input tensor.
    Args:
        x (Tensor): An N-D Tensor, the data type is float32, float64.
        nan (float, optional): the value to replace NaNs with. Default is 0.
        posinf (float, optional): if a Number, the value to replace positive infinity values with. If None, positive infinity values are replaced with the greatest finite value representable by input’s dtype. Default is None.
        neginf (float, optional): if a Number, the value to replace negative infinity values with. If None, negative infinity values are replaced with the lowest finite value representable by input’s dtype. Default is None.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        Tensor: Results of nan_to_num operation input Tensor ``x``.
    Examples:
        .. code-block:: python
            import paddle
            x = paddle.to_tensor([float('nan'), 0.3, float('+inf'), float('-inf')], dtype='float32')
            out1 = paddle.nan_to_num(x)  # [0, 0.3, 3.4028235e+38, -3.4028235e+38]
            out2 = paddle.nan_to_num(x, nan=1)  # [1, 0.3, 3.4028235e+38, -3.4028235e+38]
            out3 = paddle.nan_to_num(x, posinf=5)  # [0, 0.3, 5, -3.4028235e+38]
            out4 = paddle.nan_to_num(x, nan=10, neginf=-99)  # [10, 0.3, 3.4028235e+38, -99]
    """
    # NOTE(tiancaishaonvjituizi): it seems that paddle handles the dtype of python float number
    # incorrectly, so we have to explicitly contruct tensors here
    posinf_value = paddle.full_like(x, float("+inf"))
    neginf_value = paddle.full_like(x, float("-inf"))
    nan = paddle.full_like(x, nan)
    assert x.dtype in [paddle.float32, paddle.float64]
    is_float32 = x.dtype == paddle.float32
    if posinf is None:
        posinf = (
            np.finfo(np.float32).max if is_float32 else np.finfo(np.float64).max
        )
    posinf = paddle.full_like(x, posinf)
    if neginf is None:
        neginf = (
            np.finfo(np.float32).min if is_float32 else np.finfo(np.float64).min
        )
    neginf = paddle.full_like(x, neginf)
    x = paddle.where(paddle.isnan(x), nan, x)
    x = paddle.where(x == posinf_value, posinf, x)
    x = paddle.where(x == neginf_value, neginf, x)
    return x

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


def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    reference_points = reference_points.clone()
    lidar2img = paddle.to_tensor(lidar2img, dtype='float32')
    # lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
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
    reference_points_cam = paddle.matmul(lidar2img, reference_points)
    reference_points_cam = paddle.squeeze(reference_points_cam, -1)
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
        sampled_feats.append(sampled_feat)
    sampled_feats = paddle.stack(sampled_feats, -1)
    # sampled_feats (B, C, num_query, num_cam, num_points, len(lvl_feats))
    sampled_feats = paddle.reshape(sampled_feats, shape=(B, C, num_query, num_cam, num_points, len(mlvl_feats)))

    # ref_point_3d (B, N, num_query, 3)  maks (B, N, num_query, 1)
    return reference_points_3d, sampled_feats, mask

class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.

    Examples:

        .. code-block:: python

            import paddle

            # encoder input: [batch_size, sequence_length, d_model]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.nn.MultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            self.in_proj_weight = self.create_parameter(
                shape=[embed_dim, 3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=False)
            self.in_proj_bias = self.create_parameter(
                shape=[3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=True)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(self.kdim, embed_dim)
            self.v_proj = nn.Linear(self.vdim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._type_list = ('q_proj', 'k_proj', 'v_proj')



    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             xavier_uniform_(p)
    #         else:
    #             constant_(p)

    def compute_qkv(self, tensor, index):
        if self._qkv_same_embed_dim:
            tensor = F.linear(
                x=tensor,
                weight=self.in_proj_weight[:, index * self.embed_dim:(index + 1)
                                           * self.embed_dim],
                bias=self.in_proj_bias[index * self.embed_dim:(index + 1) *
                                       self.embed_dim]
                if self.in_proj_bias is not None else None)
        else:
            tensor = getattr(self, self._type_list[index])(tensor)
        tensor = tensor.reshape(
            [0, 0, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        return tensor

    def forward(self, query, key=None, value=None, attn_mask=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
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

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output. Or a tuple if \
                `need_weights` is True or `cache` is not None. If `need_weights` \
                is True, except for attention output, the tuple also includes \
                the attention weights tensor shaped `[batch_size, num_heads, query_length, key_length]`. \
                If `cache` is not None, the tuple then includes the new cache \
                having the same type as `cache`, and if it is `StaticCache`, it \
                is same as the input `cache`, if it is `Cache`, the new cache \
                reserves tensors concatanating raw tensors with intermediate \
                results of current query.
        """
        identity = query
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = (self.compute_qkv(t, i)
                   for i, t in enumerate([query, key, value]))

        # scale dot product attention
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        scaling = float(self.head_dim)**-0.5
        product = product * scaling

        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        out = paddle.matmul(weights, v)

        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        return out + identity

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
                                                   shape=(bs, 1, num_query, self.num_cams, self.num_points, self.num_levels))

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
            temp_radar_mask = (1.0 - paddle.unsqueeze(radar_mask, -1))*pad_xy
            radar_xy = radar_xy + temp_radar_mask
            # [B, num_query, M]
            ref_radar_dist = []
            for index in range(ref_xy.shape[0]):
                ref_radar_dist.append(-1.0 * cdist(ref_xy[index], radar_xy[index]))
            ref_radar_dist = paddle.to_tensor(ref_radar_dist, dtype='float32')
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
