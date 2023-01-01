import paddle.nn as nn
import paddle.nn.functional as F
import paddle

from math import sqrt


def _convert_attention_mask(attn_mask, dtype):
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


class MultiHeadSelfAttention(nn.Layer):
    def __init__(self, embed_dim, num_heads, attn_drop=0., proj_drop=0., batch_first=False, dim_k=None, dim_v=None):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        if dim_k == None:
            self.dim_k = embed_dim
        if dim_v == None:
            self.dim_v = embed_dim
        self.scale = 1 / sqrt(self.dim_k // num_heads)

        self.linear_q = nn.Linear(embed_dim, embed_dim, weight_attr=paddle.nn.initializer.Constant(0.02),
                                  bias_attr=paddle.nn.initializer.Constant(0.))
        self.linear_k = nn.Linear(embed_dim, embed_dim, weight_attr=paddle.nn.initializer.Constant(0.02),
                                  bias_attr=paddle.nn.initializer.Constant(0.))
        self.linear_v = nn.Linear(embed_dim, embed_dim, weight_attr=paddle.nn.initializer.Constant(0.02),
                                  bias_attr=paddle.nn.initializer.Constant(0.))
        self.out = nn.Linear(embed_dim, embed_dim, weight_attr=paddle.nn.initializer.Constant(0.02),
                             bias_attr=paddle.nn.initializer.Constant(0.))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key=None, value=None):
        # convert to batch first
        if not self.batch_first:
            query = query.transpose((1, 0, 2))
        batch, n, dim_in = query.shape
        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head
        q = self.linear_q(query).reshape((batch, n, nh, dk)).transpose((0, 2, 1, 3))  # (batch, nh, n, dk)
        k = self.linear_k(key).reshape((batch, n, nh, dk)).transpose((0, 2, 1, 3))  # (batch, nh, n, dk)
        v = self.linear_v(value).reshape((batch, n, nh, dv)).transpose((0, 2, 1, 3))  # (batch, nh, n, dv)

        dist = paddle.matmul(q, k.transpose((0, 1, 3, 2))) * self.scale  # batch, nh, n, n
        dist = F.softmax(dist)  # batch, nh, n, n
        dist = self.attn_drop(dist)

        att = paddle.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose((0, 2, 1, 3)).reshape((batch, n, self.dim_v))  # batch, n, dim_v
        out = self.out(att)
        out = self.proj_drop(out).transpose((1, 0, 2))
        # att = self.out(att).transpose((1, 0, 2))
        return out
