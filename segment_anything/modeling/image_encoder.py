# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock
from Mycodes.adapter import Adapter_Layer

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,         # Transformer Encoder 的层数
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # 绝对位置编码
        self.pos_embed: Optional[nn.Parameter] = None
        # 使用预训练图像大小初始化绝对位置嵌入
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )         # B H W C

        # Transformer Block定义
        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_adapter = False  # 是否加adapter
            # use_adapter = i in [0, 2, 4]  # 仅第 0/2/4 层加 Adapter
            block = Block(
                # 输入的通道数（每个patch编码后的向量维度）
                dim=embed_dim,
                # 自注意力机制中的注意力头数
                num_heads=num_heads,
                # MLP层的通道数相对于输入通道数的比例
                mlp_ratio=mlp_ratio,
                # 是否在QKV全连接层中使用偏置
                qkv_bias=qkv_bias,
                # 归一化层
                norm_layer=norm_layer,
                # 激活函数
                act_layer=act_layer,
                # 是否使用相对位置编码
                use_rel_pos=use_rel_pos,
                # 相对位置编码的初始化设置
                rel_pos_zero_init=rel_pos_zero_init,
                # 如果当前 Block 不是全局注意力层，则使用窗口大小，否则使用0
                window_size=window_size if i not in global_attn_indexes else 0,
                # 记录block层数
                block_id = i,
                # 输入特征的尺寸，基于原始图像大小和patch大小计算得出
                input_size=(img_size // patch_size, img_size // patch_size),
                use_adapter=use_adapter
            )
            self.blocks.append(block)

        # 将通道数降低至256，生成最终的image embedding
        # 包含2个卷积层和2个归一层
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,   # 1 x 1的卷积层，将输入通道数从embed_dim减小到out_chans，而不改变特征图的空间尺寸
                bias=False,      # 不使用偏置项
            ),
            LayerNorm2d(out_chans),   # 归一化层，用于规范化输出通道的均值和方差，提高模型的稳定性和收敛速度
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,     # 3 x 3卷积层
                padding=1,         # 保持输入和输出特征图尺寸不变
                bias=False,
            ),
            LayerNorm2d(out_chans),  # 第二个归一化层，再次规范化输出
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"input：{x.shape}")
        x = self.patch_embed(x)
        # print(f"patch_embed：{x.shape}")
        if self.pos_embed is not None:
            x = x + self.pos_embed    # 添加位置编码
        # print(f"pos_embed：{x.shape}")

        for blk in self.blocks:
            x = blk(x)
        # print(f"blocks：{x.shape}")

        # [B,H,W,C] -> [B,C,H,W]-> neck
        x = self.neck(x.permute(0, 3, 1, 2))
        return x

# Encoder Block
class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,                              # 输入通道数
        num_heads: int,                        # 注意力头个数
        mlp_ratio: float = 4.0,                # MLP层的隐藏层维度（通道数）相对于输入通道数dim的比例
        qkv_bias: bool = True,                 # 如果为True，QKV全连接层包含偏置
        norm_layer: Type[nn.Module] = nn.LayerNorm,     # 归一化层，默认LayerNorm
        act_layer: Type[nn.Module] = nn.GELU,           # 激活层，默认 GELU
        use_rel_pos: bool = False,             # 不使用相对位置编码
        rel_pos_zero_init: bool = True,        # 相对位置编码的初始化设置
        window_size: int = 0,                 # 注意力层的窗口大小为0：代表使用全局注意力
        block_id: int = 0,
        input_size: Optional[Tuple[int, int]] = None,        # 输入特征的尺寸
        use_adapter:bool = True
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.block_id = block_id
        # 第一个归一化层，用于multi-head attention
        self.norm1 = norm_layer(dim)
        # # 注意力机制（可以是窗口注意力或全局注意力）
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            # window_size = 0 使用全局注意力，特征图大小不变
            # window_size > 0 使用局部注意力，特征图划分为windows_size x windows_size大小窗口
            block_id=block_id,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        # 加入adapter模块
        self.adapter = Adapter_Layer(dim) if use_adapter else nn.Identity()

        # 第二个归一化层，用于MLP之前
        self.norm2 = norm_layer(dim)
        #  MLP 层（linear + GELU激活 + Linear）
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        # 记录窗口大小
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"第{self.block_id + 1}层block")
        # 保存输入张量副本
        shortcut = x
        # 对输入张量应用第一个归一化层
        x = self.norm1(x)
        # 如果使用窗口注意力，则进行窗口划分（Window Partition）
        if self.window_size > 0:
            # 获取特征图高度和宽度
            H, W = x.shape[1], x.shape[2]
            # 进行窗口划分
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)          # 执行注意力机制（windows attention 或 global attention)
        # 如果使用窗口注意力，则进行窗口合并（Reverse window partition）
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        # 加adapter
        x = self.adapter(x)

        # 第一层残差连接（Attention后）
        x = shortcut + x

        # 第二层归一化 + MLP计算
        x = x + self.mlp(self.norm2(x))

        return x

# multi-head attention
class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,    # 实际传入的num_heads = 12
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        block_id: int = 0,
        # 指定相对位置编码的尺寸，只有在使用相对位置编码时才需要
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.block_id = block_id
        # 输入head数目
        self.num_heads = num_heads
        # 每个head维度
        head_dim = dim // num_heads
        # 用于缩放注意力得分的因子，以避免数值溢出，取值为head_dim的平方根的倒数
        self.scale = head_dim**-0.5

        # 一个全连接层（nn.Linear），将输入映射到Q、K、V的组合
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 一个全连接层，用于将注意力机制的输出投影回原始维度
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        # 使用相对位置编码
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            """
            在初始化水平方向rel_pos_h和垂直方向rel_pos_w的相对位置嵌入
            输入尺寸为(H,W)，则水平方向的位置嵌入长度为2*H-1，垂直方向的位置嵌入长度为2*W-1，每个位置嵌入的维度为head_dim
            nn.Parameter表示在训练过程中参数会更新
            """
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入张量x的形状为(B,H,W,C)，其中B是批次大小，H和W是高度和宽度，C是通道数（即dim）
        B, H, W, _ = x.shape
        # print(f"第{self.block_id + 1}次attention:{np.shape(x)}")
        # 使用qkv层将x转换为Q、K、V的组合，然后通过重塑和重新排列来准备多头注意力计算
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        # print(f"第{self.block_id + 1}次qkv:{np.shape(qkv)}")

        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        """
        self.scale = head_dim**-0.5
        q * self.scale 稳定计算防止梯度消失
        transpose(-2,-1)：对k进行转置操作，将最后一个和倒数第二个维度互换，使q和k在计算点积时的维度匹配
        k的形状由（B * nHead, H * W, C） -> （B * nHead, C, H * W）
        最终attn with shape(B*nHead, H*W, H*W)
        """
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        # 注意力分数矩阵attn随后会经过softmax函数，将每个位置的分数归一化到[0,1]概率区间
        attn = attn.softmax(dim=-1)
        """
        @：矩阵乘法     v：向量值 (B * nHead, H * W, C)    attn @ v：得到加权后的值向量(B * nHead, H * W, C) 
        .view：将加权后的值向量重塑为(B, self.num_heads, H, W, -1)
        .permute：重排为 (B,  H, W, self.num_heads, -1)
        reshape：重塑为 (B,  H, W, C)，与输入张量形状一致
        """
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        # 使用全连接层（dim, dim）对张量 x 进行线性投影，恢复到原始的特征维度作为注意力模块的输出
        x = self.proj(x)

        return x

# 非全局注意力Block定义：W-MSA的窗口划分
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    # 获取输入张量形状
    B, H, W, C = x.shape
    # 计算填充高度 pad_h 和宽度 pad_w，使得输入尺寸能被 window_size 整除
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    # 如果需要填充，使用F.pad函数在宽度和高度方向上进行填充
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    # 更新填充后张量的高度和宽度Hp和Wp
    Hp, Wp = H + pad_h, W + pad_w
    # 张量重塑为：B, Hp/S, S, Wp/S, S, C
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    # 调整张量的形状，使其由B, Hp/S, Wp/S, S, S, C-->B*Hp*Wp/(S*S), S, S, C
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # 返回一个包含所有窗口的张量和原始张量的填充后尺寸(Hp, Wp)
    return windows, (Hp, Wp)

# 将window_partition函数分割的窗口重新组合回原始尺寸的张量
def window_unpartition(
    # 获取输入张量 windows 的形状，以及窗口大小 window_size
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    # 原始尺寸的填充高度和宽度
    Hp, Wp = pad_hw
    # 原始尺寸的无填充高度和宽度
    H, W = hw
    # 从窗口张量的总大小中计算出原始批量大小
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    # 重塑窗口张量：B*Hp*Wp/(S*S),S,S,C-->B,Hp/S,Wp/S,S,S,C
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    # 再次重塑张量：B,Hp/S,Wp/S,S,S,C-->B,Hp,Wp,C
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    # 如果原始尺寸小于填充后的尺寸
    if Hp > H or Wp > W:
        # 通过切片x[:,:H,:W,:]去除填充部分，只保留原始大小的区域，
        x = x[:, :H, :W, :].contiguous()
    # 返回合并后的张量，形状为B，H，W，C，即原始的批量大小、高度、宽度和通道数
    return x

# 表示quary与key在二维空间中的最大相对距离
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    # 相对位置最大距离
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    # 如果rel_pos的形状的第0个维度（即长度）不等于 max_rel_dist，说明需要插值
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,     # 插值目标长度
            mode="linear",         # 插值方法为线性插值
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

# 为atten注意力特征添加相对位置的嵌入特征
def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    # q 和 k 在高度和宽度方向上的相对位置编码
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    # 重塑 q 为（B，q_h，q_w，dim）
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

# RGB图像 -> patch -> 嵌入向量768维度
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),       # 被划分为 16 x 16 的patches
        stride: Tuple[int, int] = (16, 16),            # 卷积的步长，与kernel_size相同，即(16,16)
        padding: Tuple[int, int] = (0, 0),             # 无边缘填充
        in_chans: int = 3,                             # RGB图像
        embed_dim: int = 768,                          # 每个patch的向量长度，即输出的特征维度
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 卷积，将输入通道数从in_chans 转换为 embed_dim
        x = self.proj(x)
        # 将张量的维度顺序从 B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x      # [B,H/16,W/16,768]

# if __name__ == "__main__":
#     import numpy as np
#     import os
#
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     image = torch.rand(size=(1, 3, 1024, 1024))
#     image = image.cuda()
#
#     model = ImageEncoderViT().cuda()
#     output = model(image)
#     print(f"output：{output.shape}")

    # # 绘制模型图
    # import hiddenlayer as hl
    #
    # g = hl.build_graph(model, image,
    #                    transforms=None)
    # g.save("D:/project/segment-anything/dataset/network_architecture.pdf")
    # del g