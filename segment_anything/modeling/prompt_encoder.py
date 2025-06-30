# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,    #   256， 与imageEncoderViT最终输出的通道数对齐
        image_embedding_size: Tuple[int, int],    # 输入特征图尺寸
        input_image_size: Tuple[int, int],        # pad后的图像尺寸
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        # point_embedding: [1,embed_dim]
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        # 无效点
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        # Conv2d -> Norm -> Conv2d -> Norm -> GELU -> Conv2d
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        # [C,H,W]->[1,C,H,W]
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    # 点嵌入，[B, N, 2]
    def _embed_points(
        self,
        points: torch.Tensor,     # points形状[B, N, 2]，依次表示批量大小，点个数，每个点坐标（x,y）
        labels: torch.Tensor,     # 形状[B, N]， 1前景点（正样本）, 0背景点（负样本）, -1填充点（无效点,补齐维度）
        pad: bool,                # 是否需要无效点填充，确保batch中每一张图像的提示点数目相同，输入形状一致
    ) -> torch.Tensor:
        """Embeds point prompts."""
        # SAM采用离散的像素坐标，像素中心在整数位置的0.5偏移处
        points = points + 0.5  # Shift to center of pixel
        # points和boxes联合则不需要pad
        if pad:
            # 创建padding_point，形状[B, 1, 2], 填充全零（0，0）
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            # 创建padding_label，形状[B,1], 值-1（无效点）
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            # 拼接points 和labels，确保输入大小一致
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        # 对点坐标进行位置编码
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        # 处理填充点
        # label = -1时，嵌入向量清零（0，0）
        point_embedding[labels == -1] = 0.0
        # self.not_a_point_embed.weight：确保填充点不会影响后续计算
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        # 前景点1和背景点0嵌入
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        # 返回嵌入点[B, N, D]
        return point_embedding

    # box 嵌入，初始形状[B, 4]
    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        # 对齐像素
        boxes = boxes + 0.5  # Shift to center of pixel
        """
        变换形状[B, 4] ->  [B, 2, 2]两个点 (x_min, y_min) 和 (x_max, y_max)
        coords[:, 0, :] = (x_min, y_min)  # 左上角坐标
        coords[:, 1, :] = (x_max, y_max)  # 右下角坐标
        """
        coords = boxes.reshape(-1, 2, 2)
        # 坐标嵌入：2D坐标 -> 高维特征 形状变化[B, 2, 2] ->  [B, 2, D]
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        # 左上角(x_min, y_min)位置嵌入
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        # 右下角（x_max, y_max）位置嵌入
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        # 返回box嵌入 [B, 2, D], 包含 (x_min, y_min) 和 (x_max, y_max) 的嵌入
        return corner_embedding

    # mask嵌入 [B, 1, H, W]
    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        # 降维，变成低维嵌入 [B, D, H/4, W/4], 这里的HW为256
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    # 计算批次大小
    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],   # （points坐标，points坐标）
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        优先顺序：点，框，掩码
        当无任何提示时，默认batch_size=1
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    # 获取模型所在设备
    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        # 确定批量大小batch_size
        bs = self._get_batch_size(points, boxes, masks)
        # 初始化创建一个sparse_embeddings,形状[B，0，D]
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        #处理点提示
        if points is not None:
            coords, labels = points    # points:tuple 坐标[B，N，2]，label[B, N]
            # 点嵌入，形状 [B, N, D]
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))   # 没有box时进行点填充
            # 将point_embeddings拼接到sparse-embeddings
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)  # 形状 [B, 2, D]
            """
            拼接 Box 提示到sparse_embeddings
            如果已经有点提示，则变成 [B, N+2, D]
            如果没有点提示，则变成 [B, 2, D]
            """
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            # 计算mask嵌入，形状 [B, D, H/4, W/4] =256/4
            dense_embeddings = self._embed_masks(masks)
        else:
            # 如果mask为空，[1,D]->[1, D, 1, 1]，然后 expand 复制到 [B, D, H/16, W/16]，填充整个特征图。
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings

# 随机位置编码PE：随机高斯投影方法
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    num_pos_feats：位置编码维度，通常是64
    scale：用于调整高斯投影的缩放因子，默认为1.0
    """
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        # 随机初始化的高斯矩阵，形状 [2, num_pos_feats]
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    # 对输入的2D坐标进行位置编码——点提示/特征图编码
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        # 将坐标变换到 [-1, 1]，适应sin/cos的周期性投影
        # 坐标shape要求最后一个维度为2，[B,N,2]/[H,W,2]
        coords = 2 * coords - 1
        # 应用随机高斯投影矩阵[2，num_pos_feats]， 映射到高维 num_pos_feats 维度， [B, N, num_pos_feats]
        coords = coords @ self.positional_encoding_gaussian_matrix
        # 周期映射，将点映射到正弦空间中
        coords = 2 * np.pi * coords
        # 对每个点的编码，分别取sin和cos
        # 形状[B, N, 2 * num_pos_feats]/[H,W,2*num_pos_feats]
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    # 创建坐标网络（x, y), 对整个图像进行位置编码——mask或图像特征图
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        # 构建网格坐标[H,W]
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        # x,y方向位置索引，像素对齐
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        # 归一化 [0,1]
        y_embed = y_embed / h
        x_embed = x_embed / w
        # 位置编码 [H,W] -> [H, W, 2] -> [H, W, C]
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        # 变换形状  [C, H, W]
        return pe.permute(2, 0, 1)  # C x H x W

    # 归一化——点提示 框提示
    # image_size (1024,1024)
    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        # 复制坐标，防止修改原数据
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]  # 归一化 x 坐标 [0,1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]  # 归一化 y 坐标
        # 将归一化后的点坐标(x,y)送入self._pe_encoding
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
