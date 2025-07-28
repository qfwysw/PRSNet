"""
Point Transformer V2 Mode 2 (recommend)

Disable Grouped Linear

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr

import einops
from timm.models.layers import DropPath
import pointops
from mmdet3d.registry import MODELS
from .mamba import Mamba, MambaConfig
from .query_decoder import FFN

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        通用 MLP 模型
        :param input_size: 输入层大小（特征维度）
        :param hidden_sizes: 隐藏层大小列表（支持多层）
        :param output_size: 输出层大小（类别数）
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size  # 记录上一层的大小
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())  # 使用 ReLU 激活函数
            prev_size = hidden_size  # 更新上一层大小
        
        # 添加输出层（无激活函数，适用于分类问题）
        layers.append(nn.Linear(prev_size, output_size))
        
        # 将所有层组合成一个 `Sequential` 模型
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平成一维
        return self.model(x)

def gaussian_distance(dist, sigma):
    """
    Convert distances to Gaussian weights.

    :param dist: Tensor of distances.
    :param sigma: Standard deviation for the Gaussian function.
    :return: Tensor of Gaussian weights.
    """
    return torch.exp(-dist**2 / (2 * sigma**2))

def offset2batch(offset):
    return (
        torch.cat(
            [
                torch.tensor([i] * (o - offset[i - 1]))
                if i > 0
                else torch.tensor([i] * o)
                for i, o in enumerate(offset)
            ],
            dim=0,
        )
        .long()
        .to(offset.device)
    )


def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        # self.norm = nn.BatchNorm1d(embed_channels)
        self.norm = nn.LayerNorm(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError

class GroupedVectorAttention(nn.Module):
    def __init__(
        self,
        embed_channels,
        neighbours,
    ):
        super(GroupedVectorAttention, self).__init__()
        config = MambaConfig(d_model=embed_channels, n_layers=1)
        self.mamba_layers = Mamba(config)
        self.neighbours = neighbours
        self.linear_attention = nn.Sequential(
            nn.Linear(embed_channels * 2, embed_channels),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),  # 启用inplace激活，减少内存占用
            nn.Linear(embed_channels, embed_channels),
        )
        # self.linear_attention1 = nn.Sequential(
        #     nn.Linear(embed_channels, embed_channels)
        # )
        self.norm = PointBatchNorm(embed_channels)

    def forward(self, feat, coord, reference_index, gauss_dist, j):
        # 分组特征，形状为 [N, K, C]
        group_feat = pointops.grouping(reference_index, feat, coord, with_xyz=False)
        
        # 利用广播机制直接加权，无需显式扩展维度
        weighted_feat = gauss_dist.unsqueeze(-1) * group_feat
        
        # 合并邻居特征并归一化
        combined_feat = self.norm(weighted_feat.sum(dim=1))  # 形状 [N, C]
        # combined_feat = weighted_feat.sum(dim=1)  # 形状 [N, C]
        # Mamba层处理，保持unsqueeze/squeeze以适应输入格式
        feat_mamba = self.norm(self.mamba_layers(feat.unsqueeze(0)).squeeze(0))

        # 拼接特征并通过线性层
        cat_feat = torch.cat([feat_mamba, combined_feat], dim=-1)
        res_feat = self.linear_attention(cat_feat)
        
        return res_feat


class Block(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        neighbours=8, 
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            neighbours=neighbours,
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, points, reference_index, gauss_dist, j):
        coord, feat, offset, gauss_r = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = (
            self.attn(feat, coord, reference_index, gauss_dist, j)
            if not self.enable_checkpoint
            else checkpoint(self.attn, feat, coord, reference_index)
        )
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset, gauss_r]


class BlockSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        # self.adaptive_r = nn.Sequential(
        #     nn.Linear(embed_channels, embed_channels),
        #     PointBatchNorm(embed_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(embed_channels, embed_channels),
        #     PointBatchNorm(embed_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(embed_channels, 1),
        #     nn.Sigmoid(),
        # )
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                neighbours=neighbours,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint,
            )
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset, gauss_r = points
        # pos_min, pos_max = coord.min(0)[0], coord.max(0)[0]
        # xyz = (pos_max - pos_min).data
        # import pdb;pdb.set_trace()
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        reference_index, dist = pointops.knn_query(self.neighbours, coord, offset)
        # # dist: N, 8
        # # 假设 dist 形状为 (N, 8)
        # # gauss_r 形状为 (N, 1)
        # gauss_r = self.adaptive_r(feat)  # N, 1

        # 计算高斯衰减权重
        gaussian_weights = torch.exp(-dist**2 / (2 * gauss_r**2))  # N, 8

        # import pdb;pdb.set_trace()
        # 生成掩码：超出 gauss_r 的部分设为 0
        mask = dist <= gauss_r  # N, 8（布尔掩码）

        # 应用掩码
        gauss_dist = gaussian_weights * mask  # N, 8

        # gauss_dist = 1. - dist / dist[:, [-1]]
        # gauss_dist[:, 7] = gauss_dist[:, 6] / 2.
        for j, block in enumerate(self.blocks):
            points = block(points, reference_index, gauss_dist, j)
            # print(j, points[0].shape)
        return points


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset, gauss_r = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = (
            segment_csr(
                coord,
                torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                reduce="min",
            )
            if start is None
            else start
        )
        cluster = voxel_grid(
            pos=coord - start[batch], size=self.grid_size, batch=batch, start=0
        )
        unique, cluster, counts = torch.unique(
            cluster, sorted=True, return_inverse=True, return_counts=True
        )
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        gauss_r = segment_csr(gauss_r[sorted_cluster_indices], idx_ptr, reduce="max")
        # import pdb;pdb.set_trace()
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset, gauss_r], cluster


class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        bias=True,
        skip=True,
        backend="map",
    ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, skip_points, cluster=None):
        coord, feat, offset, gauss_r = points
        skip_coord, skip_feat, skip_offset, skip_gauss_r = skip_points
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster]
            gauss_r = gauss_r[cluster]
        else:
            feat = pointops.interpolation(
                coord, skip_coord, self.proj(feat), offset, skip_offset
            )
        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_offset, gauss_r]


class Encoder(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        grid_size=None,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        enable_checkpoint=False,
    ):
        super(Encoder, self).__init__()

        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        points, cluster = self.down(points)
        return self.blocks(points), cluster


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        groups,
        depth,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        enable_checkpoint=False,
        unpool_backend="map",
    ):
        super(Decoder, self).__init__()

        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points, skip_points, cluster):
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class GVAPatchEmbed(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        coord, feat, offset, gauss_r = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset, gauss_r])

@MODELS.register_module()
class PointTransformerV2MambaVoxel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.06, 0.12, 0.24, 0.48),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        enable_checkpoint=False,
        unpool_backend="map",
    ):
        super(PointTransformerV2MambaVoxel, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)
        self.adaptive_r = nn.Sequential(
            nn.Linear(patch_embed_channels, patch_embed_channels),
            PointBatchNorm(patch_embed_channels),
            nn.ReLU(inplace=True),
            nn.Linear(patch_embed_channels, patch_embed_channels),
            PointBatchNorm(patch_embed_channels),
            nn.ReLU(inplace=True),
            nn.Linear(patch_embed_channels, 1),
            nn.Softplus(beta=1)
        )
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint,
        )

        enc_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))
        ]
        dec_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))
        ]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        self.linear_list = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[
                    sum(enc_depths[:i]) : sum(enc_depths[: i + 1])
                ],
                enable_checkpoint=enable_checkpoint,
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[
                    sum(dec_depths[:i]) : sum(dec_depths[: i + 1])
                ],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend,
            )
            linear = nn.Linear(dec_channels[len(dec_channels)-1-1-i], 48, bias=False)

            self.enc_stages.append(enc)
            self.dec_stages.append(dec)
            if i == 0:
                self.linear_list.append(None)
            else:
                self.linear_list.append(linear)

        # self.MoEWeighting = MoEWeighting(dec_channels[0])
    def forward(self, data_dict):
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()
        gauss_r = torch.ones_like(coord[:, 0:1]) * 0.1
        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset, gauss_r]
        points = self.patch_embed(points)
        gauss_r = self.adaptive_r(points[1])  # N, 1
        # print(gauss_r)
        points[-1] = gauss_r
        skips = [[points]]
        cluster_list = []
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling
            cluster_list.append(cluster)
            skips.append([points])  # record points info of current stage

        cluster_list.pop(-1)
        cluster_list.reverse()
        points = skips.pop(-1)[0]  # unpooling points info in the last enc stage
        points_list = []
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster)
            if i != self.num_stages - 1:
                points_list.append(points[1])
            else:
                points_list.append(points[1].clone().detach())
        # import pdb;pdb.set_trace()
        multi_points = []
        for pi in range(len(points_list)):
            if pi == 0:
                continue
            points = points_list[pi]
            for cj in range(pi, len(cluster_list)):
                points = points[cluster_list[cj]]
            points = self.linear_list[pi](points)
            multi_points.append(points)

        # import pdb;pdb.set_trace()
        # multi_futrue = self.MoEWeighting(torch.stack(multi_points, dim=1))
        # return multi_futrue

        return torch.stack(multi_points, dim=1)