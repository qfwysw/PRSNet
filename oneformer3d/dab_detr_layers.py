# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import ModuleList
from torch import Tensor
from mmdet.models.layers.transformer.detr_layers import (DetrTransformerDecoder, DetrTransformerDecoderLayer)
from .utils import (MLP, ConditionalAttention, coordinate_to_encoding,
                    inverse_sigmoid)
from .multi_fusion_moe import MoEWeighting

class MyDABDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Implements decoder layer in DAB-DETR transformer."""

    def _init_layers(self):
        """Initialize self-attention, cross-attention, FFN, normalization and
        others."""
        self.self_attn = ConditionalAttention(**self.self_attn_cfg)
        self.cross_attn = ConditionalAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
        self.keep_query_pos = self.cross_attn.keep_query_pos

    def forward(self,
                query: Tensor,
                key: Tensor,
                query_pos: Tensor,
                key_pos: Tensor,
                ref_sine_embed: Tensor = None,
                self_attn_masks: Tensor = None,
                cross_attn_masks: Tensor = None,
                key_padding_mask: Tensor = None,
                is_first: bool = False,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query with shape [bs, num_queries,
                dim].
            key (Tensor): The key tensor with shape [bs, num_keys,
                dim].
            query_pos (Tensor): The positional encoding for query in self
                attention, with the same shape as `x`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            ref_sine_embed (Tensor): The positional encoding for query in
                cross attention, with the same shape as `x`.
                Defaults to None.
            self_attn_masks (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_masks (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
            is_first (bool): A indicator to tell whether the current layer
                is the first layer of the decoder.
                Defaults to False.

        Returns:
            Tensor: forwarded results with shape
            [bs, num_queries, dim].
        """

        query = self.cross_attn(
            query=query,
            key=key,
            query_pos=query_pos,
            key_pos=key_pos,
            ref_sine_embed=ref_sine_embed,
            attn_mask=cross_attn_masks,
            key_padding_mask=key_padding_mask,
            is_first=is_first,
            **kwargs)
        query = self.norms[0](query)
        query = self.self_attn(
            query=query,
            key=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_masks,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query


class MyDABDetrTransformerDecoder(DetrTransformerDecoder):
    """Decoder of DAB-DETR.

    Args:
        query_dim (int): The last dimension of query pos,
            4 for anchor format, 2 for point format.
            Defaults to 4.
        query_scale_type (str): Type of transformation applied
            to content query. Defaults to `cond_elewise`.
        with_modulated_hw_attn (bool): Whether to inject h&w info
            during cross conditional attention. Defaults to True.
    """

    def __init__(self,
                 in_channels, d_model,
                 *args,
                 query_dim: int = 4,
                 query_scale_type: str = 'cond_elewise',
                 with_modulated_hw_attn: bool = True,
                 **kwargs):

        self.query_dim = query_dim
        self.query_scale_type = query_scale_type
        self.with_modulated_hw_attn = with_modulated_hw_attn

        super().__init__(*args, **kwargs)
        # import pdb;pdb.set_trace()
        self.MoEWeighting = MoEWeighting(48)
        self.input_proj = nn.Sequential(
            nn.Linear(48, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.x_mask = nn.Sequential(
            nn.Linear(48, 128), nn.ReLU(),
            nn.Linear(128, d_model))

    def _init_layers(self):
        """Initialize decoder layers and other layers."""
        self.layers = ModuleList([
            MyDABDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        embed_dims = self.layers[0].embed_dims
        self.embed_dims = embed_dims
        if embed_dims == 256:
            self.xyz_dims = int(embed_dims//2*3)
        elif embed_dims == 384:
            self.xyz_dims = int(embed_dims)

        self.post_norm = build_norm_layer(self.post_norm_cfg, embed_dims)[1]
        if self.query_scale_type == 'cond_elewise':
            if embed_dims == 256:
                self.query_scale = MLP(embed_dims, embed_dims, int(embed_dims//2*3), 2)
            elif embed_dims == 384:
                 self.query_scale = MLP(embed_dims, embed_dims, embed_dims, 2)
        elif self.query_scale_type == 'cond_scalar':
            self.query_scale = MLP(embed_dims, embed_dims, 1, 2)
        elif self.query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(self.num_layers, embed_dims)
        else:
            raise NotImplementedError('Unknown query_scale_type: {}'.format(
                self.query_scale_type))

        self.ref_point_head = MLP(128*6, embed_dims,
                                  embed_dims, 2)

        if self.with_modulated_hw_attn and self.query_dim == 6:
            self.ref_anchor_head = MLP(embed_dims, embed_dims, 3, 2)

        self.keep_query_pos = self.layers[0].keep_query_pos
        if not self.keep_query_pos:
            for layer_id in range(self.num_layers - 1):
                self.layers[layer_id + 1].cross_attn.qpos_proj = None
        self.softplus = nn.Softplus()
    def forward(self,
                query: Tensor,
                key: Tensor,
                query_pos: Tensor,
                key_pos: Tensor,
                reg_branches: nn.Module,
                key_padding_mask: Tensor = None,
                mask_feat: Tensor = None,
                backbone: List[Tensor] = None,
                **kwargs) -> List[Tensor]:
        """Forward function of decoder.

        Args:
            query (Tensor): The input query with shape (bs, num_queries, dim).
            key (Tensor): The input key with shape (bs, num_keys, dim).
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            reg_branches (nn.Module): The regression branch for dynamically
                updating references in each layer.
            key_padding_mask (Tensor): ByteTensor with shape (bs, num_keys).
                Defaults to `None`.

        Returns:
            List[Tensor]: forwarded results with shape (num_decoder_layers,
            bs, num_queries, dim) if `return_intermediate` is True, otherwise
            with shape (1, bs, num_queries, dim). references with shape
            (num_decoder_layers, bs, num_queries, 2/4).
        """
        fusion_feat = self.MoEWeighting(query[0], backbone[0])
        # import pdb;pdb.set_trace()
        mask_feat = self.x_mask(fusion_feat)
        key = self.input_proj(fusion_feat)[None, ...]
        output = query
        unsigmoid_references = query_pos

        reference_points = unsigmoid_references
        intermediate_reference_points = [reference_points]
        intermediate = [self.post_norm(output)]
        pred_mask = torch.einsum('nd,md->nm', output.squeeze(0), mask_feat)
        attn_mask = (pred_mask.sigmoid() < 0.5).bool()
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        attn_mask = attn_mask.detach()
        pred_masks = [[pred_mask]]
        ##############################################
        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]
            ref_sine_embed = coordinate_to_encoding(coord_tensor=obj_center, num_feats=128)
            # import pdb;pdb.set_trace()
            query_pos = self.ref_point_head(ref_sine_embed)  # [bs, nq, 2c] -> [bs, nq, c]
            # For the first decoder layer, do not apply transformation
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]
            # apply transformation
            ref_sine_embed = ref_sine_embed[..., :384] * pos_transformation
            # modulated height and weight attention
            if self.with_modulated_hw_attn:
                assert obj_center.size(-1) == 6
                # refx, refy, refz
                ref_hw = torch.clamp(self.ref_anchor_head(output), min=1e-8).sigmoid()
                ref_sine_embed[..., 256:] *= (ref_hw[..., 0] / obj_center[..., 3]).unsqueeze(-1)
                ref_sine_embed[..., 128:256] *= (ref_hw[..., 1] / obj_center[..., 4]).unsqueeze(-1)
                ref_sine_embed[..., :128] *= (ref_hw[..., 2] / obj_center[..., 5]).unsqueeze(-1)

            output = layer(
                output,
                key,
                query_pos=query_pos,
                ref_sine_embed=ref_sine_embed,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                is_first=(layer_id == 0),
                cross_attn_masks=attn_mask,
                **kwargs)
            #################################################################
            # fusion_feat = self.MoEWeighting(query[0], backbone[0].clone().detach())
            # mask_feat = self.x_mask(fusion_feat)
            # key = self.input_proj(fusion_feat)[None, ...]
            #################################################################
            # import pdb;pdb.set_trace()
            pred_mask = torch.einsum('nd,md->nm', output.squeeze(0), mask_feat)
            attn_mask = (pred_mask.sigmoid() < 0.5).bool()
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            attn_mask = attn_mask.detach()
            # del attn_mask
            # iter update
            tmp_reg_preds = reg_branches[layer_id](output)
            tmp_reg_preds[..., :3] += inverse_sigmoid(reference_points[..., :3])
            tmp_reg_preds[..., 3:] += inverse_sigmoid(reference_points[..., 3:])
            new_reference_points = tmp_reg_preds.clone()
            new_reference_points[..., :3] = tmp_reg_preds[..., :3].sigmoid()
            # 计算 sigmoid 后的值
            sigmoid_vals = torch.clamp(tmp_reg_preds[..., 3:], min=1e-8).sigmoid()
            new_reference_points[..., 3:] = sigmoid_vals
            # import pdb;pdb.set_trace()
            # if layer_id != self.num_layers - 1:
            intermediate_reference_points.append(new_reference_points)
            intermediate.append(self.post_norm(output))
            pred_masks.append([pred_mask])
            reference_points = new_reference_points.detach()

            # if self.return_intermediate:

        output = self.post_norm(output)

        if self.return_intermediate:
            return [
                torch.stack(intermediate),
                torch.stack(intermediate_reference_points),
                pred_masks,
            ]
        else:
            return [
                output.unsqueeze(0),
                torch.stack(intermediate_reference_points)
            ]

