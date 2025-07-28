import torch
import torch.nn as nn
import math
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from .dab_detr_layers import MyDABDetrTransformerDecoder
from mmcv.cnn import Linear
from .utils import inverse_sigmoid

def save_to_ply(filename, vertices):
    with open(filename, 'w') as file:
        # 写入 PLY 文件头
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(vertices)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")

        # 写入顶点数据
        for vertex in vertices:
            file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_z, pos_y, pos_x), dim=-1)
    # posemb = torch.cat((pos_x, pos_y, pos_z), dim=-1)
    return posemb

class CrossAttentionLayer(BaseModule):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # todo: why BaseModule doesn't call it without us?
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(sources)):
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, queries_pos=None):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for i, y in enumerate(x):
            if queries_pos is not None:
                q = k = y + queries_pos[i]
            else:
                q = k = y
            z, _ = self.attn(q, k, y)
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out

@MODELS.register_module()
class DabDecoder(BaseModule):
    """Query decoder. The same as above, but for 2 datasets.

    Args:
        num_layers (int): Number of transformer layers.
        num_queries_1dataset (int): Number of queries for the first dataset.
        num_queries_2dataset (int): Number of queries for the second dataset.
        num_classes_1dataset (int): Number of classes in the first dataset.
        num_classes_2dataset (int): Number of classes in the second dataset.
        prefix_1dataset (string): Prefix for the first dataset.
        prefix_2dataset (string): Prefix for the second dataset.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, 
                 num_layers, 
                 num_queries_1dataset, 
                 num_classes_1dataset, 
                 prefix_1dataset,
                 in_channels, 
                 d_model, 
                 decoder,
                 num_heads, 
                 hidden_dim,
                 dropout, 
                 activation_fn, 
                 iter_pred, 
                 attn_mask, 
                 fix_attention, 
                 **kwargs):
        super().__init__()
        

        self.num_queries_1dataset = num_queries_1dataset
        self.reference_points = nn.Embedding(self.num_queries_1dataset, 6)
        nn.init.uniform_(self.reference_points.weight.data[..., :3], 0, 1)
        # 0.5 ~ 1 
        nn.init.uniform_(self.reference_points.weight.data[..., 3:], 0.5, 1)
        self.queries_1dataset = nn.Embedding(num_queries_1dataset, d_model)
        
        self.prefix_1dataset = prefix_1dataset 

        self.decoder = MyDABDetrTransformerDecoder(in_channels=in_channels, d_model=d_model, **decoder)
        self.out_cls = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_classes_1dataset + 1))
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.outnorm = nn.LayerNorm(d_model)
        
        
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.num_classes_1dataset = num_classes_1dataset 
        self.d_model = d_model
        # self.num_classes_2dataset = num_classes_2dataset
        self.num_reg_fcs = 2
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(d_model, d_model))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(d_model, 6))
        reg_branch = nn.Sequential(*reg_branch)
        self.generate_points = nn.Sequential(*reg_branch)
        self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_layers)])
        self.position_encoder = nn.Sequential(
                nn.Linear(3, self.d_model*4),
                nn.ReLU(),
                nn.Linear(self.d_model*4, self.d_model),
            )
        # self.softplus = nn.Softplus()

    def _get_queries(self, batch_size):
        """Get query tensor.

        Args:
            batch_size (int, optional): batch size.
            scene_names (List[string]): list of len batch size, which 
                contains scene names.
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        
        result_queries = []
        result_queries_pos = []
        reference_points = self.reference_points.weight
        queries = self.queries_1dataset.weight
        for i in range(batch_size):
            result_queries.append(queries)
            result_queries_pos.append(reference_points)

        return result_queries, result_queries_pos
    
    def position_embedding(self, pos, input_range):
        min_pos, max_pos = input_range[0]
        pos = (pos - min_pos) / (max_pos - min_pos)
        pos = inverse_sigmoid(pos)
        
        pos_emb = self.position_encoder(pos)
        
        return pos_emb
        

    def forward_iter_pred(self, x, pos, batch_offsets):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            scene_names (List[string]): list of len batch size, which 
                contains scene names.
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        prediction_labels = []
        prediction_masks = []
        prediction_scores = []
        prediction_bboxes = []

        # inst_feats = [self.input_proj(y) for y in x]
        # mask_feats = [self.x_mask(y) for y in x]
        queries, queries_pos = self._get_queries(len(x))
        
        input_ranges = []
        # coord_save = pos.detach().cpu().numpy()
        # save_to_ply("ply/output1.ply", coord_save)
        # coord_save = pos.sigmoid().detach().cpu().numpy()
        # save_to_ply("ply/output2.ply", coord_save)
        # import sys
        # sys.exit()
        pos_min, pos_max = pos.min(0)[0], pos.max(0)[0]
        input_ranges.append((pos_min, pos_max))
        pos_emb_i = self.position_embedding(pos, input_ranges)
        # queries_pos = self.generate_points(queries[0])
        intermediate_results, ref_points, pred_masks = self.decoder(query=queries[0][None, ...],
            query_pos=queries_pos[0][None, ...],
            key=None,
            key_pos=pos_emb_i[None, ...],
            reg_branches=self.reg_branches,
            # coord=pos,
            mask_feat=None,
            backbone=x
            # input_ranges=input_ranges
            )

        for i in range(intermediate_results.shape[0]):
            ouptut_i = intermediate_results[i]
            pred_labels, pred_scores, pred_box = self.prediction_head(ouptut_i, input_ranges, ref_points[i])
            prediction_labels.append(pred_labels)
            prediction_scores.append(pred_scores)
            prediction_bboxes.append(pred_box)
            prediction_masks.append(pred_masks[i])

        # import pdb;pdb.set_trace()
        return {
            'labels':
            pred_labels,
            'masks':
            pred_masks[-1],
            'scores':
            pred_scores,
            'bboxes':
            prediction_bboxes[-1],
            'aux_outputs': [{
                'labels': a,
                'masks': b,
                'scores': c,
                'bboxes': d,
            } for a, b, c, d in zip(
                prediction_labels[:-1],
                prediction_masks[:-1],
                prediction_scores[:-1],
                prediction_bboxes[:-1],
            )],
        }

    def prediction_head(self, query, input_ranges, ref_points):
        # query = self.outnorm(query)
        pred_labels = self.out_cls(query)
        pred_scores = self.out_score(query)
        pred_xyz = ref_points.clone()
        for i, input_range in enumerate(input_ranges):
            min_xyz_i, max_xyz_i = input_range
            pred_xyz[i][..., :3] = ref_points[i][..., :3] * (max_xyz_i - min_xyz_i) + min_xyz_i
            pred_xyz[i][..., 3:] = inverse_sigmoid(ref_points[i][..., 3:])
            
        # pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_offsets)
        return pred_labels, pred_scores, pred_xyz
    def forward(self, x, pos, batch_offsets):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            scene_names (List[string]): list of len batch size, which 
                contains scene names.
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        return self.forward_iter_pred(x, pos, batch_offsets)

