import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEWeighting(nn.Module):
    def __init__(self, input_dim, num_experts=3, top_k=2):
        super(MoEWeighting, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 专家网络：每个专家生成一个权重
        self.experts = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_experts)])
        
        # 门控网络：根据 query 生成专家权重
        self.gate = nn.Linear(384, num_experts)

    def forward(self, query, features):
        """
        Args:
            query: (m, d) - 查询向量
            features: (n, num_experts, d) - 多尺度特征（每个点有 num_experts 个尺度）
        Returns:
            output: (n, d) - 加权后的特征
        """
        # import pdb;pdb.set_trace()
        n, num_experts, d = features.shape
        
        # 1. 门控网络生成全局专家权重（基于 query 的全局信息）
        aggregated_query = query.mean(dim=0, keepdim=True)  # (1, d)
        gate_scores = self.gate(aggregated_query)  # (1, num_experts)
        gate_weights = F.softmax(gate_scores, dim=-1).squeeze(0)  # (num_experts,)
        
        # 2. 选择 Top-K 专家
        top_k_indices = torch.topk(gate_weights, self.top_k)[1]  # (top_k,)
        # import pdb;pdb.set_trace()
        top_k_mask = torch.zeros_like(gate_weights).scatter_(0, top_k_indices, 1)
        gate_weights = gate_weights * top_k_mask
        gate_weights = gate_weights / gate_weights.sum()  # 归一化
        # import pdb;pdb.set_trace()
        # 3. 专家网络生成每个特征的权重（n, num_experts）
        expert_weights = []
        for i, expert in enumerate(self.experts):
            # 仅处理被选中的专家
            if gate_weights[i] > 0:
                # 提取第 i 个专家的特征（n, d）
                expert_feature = features[:, i, :]  # (n, d)
                # 生成权重（n, 1）并应用门控权重
                weight = expert(expert_feature).squeeze(-1) * gate_weights[i]  # (n,)
                expert_weights.append(weight)
            else:
                expert_weights.append(torch.zeros(n, device=features.device))
        expert_weights = torch.stack(expert_weights, dim=1)  # (n, num_experts)

        # 4. 归一化专家权重并加权求和
        expert_weights = F.softmax(expert_weights, dim=1)  # (n, num_experts)
        weighted_features = features * expert_weights.unsqueeze(-1)  # (n, num_experts, d)
        final_features = weighted_features.sum(dim=1)  # (n, d)

        return final_features

# 测试代码
if __name__ == "__main__":
    # 模拟输入数据
    n, m, d = 100, 50, 128  # n: 特征数量，m: 查询数量，d: 特征维度
    aux_features = [torch.randn(n, d) for _ in range(4)]  # 4个多尺度特征（n, d）
    pcd_features = torch.randn(n, d)  # 主特征（n, d）
    query = torch.randn(m, d)  # 查询向量（m, d）

    # 合并多尺度特征为 (n, 5, d)，每个点有 5 个尺度特征
    src_pcd = torch.stack(aux_features + [pcd_features], dim=1)  # (n, 5, d)

    # 初始化 MoE 加权模块
    moe_weighting = MoEWeighting(input_dim=d, num_experts=5, top_k=3)

    # 动态加权
    output_features = moe_weighting(query, src_pcd)

    # 输出结果
    print("Output Features Shape:", output_features.shape)  # 应为 (n, d)