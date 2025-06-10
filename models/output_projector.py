"""
Output Projector
================
把 **SemanticDecoder** 生成的语义向量 & **StructuralEncoder** 生成的结构向量
融合后投影为最终的多步预测序列。

输入
-----
semantic_feats : torch.Tensor
    形状 [B, N, llm_dim] —— LLM 语义表示  
structural_feats : torch.Tensor
    形状 [B, N, d_model] —— iTransformer 结构表示  

输出
-----
pred : torch.Tensor
    形状 [B, pred_len, N] —— 目标变量未来 pred_len 步的点预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputProjector(nn.Module):
    """
    OutputProjector 采用 **特征拼接 + 前馈网络** 的轻量方式完成
    - 语义/结构信息融合
    - 多步预测映射
    默认在变量维度 **共享权重**，如需每个变量单独权重，可将 `share_weights=False`。
    """

    def __init__(
        self,
        llm_dim: int,
        pred_len: int,
        n_vars: int,
        d_model: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        share_weights: bool = True,
    ):
        super().__init__()

        self.llm_dim = llm_dim
        self.d_model = d_model
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.hidden_dim = hidden_dim or (llm_dim + d_model)  # 默认不降维
        self.share_weights = share_weights

        in_dim = llm_dim + d_model

        def _proj_block():
            return nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, pred_len),
            )

        # 共享或独立权重
        if share_weights:
            self.proj = _proj_block()
        else:
            self.proj = nn.ModuleList([_proj_block() for _ in range(n_vars)])

        # 轻量层归一化，提高稳定性
        self.norm_sem = nn.LayerNorm(llm_dim)
        self.norm_str = nn.LayerNorm(d_model)

    def forward(
        self,
        semantic_feats: torch.Tensor,   # [B, N, llm_dim]
        structural_feats: torch.Tensor  # [B, N, d_model]
    ) -> torch.Tensor:
        B, N, _ = semantic_feats.shape

        # 1. 归一化 & 拼接
        sem = self.norm_sem(semantic_feats)
        struc = self.norm_str(structural_feats)
        fused = torch.cat([sem, struc], dim=-1)           # [B, N, llm_dim+d_model]

        # 2. 前馈映射到 pred_len
        if self.share_weights:
            out = self.proj(fused)                        # [B, N, pred_len]
        else:
            outs = []
            for n in range(self.n_vars):
                outs.append(self.proj[n](fused[:, n, :]))  # [(B, pred_len), ...]
            out = torch.stack(outs, dim=1)                # [B, N, pred_len]

        # 3. 形状调整 -> [B, pred_len, N]
        out = out.permute(0, 2, 1).contiguous()
        return out
