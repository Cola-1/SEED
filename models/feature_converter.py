"""
Feature Converter - iTransformer 到 Time-LLM 特征适配层
实现结构化特征的 Patch 划分、重组与维度对齐，作为 LLM 输入 embedding。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureConverter(nn.Module):
    """
    将结构编码器（iTransformer）输出转换为适合 LLM 输入的 patch 特征。
    支持滑窗划分 patch、线性投影至 LLM embedding 维度。
    """

    def __init__(self, d_model, patch_len, stride, patch_dim, n_vars):
        """
        Args:
            d_model: 结构编码器输出维度（如 512）
            patch_len: 每个 patch 包含的 token 数
            stride: patch 滑动步长
            patch_dim: patch 投影后的 embedding 维度（需与 LLM embedding 对齐, 如 768）
            n_vars: 变量数量（多变量场景下并行 patch）
        """
        super(FeatureConverter, self).__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.patch_dim = patch_dim
        self.n_vars = n_vars

        # 每个变量独立一个 patch 投影器（可选：共用）
        self.patch_projector = nn.Linear(d_model * patch_len, patch_dim)

    def forward(self, features: torch.Tensor, x_raw: torch.Tensor):
        """
        Args:
            features: [B, N, D]，iTransformer 编码器输出
            x_raw:    [B, T, N]，原始输入序列（用于可能的辅助特征）
        Returns:
            patches:     [B, N, P, patch_dim]，每个变量的 patch embedding
            patch_masks: [B, N, P]，padding 掩码（True 表示有效 patch，False 为补齐）
        """

        # 1. 特征转序列: [B, N, D] → [B, N, T, D]（若 T = seq_len）
        # 但实际 iTransformer 常只输出 [B, N, D]，需辅助构造 patch
        # 这里假设每个变量的序列特征已被 iTransformer 编码到 [B, N, D]
        # 如果 features 是 [B, T, N, D]，则需要调整

        # 为兼容性，这里使用 x_raw 的时间长度作为 patch 滑窗主维
        B, T, N = x_raw.shape
        assert N == self.n_vars, "n_vars mismatch between input and feature_converter config"

        # 假设 features 需重复/广播至时间维，或只用作全局 patch（简化版）
        # Step 1: 构造每个变量的 token 序列 [B, N, T, d_model]
        features_seq = features.unsqueeze(2).expand(-1, -1, T, -1)  # [B, N, T, d_model]

        # Step 2: 对每个变量并行滑窗分 patch
        patches = []
        patch_masks = []

        for var_idx in range(N):
            # 提取该变量的时间序列特征 [B, T, d_model]
            var_seq = features_seq[:, var_idx, :, :]  # [B, T, d_model]

            # 构造所有滑窗 patch
            var_patches = []
            var_masks = []

            for start in range(0, T - self.patch_len + 1, self.stride):
                end = start + self.patch_len
                patch = var_seq[:, start:end, :]  # [B, patch_len, d_model]
                # 展平成一维
                patch_flat = patch.reshape(patch.shape[0], -1)  # [B, patch_len * d_model]
                var_patches.append(patch_flat)
                var_masks.append(torch.ones(patch.shape[0], dtype=torch.bool, device=features.device))
            # 处理最后不足 patch_len 的情况（padding）
            last = T - ( ( (T - self.patch_len) // self.stride ) * self.stride + self.patch_len )
            if last > 0:
                # 取最后 patch_len 个
                patch = var_seq[:, -self.patch_len:, :]
                patch_flat = patch.reshape(patch.shape[0], -1)
                var_patches.append(patch_flat)
                var_masks.append(torch.zeros(patch.shape[0], dtype=torch.bool, device=features.device))  # 标记为 padding

            # 堆叠 [P, B, patch_len*d_model] → [B, P, patch_len*d_model]
            var_patches = torch.stack(var_patches, dim=1)
            var_masks = torch.stack(var_masks, dim=1)
            patches.append(var_patches)
            patch_masks.append(var_masks)

        # 汇总 [N, B, P, patch_len*d_model] → [B, N, P, patch_len*d_model]
        patches = torch.stack(patches, dim=1)  # [B, N, P, patch_len*d_model]
        patch_masks = torch.stack(patch_masks, dim=1)  # [B, N, P]

        # Step 3: Patch 投影到 LLM embedding 空间
        B, N, P, D_in = patches.shape
        patches = patches.view(B * N * P, -1)  # [B*N*P, patch_len*d_model]
        patches = self.patch_projector(patches)  # [B*N*P, patch_dim]
        patches = patches.view(B, N, P, self.patch_dim)  # [B, N, P, patch_dim]

        return patches, patch_masks

