"""
SEED: Structural Encoder for Embedding-Driven Decoding
主模型文件，整合 iTransformer 和 Time-LLM 的核心功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np

from .structural_encoder import StructuralEncoder
from .feature_converter import FeatureConverter
from .semantic_decoder import SemanticDecoder
from .output_projector import OutputProjector
from data_provider.prompt_builder import PromptBuilder
from utils.tools import StandardScaler


class SEED(nn.Module):
    """
    SEED 模型主类
    
    该模型通过以下步骤进行时间序列预测：
    1. 使用 iTransformer 提取结构化特征
    2. 将特征转换为适合 LLM 的 patch 格式
    3. 通过 Time-LLM 的重编程机制进行语义增强
    4. 使用冻结的 LLM 进行推理
    5. 投影输出得到最终预测
    """
    
    def __init__(self, configs):
        super(SEED, self).__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # 输入输出维度
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        
        # 模型维度
        self.d_model = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        
        # 初始化各个组件
        self._init_components(configs)
        
        # 数据标准化
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(
                num_features=self.enc_in, 
                affine=configs.affine, 
                subtract_last=configs.subtract_last
            )
            
        # 初始化参数
        self._init_weights()
        
    def _init_components(self, configs):
        """初始化 SEED 的各个组件"""
        
        # 1. 结构编码器 (iTransformer)
        self.structural_encoder = StructuralEncoder(configs)
        
        # 2. 特征转换器
        self.feature_converter = FeatureConverter(
            d_model=configs.d_model,
            patch_len=configs.patch_len,
            stride=configs.stride,
            patch_dim=configs.patch_dim,
            n_vars=configs.enc_in
        )
        
        # 3. 语义解码器 (Time-LLM components)
        self.semantic_decoder = SemanticDecoder(configs)
        
        # 4. 提示词构建器
        self.prompt_builder = PromptBuilder(
            dataset_name=configs.data,
            task_type=configs.task_name,
            pred_len=configs.pred_len,
            top_k=configs.top_k
        )
        
        # 5. 输出投影器
        self.output_projector = OutputProjector(
            llm_dim=configs.llm_dim,
            pred_len=configs.pred_len,
            n_vars=configs.c_out,
            d_model=configs.d_model
        )
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        enc_self_mask: Optional[torch.Tensor] = None,
        dec_self_mask: Optional[torch.Tensor] = None,
        dec_enc_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x_enc: 编码器输入 [B, T, N]
            x_mark_enc: 编码器时间特征 [B, T, C]
            x_dec: 解码器输入 [B, T', N]
            x_mark_dec: 解码器时间特征 [B, T', C]
            enc_self_mask: 编码器自注意力掩码
            dec_self_mask: 解码器自注意力掩码
            dec_enc_mask: 解码器-编码器注意力掩码
            
        Returns:
            output: 预测结果 [B, L, N]
        """
        
        # 数据标准化
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
            
        # Step 1: 结构特征提取 (iTransformer)
        # 获取多变量之间的结构化表示
        structural_features, attns = self.structural_encoder(
            x_enc, x_mark_enc, x_dec, x_mark_dec,
            enc_self_mask, dec_self_mask, dec_enc_mask
        )
        # structural_features: [B, N, D]
        
        # Step 2: 特征到 Patch 的转换
        # 将 iTransformer 的输出转换为适合 LLM 的格式
        patches, patch_masks = self.feature_converter(
            structural_features, x_enc
        )
        # patches: [B, N, P, patch_dim]
        
        # Step 3: 计算输入统计信息
        stats = self._compute_statistics(x_enc)
        
        # Step 4: 构建提示词
        prompt_text = self.prompt_builder.build_prompt(x_enc, stats)
        
        # Step 5: 语义解码
        # 使用 Time-LLM 的重编程机制和 LLM 推理
        semantic_output = self.semantic_decoder(
            patches=patches,
            prompt_text=prompt_text,
            patch_masks=patch_masks
        )
        # semantic_output: [B, N, llm_dim]
        
        # Step 6: 输出投影
        # 将 LLM 的输出投影到预测空间
        output = self.output_projector(
            semantic_output, 
            structural_features
        )
        # output: [B, pred_len, N]
        
        # 反标准化
        if self.revin:
            output = self.revin_layer(output, 'denorm')
            
        # 处理输出格式
        if self.output_attention:
            return output, attns
        else:
            return output
            
    def _compute_statistics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算输入序列的统计信息
        
        Args:
            x: 输入序列 [B, T, N]
            
        Returns:
            stats: 包含各种统计信息的字典
        """
        with torch.no_grad():
            stats = {
                'min': x.min(dim=1)[0],  # [B, N]
                'max': x.max(dim=1)[0],  # [B, N]
                'mean': x.mean(dim=1),   # [B, N]
                'std': x.std(dim=1),     # [B, N]
                'median': x.median(dim=1)[0],  # [B, N]
            }
            
            # 计算趋势（简单的线性趋势）
            time_steps = torch.arange(x.shape[1], device=x.device).float()
            time_steps = time_steps - time_steps.mean()
            
            # 对每个批次和变量计算趋势
            trends = []
            for b in range(x.shape[0]):
                batch_trends = []
                for n in range(x.shape[2]):
                    series = x[b, :, n]
                    trend = torch.sum(series * time_steps) / torch.sum(time_steps ** 2)
                    batch_trends.append(trend)
                trends.append(torch.stack(batch_trends))
            stats['trend'] = torch.stack(trends)  # [B, N]
            
            # 计算自相关的 top-k lags
            stats['top_k_lags'] = self._compute_top_k_lags(x, k=5)
            
        return stats
        
    def _compute_top_k_lags(self, x: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        计算自相关最高的 k 个滞后值
        
        Args:
            x: 输入序列 [B, T, N]
            k: 返回的滞后数量
            
        Returns:
            top_k_lags: [B, N, k]
        """
        B, T, N = x.shape
        top_k_lags = []
        
        with torch.no_grad():
            for b in range(B):
                batch_lags = []
                for n in range(N):
                    series = x[b, :, n]
                    # 计算自相关
                    autocorr = torch.zeros(T // 2)
                    for lag in range(1, T // 2):
                        if lag < T:
                            corr = torch.corrcoef(
                                torch.stack([series[:-lag], series[lag:]])
                            )[0, 1]
                            autocorr[lag] = corr if not torch.isnan(corr) else 0
                    
                    # 获取 top-k
                    _, indices = torch.topk(autocorr, k)
                    batch_lags.append(indices)
                    
                top_k_lags.append(torch.stack(batch_lags))
                
        return torch.stack(top_k_lags)
        
    def freeze_llm(self):
        """冻结 LLM 参数，只训练其他组件"""
        self.semantic_decoder.freeze_llm()
        
    def unfreeze_llm(self):
        """解冻 LLM 参数，允许微调"""
        self.semantic_decoder.unfreeze_llm()
        
    def get_trainable_parameters(self):
        """获取可训练参数列表"""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
        return trainable_params
        
    def load_pretrained_components(self, itransformer_path=None, timellm_path=None):
        """加载预训练的组件权重"""
        if itransformer_path:
            self.structural_encoder.load_pretrained(itransformer_path)
            print(f"Loaded pretrained iTransformer from {itransformer_path}")
            
        if timellm_path:
            self.semantic_decoder.load_pretrained(timellm_path)
            print(f"Loaded pretrained Time-LLM components from {timellm_path}")


class RevIN(nn.Module):
    """
    Reversible Instance Normalization
    用于处理时间序列的分布偏移问题
    """
    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
            
    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return x
        
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.mean = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True)
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True) + self.eps
        )
        
    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x
        
    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x