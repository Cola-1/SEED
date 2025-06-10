"""
Structural Encoder - iTransformer 封装
用于提取多变量时间序列的结构化特征
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Optional, Tuple, Dict

# 添加 iTransformer 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'iTransformer'))

# 导入 iTransformer 组件
from models.iTransformer import Model as iTransformer
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import layers.Transformer_EncDec as Transformer_EncDec


class StructuralEncoder(nn.Module):
    """
    结构编码器 - 封装 iTransformer
    
    主要功能：
    1. 使用 iTransformer 捕捉多变量之间的相关性
    2. 在变量维度上应用注意力机制
    3. 在时间维度上使用前馈网络学习序列表示
    """
    
    def __init__(self, configs):
        super(StructuralEncoder, self).__init__()
        
        # 基本配置
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len if hasattr(configs, 'label_len') else configs.seq_len // 2
        self.output_attention = configs.output_attention
        
        # 模型维度
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.embed = configs.embed if hasattr(configs, 'embed') else 'timeF'
        self.freq = configs.freq if hasattr(configs, 'freq') else 'h'
        self.dropout = configs.dropout
        
        # Transformer 配置
        self.factor = configs.factor if hasattr(configs, 'factor') else 1
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.e_layers = configs.e_layers
        self.activation = configs.activation if hasattr(configs, 'activation') else 'gelu'
        
        # 使用类别来选择使用原始 iTransformer 还是自定义版本
        self.use_custom = configs.use_custom_itransformer if hasattr(configs, 'use_custom_itransformer') else False
        
        if not self.use_custom:
            # 使用原始 iTransformer
            self._init_original_itransformer(configs)
        else:
            # 使用自定义实现（更灵活）
            self._init_custom_itransformer(configs)
            
        # 特征提取模式
        self.extract_mode = configs.extract_mode if hasattr(configs, 'extract_mode') else 'last_layer'
        
    def _init_original_itransformer(self, configs):
        """初始化原始 iTransformer"""
        # 创建配置对象
        itransformer_configs = type('Config', (), {})()
        
        # 复制所需配置
        for attr in ['seq_len', 'label_len', 'pred_len', 'enc_in', 'd_model', 
                     'embed', 'freq', 'dropout', 'factor', 'n_heads', 'd_ff', 
                     'e_layers', 'activation', 'output_attention', 'class_strategy']:
            if hasattr(configs, attr):
                setattr(itransformer_configs, attr, getattr(configs, attr))
            else:
                # 设置默认值
                if attr == 'class_strategy':
                    setattr(itransformer_configs, attr, 'projection')
                elif attr == 'label_len':
                    setattr(itransformer_configs, attr, configs.seq_len // 2)
                    
        # 添加任务名称
        itransformer_configs.task_name = 'long_term_forecast'
        itransformer_configs.d_layers = 1
        itransformer_configs.moving_avg = 25
        itransformer_configs.c_out = configs.enc_in
        itransformer_configs.use_norm = True
        
        # 初始化模型
        self.model = iTransformer(itransformer_configs)
        
    def _init_custom_itransformer(self, configs):
        """初始化自定义 iTransformer 实现"""
        # 嵌入层 - 反转的嵌入（变量作为 token）
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=configs.factor,
                            scale=None,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # 投影层（可选）
        self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        
    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
        enc_self_mask: Optional[torch.Tensor] = None,
        dec_self_mask: Optional[torch.Tensor] = None,
        dec_enc_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x_enc: 编码器输入 [B, T, N]
            x_mark_enc: 时间特征 [B, T, C]
            其他参数用于兼容性
            
        Returns:
            features: 结构化特征 [B, N, D]
            attns: 注意力权重
        """
        
        if not self.use_custom:
            # 使用原始 iTransformer
            return self._forward_original(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            # 使用自定义实现
            return self._forward_custom(x_enc, x_mark_enc)
            
    def _forward_original(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """使用原始 iTransformer 的前向传播"""
        # 准备解码器输入（如果没有提供）
        if x_dec is None:
            x_dec = torch.zeros(
                [x_enc.shape[0], self.pred_len, x_enc.shape[2]],
                device=x_enc.device
            )
            
        if x_mark_dec is None:
            x_mark_dec = torch.zeros(
                [x_enc.shape[0], self.pred_len, x_mark_enc.shape[2]],
                device=x_enc.device
            )
            
        # 调用原始模型
        if self.output_attention:
            output, attns = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            attns = None
            
        # 提取特征而不是最终预测
        features = self._extract_features(output, x_enc)
        
        return features, attns
        
    def _forward_custom(self, x_enc, x_mark_enc):
        """自定义 iTransformer 的前向传播"""
        # 反转维度：[B, T, N] -> [B, N, T]
        x_enc = x_enc.permute(0, 2, 1)
        B, N, T = x_enc.shape
        
        # 嵌入
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, N, D]
        
        # 编码
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # 根据提取模式返回特征
        if self.extract_mode == 'last_layer':
            features = enc_out  # [B, N, D]
        elif self.extract_mode == 'mean_pooling':
            features = enc_out.mean(dim=1, keepdim=True).expand(-1, N, -1)
        else:
            features = enc_out
            
        return features, attns
        
    def _extract_features(self, output, x_enc):
        """
        从 iTransformer 输出中提取特征
        
        Args:
            output: iTransformer 的输出 [B, pred_len, N]
            x_enc: 原始输入 [B, T, N]
            
        Returns:
            features: 提取的特征 [B, N, D]
        """
        B, T, N = x_enc.shape
        
        # 方案1：使用编码器的中间表示
        if hasattr(self.model, 'enc_out'):
            return self.model.enc_out  # [B, N, D]
            
        # 方案2：从输出重构特征
        # 这里简单地使用输出的统计信息作为特征
        features = []
        
        # 输出的均值和标准差
        output_mean = output.mean(dim=1)  # [B, N]
        output_std = output.std(dim=1)    # [B, N]
        
        # 输入的统计信息
        input_mean = x_enc.mean(dim=1)    # [B, N]
        input_std = x_enc.std(dim=1)      # [B, N]
        
        # 组合特征
        features = torch.stack([
            output_mean, output_std, input_mean, input_std
        ], dim=-1)  # [B, N, 4]
        
        # 投影到 d_model 维度
        if features.shape[-1] != self.d_model:
            projection = nn.Linear(features.shape[-1], self.d_model).to(features.device)
            features = projection(features)
            
        return features
        
    def get_encoder_output(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        """
        获取编码器的输出（用于 SEED 主模型）
        
        这是一个专门的接口，用于获取中间层表示
        """
        if self.use_custom:
            # 自定义实现直接返回编码器输出
            x_enc = x_enc.permute(0, 2, 1)  # [B, T, N] -> [B, N, T]
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, _ = self.encoder(enc_out)
            return enc_out  # [B, N, D]
        else:
            # 对于原始 iTransformer，需要特殊处理
            # 临时修改模型以获取中间输出
            original_forward = self.model.forward
            encoder_output = None
            
            def modified_forward(x_enc, x_mark_enc, x_dec, x_mark_dec):
                nonlocal encoder_output
                # 执行编码
                x_enc = self.model.enc_embedding(x_enc, x_mark_enc)
                encoder_output, _ = self.model.encoder(x_enc)
                # 继续原始的前向传播
                return original_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
            # 临时替换前向函数
            self.model.forward = modified_forward
            
            # 执行前向传播
            _ = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # 恢复原始前向函数
            self.model.forward = original_forward
            
            return encoder_output
            
    def load_pretrained(self, checkpoint_path):
        """加载预训练的 iTransformer 权重"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # 处理键名不匹配的情况
        if not self.use_custom:
            # 对于原始 iTransformer
            self.model.load_state_dict(state_dict, strict=False)
        else:
            # 对于自定义实现，需要映射键名
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    k = k[6:]  # 移除 'model.' 前缀
                new_state_dict[k] = v
                
            self.load_state_dict(new_state_dict, strict=False)
            
        print(f"Loaded pretrained weights from {checkpoint_path}")
        
    def freeze(self):
        """冻结编码器参数"""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """解冻编码器参数"""
        for param in self.parameters():
            param.requires_grad = True