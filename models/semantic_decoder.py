"""
Semantic Decoder - Time-LLM 封装
实现 ReprogrammingLayer 与冻结 LLM 推理，将 Patch 特征重编码为语义向量。
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# 将 Time-LLM 子模块路径加入搜索
sys.path.append(os.path.join(os.path.dirname(__file__),
                             '..', 'third_party', 'Time-LLM'))
from models.TimeLLM import ReprogrammingLayer  # Time-LLM 重编程层 :contentReference[oaicite:0]{index=0}

class SemanticDecoder(nn.Module):
    """
    语义解码器 - 负责：
      1) 用 ReprogrammingLayer 将数值 Patch 映射到原型 token embedding
      2) 将这些原型 embedding 与文本 prompt 拼接，输入冻结的大语言模型
      3) 提取 LLM 的前几个 Hidden State，平均后作为每个变量的语义表示
    """

    def __init__(self, configs):
        super(SemanticDecoder, self).__init__()
        # 配置参数
        self.llm_model_name = configs.llm_model      # HF 模型名称
        self.llm_dim        = configs.llm_dim        # LLM 隐藏维度
        self.num_prototypes= getattr(configs, 'num_prototypes', 8)  # 原型数量，默认 8

        # 1. 初始化 Time-LLM 的重编程层
        self.reprogram = ReprogrammingLayer(
            in_dim = configs.patch_dim,             # Patch 特征维度
            out_dim = self.llm_dim,                 # 输出对齐到 LLM 隐藏维度
            num_prototypes = self.num_prototypes    # 原型 token 数量
        )

        # 2. 加载 tokenizer 和 LLM（冻结）
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            output_hidden_states=True              # 需要 Hidden States 以便提取前几个 token
        )
        self.freeze_llm()

    def freeze_llm(self):
        """冻结 LLM 参数，只训练重编程层等其他组件"""
        for p in self.llm.parameters():
            p.requires_grad = False

    def unfreeze_llm(self):
        """解冻 LLM 参数，允许微调"""
        for p in self.llm.parameters():
            p.requires_grad = True

    def load_pretrained(self, checkpoint_path: str):
        """加载 ReprogrammingLayer 的预训练权重"""
        state = torch.load(checkpoint_path, map_location='cpu')
        self.reprogram.load_state_dict(state, strict=False)

    def forward(self,
                patches: torch.Tensor,
                prompt_text: list,
                patch_masks: torch.Tensor
               ) -> torch.Tensor:
        """
        前向传播

        Args:
            patches:      [B, N, P, patch_dim] —— FeatureConverter 输出的 Patch Embeddings
            prompt_text:  List[str] of length B —— PromptBuilder 生成的文本提示
            patch_masks:  [B, N, P] —— Patch 有效性掩码

        Returns:
            semantic_output: [B, N, llm_dim] —— 每个变量的语义向量
        """
        B, N, P, D = patches.size()
        device = patches.device

        # 1) 重编程层：将 Patch → 原型 token embedding
        patches_flat = patches.view(B * N, P, D)            # [B*N, P, D]
        masks_flat   = patch_masks.view(B * N, P)           # [B*N, P]
        proto_emb    = self.reprogram(patches_flat, masks_flat)
        # proto_emb: [B*N, num_prototypes, llm_dim]

        # 2) 构建与 Patch 对应的 prompt 文本列表
        prompt_flat = []
        for b in range(B):
            for _ in range(N):
                prompt_flat.append(prompt_text[b])

        # 3) 分词 & 获取文本 embedding
        encoding = self.tokenizer(
            prompt_flat,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        input_ids      = encoding['input_ids'].to(device)         # [B*N, L]
        text_mask      = encoding['attention_mask'].to(device)    # [B*N, L]
        text_emb       = self.llm.get_input_embeddings()(input_ids)
        # text_emb: [B*N, L, llm_dim]

        # 4) 拼接原型 embedding 与文本 embedding
        proto_mask     = torch.ones(B * N, self.num_prototypes, device=device)
        llm_input_emb  = torch.cat([proto_emb, text_emb], dim=1)      # [B*N, num_prototypes+L, llm_dim]
        llm_attention_mask = torch.cat([proto_mask, text_mask], dim=1) # [B*N, num_prototypes+L]

        # 5) 冻结 LLM 推理
        outputs = self.llm(
            inputs_embeds = llm_input_emb,
            attention_mask= llm_attention_mask
        )
        # 取最后一层 Hidden States
        hidden_states = outputs.hidden_states[-1]  # [B*N, S, llm_dim]

        # 6) 提取原型部分的语义 token 表示并平均
        semantic_tokens = hidden_states[:, :self.num_prototypes, :]  # [B*N, num_prototypes, llm_dim]
        semantic_repr   = semantic_tokens.mean(dim=1)               # [B*N, llm_dim]

        # 7) 还原回 [B, N, llm_dim]
        semantic_output = semantic_repr.view(B, N, self.llm_dim)
        return semantic_output
