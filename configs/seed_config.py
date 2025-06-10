# configs/seed_config.py
# =========================================================
# Default hyper-parameter configuration for the **SEED**
# framework (Structural Encoder for Embedding-Driven Decoding)
# ---------------------------------------------------------
# These values are chosen so that the reference
# implementations in `models/` run out-of-the-box on the
# common ETTh1 dataset with a 7-variable input.
#
# You can create a customised configuration either
# 1) by subclassing `SEEDConfig`,
# 2) by calling `SEEDConfig.from_args(argparse_args)`, or
# 3) by editing/over-writing any attribute after instantiation.
# =========================================================

from __future__ import annotations
from dataclasses import dataclass, asdict, field
import json
import argparse
from typing import Any, Dict


@dataclass
class SEEDConfig:
    # -----------------------------------------------------
    # —— 任务与数据 ——                                     
    # -----------------------------------------------------
    task_name: str = "long_term_forecast"     # 任务类型：long_term_forecast | imputation | ...
    data: str = "ETTh1"                       # 数据集名称
    device: str = "cuda"                      # 训练/推理设备
    
    # -----------------------------------------------------
    # —— 输入/输出长度 ——                                  
    # -----------------------------------------------------
    seq_len: int = 96                         # 编码器输入序列长度
    label_len: int = 48                       # 解码器强制输入长度
    pred_len: int = 96                        # 需要预测的步数
    
    # -----------------------------------------------------
    # —— 特征维度 ——                                      
    # -----------------------------------------------------
    enc_in: int = 7                           # 编码器输入变量数 (= n_vars)
    dec_in: int = 7                           # 解码器输入变量数
    c_out: int = 7                            # 输出变量数 (= 要预测的通道)
    
    # -----------------------------------------------------
    # —— StructuralEncoder (iTransformer) 参数 ————        
    # -----------------------------------------------------
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    e_layers: int = 4
    factor: int = 1
    dropout: float = 0.1
    activation: str = "gelu"
    embed: str = "timeF"                      # time features embedding
    freq: str = "h"                           # 时间粒度：h, t, s …
    output_attention: bool = False
    use_custom_itransformer: bool = False     # 若为 True 则使用自定义轻量实现
    extract_mode: str = "last_layer"          # last_layer | mean_pooling | ...
    
    # -----------------------------------------------------
    # —— FeatureConverter / Patch 参数 ————————           
    # -----------------------------------------------------
    patch_len: int = 16                       # 单个 patch 包含的步数
    stride: int = 8                           # 相邻 patch 滑动步长
    patch_dim: int = 768                      # patch 投影后的维度
    
    # -----------------------------------------------------
    # —— SemanticDecoder / LLM 参数 ————————              
    # -----------------------------------------------------
    llm_model: str = "meta-llama/Llama-2-7b-hf"
    llm_dim: int = 4096                       # Llama-2-7B 隐藏层宽度
    num_prototypes: int = 8                   # ReprogrammingLayer 原型 token 数
    
    # -----------------------------------------------------
    # —— RevIN (可逆归一化) ————————————————             
    # -----------------------------------------------------
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    
    # -----------------------------------------------------
    # —— PromptBuilder 参数 ——————————————               
    # -----------------------------------------------------
    top_k: int = 5                            # 自相关 lag 数，用于生成 prompt
    
    # -----------------------------------------------------
    # —— 训练超参 (可选) ——————————————                 
    # -----------------------------------------------------
    lr: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    patience: int = 5                         # Early-Stopping
    
    # -----------------------------------------------------
    # —— 其它杂项 —————————————————                       
    # -----------------------------------------------------
    seed: int = 42
    log_interval: int = 100                  # 每多少 step 打印一次 log
    wandb: bool = False                      # 开 / 关 Weights&Biases 记录
    
    # ============== Helper utilities =====================
    def to_dict(self) -> Dict[str, Any]:
        """Return an *ordered* python dict of all fields."""
        return asdict(self)
    
    def save_json(self, path: str) -> None:
        """Serialize current configuration to a JSON file."""
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_json(path: str) -> "SEEDConfig":
        """Instantiate a configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)
        return SEEDConfig(**cfg)
    
    # ---------------- Argparse bridge -------------------
    @staticmethod
    def build_argparser() -> argparse.ArgumentParser:
        """Return a parser pre-populated with all config fields."""
        parser = argparse.ArgumentParser("SEED configuration", add_help=False)
        for field_name, field_def in SEEDConfig.__dataclass_fields__.items():
            f_type = field_def.type
            default_val = field_def.default
            arg_type = f_type if f_type in (int, float, str, bool) else str
            # Boolean 特殊处理：--flag / --no-flag
            if arg_type is bool:
                group = parser.add_mutually_exclusive_group()
                group.add_argument(f"--{field_name}", dest=field_name, action="store_true")
                group.add_argument(f"--no-{field_name}", dest=field_name, action="store_false")
                parser.set_defaults(**{field_name: default_val})
            else:
                parser.add_argument(f"--{field_name}", type=arg_type, default=default_val)
        return parser
    
    @staticmethod
    def from_args(args: argparse.Namespace | list[str] | None = None) -> "SEEDConfig":
        """Create config from command-line arguments (or list)."""
        if not isinstance(args, argparse.Namespace):
            parser = SEEDConfig.build_argparser()
            args = parser.parse_args(args=args)
        # 过滤未知字段（保持向前兼容）
        valid_keys = SEEDConfig.__dataclass_fields__.keys()
        cfg_dict = {k: v for k, v in vars(args).items() if k in valid_keys}
        return SEEDConfig(**cfg_dict)


# When used as a script: quick CLI to dump a JSON template
if __name__ == "__main__":
    cfg = SEEDConfig()
    cfg.save_json("seed_config.default.json")
    print("Default configuration written to seed_config.default.json")
