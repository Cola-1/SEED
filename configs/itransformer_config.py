# configs/itransformer_config.py
# =========================================================
# Default hyper-parameter configuration for **iTransformer**
# ---------------------------------------------------------
# 仅关注结构编码器本身所需的参数；若要在 SEED 之外独立微调
# iTransformer，可直接修改本文件或通过 CLI/JSON 覆盖。
# =========================================================

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict
import argparse
import json


@dataclass
class ITransformerConfig:
    # -----------------------------------------------------
    # —— 任务与数据 ——                                     
    # -----------------------------------------------------
    task_name: str = "long_term_forecast"
    data: str = "ETTh1"
    device: str = "cuda"                       # cuda | cpu
    
    # -----------------------------------------------------
    # —— 序列长度 ——                                      
    # -----------------------------------------------------
    seq_len: int = 96                          # 编码器输入长度
    label_len: int = 48                        # 解码器强制输入
    pred_len: int = 96                         # 预测步长
    
    # -----------------------------------------------------
    # —— 特征维度 ——                                      
    # -----------------------------------------------------
    enc_in: int = 7                            # 输入变量数 (= n_vars)
    c_out: int = 7                             # 输出变量数
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    e_layers: int = 4                          # Encoder 层数
    d_layers: int = 1                          # Decoder 层数（iTransformer 原论文中为 1）
    factor: int = 1                            # ProbSparse 因子
    
    # -----------------------------------------------------
    # —— 嵌入/正则 ——                                    
    # -----------------------------------------------------
    dropout: float = 0.1
    embed: str = "timeF"                       # value | timeF
    freq: str = "h"                            # h, t, s, m, d, w, M
    activation: str = "gelu"
    output_attention: bool = False
    moving_avg: int = 25                       # Series Decomposition，Informers 兼容
    use_norm: bool = True
    class_strategy: str = "projection"         # projection | cls_token
    
    # -----------------------------------------------------
    # —— 训练超参 ——                                      
    # -----------------------------------------------------
    lr: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    weight_decay: float = 1e-2
    patience: int = 5
    seed: int = 42
    log_interval: int = 100
    
    # ============== Helper utilities =====================
    def to_dict(self) -> Dict[str, Any]:
        """Return an ordered python dict of all fields."""
        return asdict(self)
    
    def save_json(self, path: str) -> None:
        """Serialize current configuration to a JSON file."""
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_json(path: str) -> "ITransformerConfig":
        """Instantiate a configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)
        return ITransformerConfig(**cfg)
    
    # ---------------- Argparse bridge -------------------
    @staticmethod
    def build_argparser() -> argparse.ArgumentParser:
        """Return a parser pre-populated with all config fields."""
        parser = argparse.ArgumentParser("iTransformer configuration",
                                         add_help=False)
        for name, field_def in ITransformerConfig.__dataclass_fields__.items():
            f_type = field_def.type
            default_val = field_def.default
            arg_type = f_type if f_type in (int, float, str, bool) else str
            
            # Boolean: --flag / --no-flag
            if arg_type is bool:
                group = parser.add_mutually_exclusive_group()
                group.add_argument(f"--{name}", dest=name, action="store_true")
                group.add_argument(f"--no-{name}", dest=name,
                                   action="store_false")
                parser.set_defaults(**{name: default_val})
            else:
                parser.add_argument(f"--{name}", type=arg_type,
                                    default=default_val)
        return parser
    
    @staticmethod
    def from_args(args: argparse.Namespace | list[str] | None = None
                  ) -> "ITransformerConfig":
        """Create a config from CLI args (or list of strings)."""
        if not isinstance(args, argparse.Namespace):
            parser = ITransformerConfig.build_argparser()
            args = parser.parse_args(args=args)
        
        valid = ITransformerConfig.__dataclass_fields__.keys()
        cfg_dict = {k: v for k, v in vars(args).items() if k in valid}
        return ITransformerConfig(**cfg_dict)


# Quick utility: dump a JSON template when executed directly
if __name__ == "__main__":
    cfg = ITransformerConfig()
    cfg.save_json("itransformer_config.default.json")
    print("Default configuration written to itransformer_config.default.json")
