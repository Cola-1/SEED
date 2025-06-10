# configs/timellm_config.py
# =========================================================
# Default hyper-parameter configuration for **Time-LLM**
# ---------------------------------------------------------
# This file contains only the parameters that are relevant
# when you want to train / finetune the Time-LLM side
# (ReprogrammingLayer + Frozen/Un-Frozen LLM) in isolation,
# e.g. for ablation studies or pre-training the reprogram-
# ming prototypes on a large corpus of time-series patches.
#
# The SEEDConfig already re-exports a subset of these fields
# for end-to-end experiments; however, keeping a dedicated
# config makes the *Time-LLM* component reusable and easier
# to benchmark independently.
# =========================================================

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict
import argparse
import json


@dataclass
class TimeLLMConfig:
    # -----------------------------------------------------
    # —— 任务与基础信息 ——                                 
    # -----------------------------------------------------
    task_name: str = "patch_reprogramming"      # 典型任务：patch_reprogramming | prompt_tuning
    data: str = "ETTh1"                         # 对应的数据集（决定 prompt 构建逻辑）
    device: str = "cuda"                        # cuda | cpu
    
    # -----------------------------------------------------
    # —— Patch / Prototype 超参 ——                         
    # -----------------------------------------------------
    patch_dim: int = 768                        # FeatureConverter 投影后的维度
    num_prototypes: int = 8                     # ReprogrammingLayer 中的原型 token 数
    max_patches: int = 32                       # 单个变量可接受的最大 patch 数
    prompt_max_len: int = 128                   # 附加文本 prompt 的最大长度
    
    # -----------------------------------------------------
    # —— LLM 相关 ——                                      
    # -----------------------------------------------------
    llm_model: str = "meta-llama/Llama-2-7b-hf" # HuggingFace repo or path
    llm_dim: int = 4096                         # 隐藏层宽度（自动从模型推断亦可）
    freeze_llm: bool = True                     # 默认冻结，训练仅更新 ReprogrammingLayer
    
    # -----------------------------------------------------
    # —— 训练超参 ——                                      
    # -----------------------------------------------------
    lr: float = 1e-4
    batch_size: int = 8                         # 8 GPUs × 8 = 64 effective batch
    num_epochs: int = 5
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    patience: int = 3                           # Early-stopping
    
    # -----------------------------------------------------
    # —— 其它杂项 ——                                      
    # -----------------------------------------------------
    seed: int = 42
    log_interval: int = 50                      # 每多少 step 打印一次日志
    wandb: bool = False                         # 是否启用 Weights & Biases
    
    # =====================================================
    # Helper utilities
    # =====================================================
    def to_dict(self) -> Dict[str, Any]:
        """Return an ordered python dict of all fields."""
        return asdict(self)
    
    def save_json(self, path: str) -> None:
        """Serialize current configuration to a JSON file."""
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_json(path: str) -> "TimeLLMConfig":
        """Instantiate configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)
        return TimeLLMConfig(**cfg)
    
    # ---------------- Argparse bridge -------------------
    @staticmethod
    def build_argparser() -> argparse.ArgumentParser:
        """Return a parser pre-populated with all config fields."""
        parser = argparse.ArgumentParser("Time-LLM configuration",
                                         add_help=False)
        
        for name, field_def in TimeLLMConfig.__dataclass_fields__.items():
            f_type = field_def.type
            default_val = field_def.default
            arg_type = f_type if f_type in (int, float, str, bool) else str
            
            # Boolean: generate --flag / --no-flag
            if arg_type is bool:
                group = parser.add_mutually_exclusive_group()
                group.add_argument(f"--{name}", dest=name, action="store_true")
                group.add_argument(f"--no-{name}", dest=name, action="store_false")
                parser.set_defaults(**{name: default_val})
            else:
                parser.add_argument(f"--{name}", type=arg_type, default=default_val)
        return parser
    
    @staticmethod
    def from_args(args: argparse.Namespace | list[str] | None = None
                  ) -> "TimeLLMConfig":
        """Create config from CLI arguments (or list of strings)."""
        if not isinstance(args, argparse.Namespace):
            parser = TimeLLMConfig.build_argparser()
            args = parser.parse_args(args=args)
        
        valid_keys = TimeLLMConfig.__dataclass_fields__.keys()
        cfg_dict = {k: v for k, v in vars(args).items() if k in valid_keys}
        return TimeLLMConfig(**cfg_dict)


# Quick utility: dump a JSON template when executed directly
if __name__ == "__main__":
    cfg = TimeLLMConfig()
    cfg.save_json("timellm_config.default.json")
    print("Default configuration written to timellm_config.default.json")
