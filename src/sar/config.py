from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    model_id: str = "Qwen/Qwen2.5-Omni-3B"
    quantization: Literal["nf4", "none"] = "nf4"
    thinker_only: bool = True
    device_map: str = "auto"
    attention_backend: str = "eager"


class DataConfig(StrictModel):
    manifest: str
    seed: int = 0


class DAAConfig(StrictModel):
    block_strategy: Literal["declared", "fixed"] = "declared"
    block_seconds: float = Field(default=1.5, gt=0)
    max_blocks: int = Field(default=8, ge=1)
    max_focus_blocks: int = Field(default=2, ge=1)
    scan_max_new_tokens: int = Field(default=192, ge=8)
    select_max_new_tokens: int = Field(default=48, ge=4)
    answer_max_new_tokens: int = Field(default=64, ge=4)


class MethodConfig(StrictModel):
    name: Literal[
        "base",
        "qacr",
        "daa",
        "static_gate",
        "layer_router",
        "head_router",
        "fixed_kv_subspace",
        "oracle_mask",
    ]
    layers: list[int] = Field(default_factory=lambda: [0])
    router_dim: int = 32
    lambda_switch: float = 1.0
    lambda_identity: float = 0.1
    checkpoint: str | None = None
    daa: DAAConfig | None = None


class TrainingConfig(StrictModel):
    epochs: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    grad_clip_norm: float = Field(default=1.0, gt=0)


class ExperimentConfig(StrictModel):
    model: ModelConfig
    data: DataConfig
    method: MethodConfig
    output_dir: str
    seed: int = 0
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ExperimentConfig.model_validate(data)
