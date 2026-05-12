"""Training entrypoint. Logs to MLflow, saves LoRA adapters to the model registry."""
from __future__ import annotations

import mlflow
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from platform.config import TrainConfig


def train(cfg: TrainConfig) -> None:
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.experiment)

    with mlflow.start_run():
        mlflow.log_params(cfg.as_dict())
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
        model = get_peft_model(
            model,
            LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules=cfg.target_modules),
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        # Training loop omitted — see platform/trainer.py
        adapter_dir = cfg.artifact_dir / "adapter"
        model.save_pretrained(adapter_dir)
        mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/adapter", cfg.model_name)
