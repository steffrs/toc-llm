import json
from typing import Any
from datetime import datetime
import random
import secrets

import numpy as np
import torch


def load_json_data(filepath: str) -> Any:
    with open(filepath, encoding="utf-8") as inp:
        return json.load(inp)


def get_experiment_id(prefix: str | None = None) -> str:
    time_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    short_id = secrets.token_hex(3)
    full_id = f"{time_id}_{short_id}"
    if prefix:
        full_id = f"{prefix}_{full_id}"
    return full_id


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"- Model: total parameters: {total_params:,}", flush=True)
    print(f"- Model: trainable parameters: {trainable_params:,}", flush=True)
    print(f"- Model: max input tokens: {model.config.max_position_embeddings:,}\n", flush=True)


def get_linear_segmentation(segmentation: dict[int, list[int]], transform_to_change_labels: bool
                            ) -> tuple[list[int], int]:
    max_level = max(segmentation.keys())
    lin = segmentation[max_level]
    if transform_to_change_labels:
        return lin[1:], max_level  # remove label of first sentence
    return lin, max_level


def set_seed(seed: int):
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For CUDA
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    np.random.seed(seed)  # For numpy operations
    random.seed(seed)  # For random operations
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables performance optimizations for deterministic results
