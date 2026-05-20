from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.reset_parameters()

        for param in self.linear.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return result + lora * self.scaling


def apply_lora_to_modules(
    module: nn.Module,
    targets: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    target_set = set(targets)
    replaced = 0

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_set:
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            replaced += 1
            continue

        matched_targets = [
            target.removeprefix(f"{name}.")
            for target in target_set
            if target.startswith(f"{name}.")
        ]
        if matched_targets:
            replaced += apply_lora_to_modules(
                child,
                matched_targets,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

    return replaced


def is_lora_parameter(name: str) -> bool:
    return ".lora_A" in name or ".lora_B" in name
