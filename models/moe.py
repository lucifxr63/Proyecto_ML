import torch
import torch.nn as nn
from typing import List

class MixtureOfExperts(nn.Module):
    """Mixture of Experts architecture."""
    def __init__(self, expert_cls, num_experts: int = 4, num_classes: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([expert_cls(num_classes) for _ in range(num_experts)])
        # Gating network takes the same input and outputs a distribution over experts
        self.gating = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_weights = self.gating(x)  # (batch_size, num_experts)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # expert_outputs shape: (batch, num_experts, num_classes)
        gate_weights = gate_weights.unsqueeze(-1)
        output = torch.sum(gate_weights * expert_outputs, dim=1)
        return output
