import torch


def solution(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)
