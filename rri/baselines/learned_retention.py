"""Learn one global observed-versus-reconstructed weight on training data."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

from rri.models.system import mix_features


def _freeze(modules: Iterable[torch.nn.Module]):
    states: list[tuple[torch.nn.Parameter, bool]] = []
    for module in modules:
        module.eval()
        for parameter in module.parameters():
            states.append((parameter, parameter.requires_grad))
            parameter.requires_grad_(False)
    return states


def _restore(states: list[tuple[torch.nn.Parameter, bool]]) -> None:
    for parameter, required in states:
        parameter.requires_grad_(required)


def fit_global_retention(
    observed: torch.Tensor,
    rebuilt: torch.Tensor,
    targets: torch.Tensor,
    predict_from_mixed: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    modules_to_freeze: Iterable[torch.nn.Module] = (),
    learning_rate: float = 1e-2,
    epochs: int = 100,
    batch_size: int = 64,
    normalize: bool = True,
) -> float:
    """Fit a scalar alpha using training labels while keeping the system frozen."""
    if observed.shape != rebuilt.shape or len(observed) != len(targets):
        raise ValueError("Observed, rebuilt, and targets must describe the same samples")
    states = _freeze(modules_to_freeze)
    raw_alpha = torch.nn.Parameter(torch.zeros((), device=observed.device))
    optimizer = torch.optim.Adam([raw_alpha], lr=learning_rate)
    try:
        for _ in range(epochs):
            order = torch.randperm(len(targets), device=targets.device)
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                optimizer.zero_grad()
                alpha = torch.sigmoid(raw_alpha)
                mixed = mix_features(
                    observed[indices], rebuilt[indices], alpha, normalize=normalize
                )
                loss = loss_fn(predict_from_mixed(mixed, indices), targets[indices])
                loss.backward()
                optimizer.step()
        return float(torch.sigmoid(raw_alpha).detach().cpu())
    finally:
        _restore(states)


@torch.no_grad()
def retention_predictions(
    observed: torch.Tensor,
    rebuilt: torch.Tensor,
    alpha: float,
    predict_from_mixed: Callable[[torch.Tensor], torch.Tensor],
    *,
    normalize: bool = True,
) -> torch.Tensor:
    mixed = mix_features(observed, rebuilt, alpha, normalize=normalize)
    return predict_from_mixed(mixed)

