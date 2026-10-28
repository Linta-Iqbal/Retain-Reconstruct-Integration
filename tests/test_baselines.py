import torch
import torch.nn.functional as F
from torch import nn

from rri.baselines import (
    fit_global_retention,
    oracle_fusion_weights,
    oracle_retention_weight,
)
from rri.metrics import accuracy


def test_learned_and_oracle_retention_prefer_rebuilt_feature():
    observed = torch.tensor([[3.0, 0.0]]).repeat(8, 1)
    rebuilt = torch.tensor([[0.0, 3.0]]).repeat(8, 1)
    targets = torch.ones(8, dtype=torch.long)

    def train_predict(mixed, indices):
        return mixed

    alpha = fit_global_retention(
        observed,
        rebuilt,
        targets,
        train_predict,
        F.cross_entropy,
        epochs=50,
        learning_rate=0.1,
        batch_size=8,
        normalize=False,
    )
    assert alpha < 0.5

    oracle = oracle_retention_weight(
        observed,
        rebuilt,
        targets,
        lambda mixed: mixed,
        lambda predictions, _: float(predictions[:, 1].mean()),
        alpha_grid=[0.0, 0.5, 1.0],
        normalize=False,
    )
    assert oracle.alpha == 0.0


class SimpleFusion(nn.Module):
    def project(self, first, second):
        return first, second

    def fuse_projected(self, first, second, first_weight=1.0, second_weight=1.0):
        score = first_weight * first[:, 0] + second_weight * second[:, 0]
        return torch.stack((-score, score), dim=-1)


def test_oracle_fusion_searches_independent_weights():
    fusion = SimpleFusion()
    first = -torch.ones(4, 1)
    second = torch.ones(4, 1)
    targets = torch.ones(4, dtype=torch.long)
    result = oracle_fusion_weights(
        fusion, first, second, targets, accuracy, weight_grid=[0.0, 1.0]
    )
    assert result.score == 1.0
    assert result.first_weight == 0.0
    assert result.second_weight == 1.0

