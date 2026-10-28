import torch
from pathlib import Path

from rri.config import load_config
from rri.models import build_system


def test_bimodal_classification_shapes_and_gate_range():
    system = build_system(
        ["image", "text"],
        {"image": 12, "text": 8},
        task_type="classification",
        output_dim=4,
        fusion_hidden_dim=16,
        reconstructor_hidden_dim=20,
        gate_hidden_dim=12,
    )
    features = {"image": torch.randn(5, 12), "text": torch.randn(5, 8)}
    assert system.predict(features).shape == (5, 4)
    predictions, alpha = system.forward_corrupted(features, "image", torch.randn(5, 12))
    assert predictions.shape == (5, 4)
    assert alpha.shape == (5, 1)
    assert torch.all((alpha >= 0) & (alpha <= 1))


def test_trimodal_regression_shapes():
    system = build_system(
        ["visual", "audio", "text"],
        {"visual": 10, "audio": 6, "text": 7},
        task_type="regression",
        output_dim=1,
        fusion_hidden_dim=18,
        reconstructor_hidden_dim=20,
        gate_hidden_dim=12,
    )
    features = {
        "visual": torch.randn(4, 10),
        "audio": torch.randn(4, 6),
        "text": torch.randn(4, 7),
    }
    assert system.predict(features).shape == (4,)
    predictions, alpha = system.forward_corrupted(features, "audio", torch.randn(4, 6))
    assert predictions.shape == (4,)
    assert alpha.shape == (4, 1)


def test_compact_dataset_configs_match_parameter_budgets():
    project_root = Path(__file__).resolve().parents[1]
    expected = {
        "mvsa.yaml": 1_036_553,
        "food101.yaml": 1_074_479,
        "nyud.yaml": 1_039_262,
    }
    for filename, target_count in expected.items():
        config = load_config(project_root / "configs" / filename)
        modalities = config["dataset"]["modalities"]
        system = build_system(
            modalities,
            {name: 512 for name in modalities},
            task_type="classification",
            output_dim=int(config["dataset"]["num_classes"]),
            fusion_hidden_dim=int(config["model"]["fusion_hidden_dim"]),
            reconstructor_hidden_dim=int(config["model"]["reconstructor_hidden_dim"]),
            gate_hidden_dim=int(config["model"]["gate_hidden_dim"]),
            dropout=float(config["model"]["dropout"]),
            normalize_representations=bool(
                config["model"]["normalize_representations"]
            ),
        )
        trainable = sum(
            parameter.numel()
            for parameter in system.parameters()
            if parameter.requires_grad
        )
        assert trainable == target_count

