import torch

from rri.checkpoint import load_checkpoint, make_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    model = torch.nn.Linear(3, 2)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    payload = make_checkpoint(
        model=model,
        epoch=7,
        config={"dataset": {"name": "test"}},
        metrics={"accuracy": 0.75},
    )
    path = save_checkpoint(payload, tmp_path / "nested" / "best.pt")
    with torch.no_grad():
        model.weight.zero_()
    checkpoint = load_checkpoint(path, model=model)
    assert checkpoint["epoch"] == 7
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])

