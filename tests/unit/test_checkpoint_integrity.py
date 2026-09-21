import copy
import pytest
import torch
from factories import make_tiny_mtwam


@pytest.mark.parametrize(
    "corruption",
    [
        "partial_branch",
        "partial_trunk",
        "extra_key",
        "wrong_shape",
        "missing_proprio",
        "wrong_proprio",
        "legacy_video",
    ],
)
def test_invalid_checkpoint_is_rejected_before_any_weight_changes(tmp_path, corruption):
    torch.manual_seed(1)
    source = make_tiny_mtwam(enable_dynamic_branch=True, proprio_dim=8)
    path = tmp_path / "checkpoint.pt"
    source.save_checkpoint(path, step=1)
    payload = torch.load(path, weights_only=True)
    if corruption == "partial_branch":
        del payload["mot"]["dynamic_branch.role_t"]
    elif corruption == "partial_trunk":
        del payload["mot"][
            next((k for k in payload["mot"] if k.startswith("mixtures.video.")))
        ]
    elif corruption == "extra_key":
        payload["mot"]["unexpected.weight"] = torch.ones(1)
    elif corruption == "wrong_shape":
        payload["mot"]["dynamic_branch.role_t"] = torch.ones(2)
    elif corruption == "missing_proprio":
        del payload["proprio_encoder"]
    elif corruption == "wrong_proprio":
        payload["proprio_encoder"]["weight"] = torch.ones(1)
    elif corruption == "legacy_video":
        payload = {"dit": source.video_expert.state_dict()}
    torch.save(payload, path)
    torch.manual_seed(2)
    destination = make_tiny_mtwam(enable_dynamic_branch=True, proprio_dim=8)
    before = copy.deepcopy(destination.state_dict())
    with pytest.raises(ValueError):
        destination.load_checkpoint(path)
    after = destination.state_dict()
    assert before.keys() == after.keys()
    assert all((torch.equal(before[k], after[k]) for k in before))
