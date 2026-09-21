import torch
from types import SimpleNamespace
from factories import make_tiny_mot, build_tiny_payload, assemble_mot_kwargs
from mtwam.models.wan22.mtwam import MTWAM


def _mask(mot, meta, traj_seq_len=0):
    stub = SimpleNamespace(video_expert=mot.mixtures["video"])
    return MTWAM._build_mot_attention_mask(
        stub,
        video_seq_len=meta["Sv"],
        action_seq_len=meta["Sa"],
        video_tokens_per_frame=meta["tpf"],
        device=torch.device("cpu"),
        traj_seq_len=traj_seq_len,
    )


def test_harness_drives_existing_two_tower_forward():
    mot = make_tiny_mot(enable_dynamic_branch=False)
    mot.eval()
    vpre, apre, meta = build_tiny_payload(mot)
    assert (meta["Sv"], meta["Sa"], meta["tpf"]) == (4, 3, 2)
    mask = _mask(mot, meta, traj_seq_len=0)
    out = mot.forward(**assemble_mot_kwargs(mot, vpre, apre, mask))
    assert out["video"].shape == vpre["tokens"].shape
    assert out["action"].shape == apre["tokens"].shape
    assert torch.isfinite(out["video"]).all() and torch.isfinite(out["action"]).all()


def test_trunk_video_bit_identical_branch_off_vs_on():
    mot = make_tiny_mot(enable_dynamic_branch=True, dynamic_branch_num_layers=1)
    mot.eval()
    vpre, apre, meta = build_tiny_payload(mot)
    St = 2 * meta["tpf"]
    out_off = mot.forward(**assemble_mot_kwargs(mot, vpre, apre, _mask(mot, meta, 0)))
    out_on = mot.forward(
        **assemble_mot_kwargs(mot, vpre, apre, _mask(mot, meta, St)),
        dynamic_branch_payload={"enable": True, "first_frame_tokens": meta["tpf"]},
    )
    assert torch.equal(out_off["video"], out_on["video"])
    assert out_on["action"].shape == out_off["action"].shape


def test_branch_injection_reinforces_action():
    mot = make_tiny_mot(enable_dynamic_branch=True, dynamic_branch_num_layers=1)
    mot.eval()
    vpre, apre, meta = build_tiny_payload(mot)
    St = 2 * meta["tpf"]
    out_off = mot.forward(**assemble_mot_kwargs(mot, vpre, apre, _mask(mot, meta, 0)))
    out_on = mot.forward(
        **assemble_mot_kwargs(mot, vpre, apre, _mask(mot, meta, St)),
        dynamic_branch_payload={"enable": True, "first_frame_tokens": meta["tpf"]},
    )
    assert not torch.equal(out_off["action"], out_on["action"])


def test_branch_read_text_toggles_branch_hidden():
    mot = make_tiny_mot(enable_dynamic_branch=True, dynamic_branch_num_layers=1)
    mot.eval()
    vpre, apre, meta = build_tiny_payload(mot)
    St = 2 * meta["tpf"]
    kwargs = assemble_mot_kwargs(mot, vpre, apre, _mask(mot, meta, St))
    payload = {"enable": True, "first_frame_tokens": meta["tpf"]}
    mot.dynamic_branch.read_text = False
    mot.forward(**kwargs, dynamic_branch_payload=payload)
    traj_off = mot._last_branch_hidden["traj"].clone()
    mot.dynamic_branch.read_text = True
    mot.forward(**kwargs, dynamic_branch_payload=payload)
    traj_on = mot._last_branch_hidden["traj"].clone()
    assert not torch.equal(traj_off, traj_on)


def test_branch_read_text_mask_reshaped_when_st_neq_sv():
    mot = make_tiny_mot(enable_dynamic_branch=True, dynamic_branch_num_layers=1)
    mot.eval()
    vpre, apre, meta = build_tiny_payload(mot, T=3)
    assert meta["Sv"] != 2 * meta["tpf"]
    St = 2 * meta["tpf"]
    kwargs = assemble_mot_kwargs(mot, vpre, apre, _mask(mot, meta, St))
    payload = {"enable": True, "first_frame_tokens": meta["tpf"]}
    mot.dynamic_branch.read_text = True
    mot.forward(**kwargs, dynamic_branch_payload=payload)
    assert torch.isfinite(mot._last_branch_hidden["traj"]).all()
