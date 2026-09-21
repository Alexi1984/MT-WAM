from factories import make_tiny_mot


def test_tiny_mot_constructs():
    mot = make_tiny_mot()
    assert mot.expert_order == ["video", "action"]
    assert mot.num_layers == 4
    assert mot.num_heads == 2
    assert mot.attn_head_dim == 8
