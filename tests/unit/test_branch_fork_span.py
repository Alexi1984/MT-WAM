import pytest
from mtwam.models.wan22._load_guards import branch_fork_span


@pytest.mark.parametrize("ff", [2, 98, 120])
@pytest.mark.parametrize("num_ref_latents", [1, 2])
def test_dual_f0_forks_one_frame_per_role(ff, num_ref_latents):
    assert branch_fork_span(ff, num_ref_latents, True) == ff
    for video_seq_len in (num_ref_latents * ff, (num_ref_latents + 2) * ff):
        assert (
            branch_fork_span(ff, num_ref_latents, True, video_seq_len=video_seq_len)
            == ff
        )


@pytest.mark.parametrize("ff", [2, 98, 120])
def test_single_reference_forks_one_frame(ff):
    assert branch_fork_span(ff, 1, False) == ff
    assert branch_fork_span(ff, 1, False, video_seq_len=ff) == ff


def test_nondual_clamping_is_explicit():
    assert branch_fork_span(98, 2, False) == 196
    assert branch_fork_span(98, 2, False, video_seq_len=98) == 98


@pytest.mark.parametrize("ff", [98, 120])
def test_main_training_and_inference_token_counts(ff):
    fork_tokens = 2 * branch_fork_span(ff, 2, True)
    assert fork_tokens == 2 * ff
    assert 4 * ff + fork_tokens == 6 * ff
    assert 2 * ff + fork_tokens == 4 * ff
