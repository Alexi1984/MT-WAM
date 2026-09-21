TOTAL_SPATIAL_DOWNSAMPLE = 32


def _token_grid(canvas_h, canvas_w):
    assert canvas_h % TOTAL_SPATIAL_DOWNSAMPLE == 0, canvas_h
    assert canvas_w % TOTAL_SPATIAL_DOWNSAMPLE == 0, canvas_w
    return (canvas_h // TOTAL_SPATIAL_DOWNSAMPLE, canvas_w // TOTAL_SPATIAL_DOWNSAMPLE)


def _robotwin_token_regions(gh, gw):
    top_rows = gh * 2 // 3
    mid_col = gw // 2
    high, left, right = (set(), set(), set())
    for r in range(gh):
        for c in range(gw):
            idx = r * gw + c
            if r < top_rows:
                high.add(idx)
            elif c < mid_col:
                left.add(idx)
            else:
                right.add(idx)
    return (high, left, right)


def test_downsample_factor_matches_known_libero_sv():
    gh, gw = _token_grid(224, 448)
    assert (gh, gw) == (7, 14)
    assert gh * gw == 98


def test_robotwin_token_grid_is_12x10():
    gh, gw = _token_grid(384, 320)
    assert (gh, gw) == (12, 10)
    assert gh * gw == 120


def test_robotwin_per_camera_token_partition():
    gh, gw = (12, 10)
    high, left, right = _robotwin_token_regions(gh, gw)
    assert (len(high), len(left), len(right)) == (80, 20, 20)
    assert high.isdisjoint(left) and high.isdisjoint(right) and left.isdisjoint(right)
    assert high | left | right == set(range(120))
    assert high == set(range(80))
    assert left == {r * gw + c for r in range(8, 12) for c in range(0, 5)}
    assert right == {r * gw + c for r in range(8, 12) for c in range(5, 10)}


def test_robotwin_wrists_are_row_interleaved_not_contiguous():
    gh, gw = (12, 10)
    _, left, right = _robotwin_token_regions(gh, gw)
    assert {80, 81, 82, 83, 84} <= left
    assert {85, 86, 87, 88, 89} <= right
    assert sorted(left) != list(range(min(left), max(left) + 1))
    assert sorted(left) == [
        80,
        81,
        82,
        83,
        84,
        90,
        91,
        92,
        93,
        94,
        100,
        101,
        102,
        103,
        104,
        110,
        111,
        112,
        113,
        114,
    ]
