import importlib.util
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SAMPLER = os.path.normpath(
    os.path.join(_HERE, "..", "..", "experiments", "libero", "sample_ood_variants.py")
)


def _load_module():
    spec = importlib.util.spec_from_file_location("sample_ood_under_test", _SAMPLER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_module()


def _mk_json(tmp_path):
    data = {
        "libero_spatial": [
            {
                "name": f"s{i}",
                "category": "Camera Viewpoints",
                "difficulty_level": 1 + i % 5,
            }
            for i in range(8)
        ]
        + [
            {"name": f"sl{i}", "category": "Light Conditions", "difficulty_level": None}
            for i in range(2)
        ],
        "libero_goal": [
            {
                "name": f"g{i}",
                "category": "Light Conditions",
                "difficulty_level": 1 + i % 3,
            }
            for i in range(6)
        ]
        + [{"name": "gl0", "category": "Light Conditions", "difficulty_level": None}],
    }
    p = os.path.join(str(tmp_path), "cls.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return (p, data)


def test_default_drops_null_unchanged(tmp_path):
    p, _ = _mk_json(tmp_path)
    manifest, sidecar, report = M.sample_variants(
        classification_json=p,
        suites=["libero_spatial", "libero_goal"],
        per_axis_n=100,
        seed=1,
    )
    assert report["dropped_null"] == {"Light Conditions": 3}
    assert report.get("kept_null", {}) == {}
    assert len(manifest) == 8 + 6
    assert all((d != -1 for _, _, _, d in sidecar))


def test_keep_null_census(tmp_path):
    p, data = _mk_json(tmp_path)
    total = sum((len(v) for v in data.values()))
    manifest, sidecar, report = M.sample_variants(
        classification_json=p,
        suites=["libero_spatial", "libero_goal"],
        per_axis_n=100,
        seed=1,
        keep_null_difficulty=True,
    )
    assert len(manifest) == total == 17
    assert report["dropped_null"] == {}
    assert report["kept_null"] == {"Light Conditions": 3}
    null_rows = [(s, t) for s, t, _, d in sidecar if d == -1]
    assert sorted(null_rows) == [
        ("libero_goal", 6),
        ("libero_spatial", 8),
        ("libero_spatial", 9),
    ]


def test_keep_null_deterministic(tmp_path):
    p, _ = _mk_json(tmp_path)
    kw = dict(
        classification_json=p,
        suites=["libero_spatial", "libero_goal"],
        per_axis_n=2,
        seed=42,
        keep_null_difficulty=True,
    )
    a = M.sample_variants(**kw)
    b = M.sample_variants(**kw)
    assert a[0] == b[0] and a[1] == b[1]


def test_keep_null_subset_property(tmp_path):
    p, _ = _mk_json(tmp_path)
    sub, _, _ = M.sample_variants(
        classification_json=p,
        suites=["libero_spatial", "libero_goal"],
        per_axis_n=100,
        seed=1,
    )
    full, _, _ = M.sample_variants(
        classification_json=p,
        suites=["libero_spatial", "libero_goal"],
        per_axis_n=100,
        seed=1,
        keep_null_difficulty=True,
    )
    assert set(sub).issubset(set(full))
