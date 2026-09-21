import importlib.util
import json
import os
import pytest

pytest.importorskip("pandas", reason="Result aggregation requires pandas")
_HERE = os.path.dirname(os.path.abspath(__file__))
_SUMMARIZE = os.path.normpath(
    os.path.join(_HERE, "..", "..", "experiments", "libero", "summarize_results.py")
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "summarize_results_census_under_test", _SUMMARIZE
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_module()
_OK_RESULT = {
    "total_episodes": 1,
    "successes": 1,
    "duration": 50.0,
    "task_description": "t",
}


def _write_result(root, suite, gpu, task_id, payload=None):
    d = os.path.join(root, suite)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"gpu{gpu}_task{task_id}_results.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload or _OK_RESULT, f)
    return p


def _setup_run(tmp_path):
    root = str(tmp_path)
    _write_result(root, "libero_spatial", 0, 5)
    _write_result(root, "libero_spatial", 1, 7)
    d = os.path.join(root, "libero_spatial")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "gpu2_task9_results.json"), "w", encoding="utf-8") as f:
        f.write('{"total_episodes": 1, "succ')
    _write_result(root, "libero_spatial", 3, 99)
    manifest = os.path.join(root, "manifest.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("libero_spatial,5\nlibero_spatial,7\nlibero_spatial,9\n")
    cls = {
        "libero_spatial": [{"category": "Camera Viewpoints", "difficulty_level": 1}]
        * 100
    }
    cls_json = os.path.join(root, "cls.json")
    with open(cls_json, "w", encoding="utf-8") as f:
        json.dump(cls, f)
    return (root, manifest, cls_json)


def test_census_audit_flags_missing_extra_and_corrupt(tmp_path, capsys):
    root, manifest, cls_json = _setup_run(tmp_path)
    M.summarize_results(
        root, strict=False, classification_json=cls_json, manifest=manifest
    )
    out = capsys.readouterr().out
    assert "1/3 manifest variants have no result" in out
    assert "NOT in the manifest" in out and "99" in out
    assert "CORRUPT result JSON skipped" in out and "gpu2_task9_results.json" in out


def test_census_strict_exits_nonzero(tmp_path):
    root, manifest, cls_json = _setup_run(tmp_path)
    with pytest.raises(SystemExit) as ei:
        M.summarize_results(
            root, strict=True, classification_json=cls_json, manifest=manifest
        )
    assert ei.value.code == 1


def test_census_complete_run_no_warnings(tmp_path, capsys):
    root = str(tmp_path)
    for tid in (5, 7, 9):
        _write_result(root, "libero_spatial", 0, tid)
    manifest = os.path.join(root, "manifest.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("libero_spatial,5\nlibero_spatial,7\nlibero_spatial,9\n")
    cls = {
        "libero_spatial": [{"category": "Camera Viewpoints", "difficulty_level": 1}]
        * 100
    }
    cls_json = os.path.join(root, "cls.json")
    with open(cls_json, "w", encoding="utf-8") as f:
        json.dump(cls, f)
    M.summarize_results(
        root, strict=True, classification_json=cls_json, manifest=manifest
    )
    out = capsys.readouterr().out
    assert "manifest variants" not in out
    assert "CORRUPT" not in out
    with open(os.path.join(root, "summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["completeness_warnings"] == []
    assert "__total_variant_weighted__" in summary["axis_summary"]


def test_ood_without_manifest_backcompat(tmp_path, capsys):
    root, _, cls_json = _setup_run(tmp_path)
    M.summarize_results(root, strict=False, classification_json=cls_json)
    out = capsys.readouterr().out
    assert "manifest variants" not in out and "NOT in the manifest" not in out
    assert "CORRUPT result JSON skipped" in out


def test_duplicate_results_are_rejected_before_aggregation(tmp_path):
    for gpu in (0, 1):
        _write_result(tmp_path, "libero_spatial", gpu, 0)
    with pytest.raises(ValueError, match="Duplicate result"):
        M.summarize_results(str(tmp_path))
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize(
    "updates",
    [
        {"total_episodes": 0},
        {"successes": 2},
        {"duration": float("nan")},
        {"task_suite": "libero_goal"},
        {"task_id": 3},
    ],
)
def test_invalid_result_metadata_is_rejected(tmp_path, updates):
    _write_result(tmp_path, "libero_spatial", 0, 0, {**_OK_RESULT, **updates})
    with pytest.raises(ValueError):
        M.summarize_results(str(tmp_path))
