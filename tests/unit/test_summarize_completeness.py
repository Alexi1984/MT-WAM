import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SUMMARIZE = os.path.normpath(
    os.path.join(_HERE, "..", "..", "experiments", "libero", "summarize_results.py")
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "summarize_results_under_test", _SUMMARIZE
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_module()
audit = M.audit_suite_completeness
EXPECTED = M.SUITE_EXPECTED_TASKS


def test_complete_object_suite_no_warning():
    seen = {"libero_object": set(range(10))}
    trials = {("libero_object", t): 50 for t in range(10)}
    assert audit(seen, trials) == []


def test_missing_task_flagged():
    present = [0, 1, 2, 3, 4, 5, 6, 8, 9]
    seen = {"libero_object": set(present)}
    trials = {("libero_object", t): 50 for t in present}
    warnings = audit(seen, trials)
    assert len(warnings) >= 1
    suite, msg = warnings[0]
    assert suite == "libero_object"
    assert "9/10" in msg and "7" in msg


def test_trial_shortfall_flagged():
    seen = {"libero_object": set(range(10))}
    trials = {("libero_object", t): 50 for t in range(10)}
    trials["libero_object", 3] = 30
    warnings = audit(seen, trials)
    assert any(
        ("3" in msg and ("30" in msg or "trial" in msg.lower()) for _, msg in warnings)
    )


def test_unknown_suite_skipped():
    seen = {"some_custom_suite": {0, 1}}
    trials = {("some_custom_suite", 0): 50, ("some_custom_suite", 1): 50}
    assert audit(seen, trials) == []


def test_expected_counts_present():
    for s in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        assert EXPECTED[s] == 10
    assert EXPECTED["libero_90"] == 90
