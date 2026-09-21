import hashlib
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "release_rules", Path(__file__).resolve().parents[2] / "scripts/check_release.py"
)
RULES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RULES)


def test_runtime_strings_are_preserved():
    RULES.check_python(
        "example.py",
        'text = """A string with # characters"""\n',
        {"legal_comment_sha256": {}},
    )
    RULES.check_yaml(
        "example.yaml", 'text: "A # character"\nblock: |\n  # literal text\n'
    )
    RULES.check_shell("example.sh", 'text="# character"\nsize=${#values[@]}\n')


@pytest.mark.parametrize(
    "source",
    [
        "# instruction\nvalue = 1\n",
        '"""module explanation"""\nvalue = 1\n',
        'def function():\n    """explanation"""\n    pass\n',
    ],
)
def test_python_annotations_are_rejected(source):
    with pytest.raises(ValueError):
        RULES.check_python("example.py", source, {"legal_comment_sha256": {}})


@pytest.mark.parametrize(
    "source",
    [
        "# instruction\nvalue: 1\n",
        "value: 1 # instruction\n",
        "value: | # instruction\n  literal text\n",
    ],
)
def test_yaml_comments_are_rejected(source):
    with pytest.raises(ValueError, match="YAML comment"):
        RULES.check_yaml("example.yaml", source)


def test_license_comment_exception_is_bound_to_exact_file_and_text():
    header = "# Copyright Example\n# SPDX-License-Identifier: MIT"
    manifest = {
        "legal_comment_sha256": {
            "example.py": hashlib.sha256(header.encode()).hexdigest()
        }
    }
    RULES.check_python("example.py", header + "\nvalue = 1\n", manifest)
    with pytest.raises(ValueError):
        RULES.check_python("another.py", header + "\nvalue = 1\n", manifest)
    with pytest.raises(ValueError):
        RULES.check_python(
            "example.py", header + "\n# extra explanation\nvalue = 1\n", manifest
        )
