import os
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "selection", [["--config-name=train_libero"], ["--config-name", "train_robotwin"]]
)
def test_launcher_forwards_root_config_and_process_count(tmp_path, selection):
    executable = tmp_path / "accelerate"
    executable.write_text('printf "%s\\n" "$@" > "$MTWAM_CAPTURE_ARGS"\n')
    executable.chmod(493)
    args_file = tmp_path / "args.txt"
    env = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        MTWAM_CAPTURE_ARGS=str(args_file),
        MTWAM_RUN_ID="test",
        MTWAM_NNODES="1",
        MTWAM_NODE_RANK="0",
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train_zero1.sh"), "8", *selection],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    args = args_file.read_text().splitlines()
    assert args[args.index("--num_processes") + 1] == "8"
    assert args[args.index("--num_machines") + 1] == "1"
    assert args[-len(selection) :] == selection
