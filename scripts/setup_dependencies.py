import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

SOURCES = {
    "cotracker": (
        "https://github.com/facebookresearch/co-tracker.git",
        "82e02e8029753ad4ef13cf06be7f4fc5facdda4d",
    ),
    "dinov2": (
        "https://github.com/facebookresearch/dinov2.git",
        "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
    ),
    "libero": (
        "https://github.com/Lifelong-Robot-Learning/LIBERO.git",
        "8f1084e3132a39270c3a13ebe37270a43ece2a01",
    ),
    "libero-plus": (
        "https://github.com/sylvestf/LIBERO-plus.git",
        "4976dc30028e805ff8094b55501d532c48fec182",
    ),
    "curobo": (
        "https://github.com/NVlabs/curobo.git",
        "d64c4b005459db10c5dd867d8b30a87d5bda9bdb",
    ),
}
COTRACKER_FILE = "cotracker/models/core/cotracker/cotracker3_offline.py"
COTRACKER_SHA256 = "41cff5b775ca071ff12bce23ef7887cb639171da84a32e0d75194661101ce238"
DINO_REQUIREMENTS_SHA256 = (
    "553faa42e1ad3975631e46df26bcfd76ec3add6beacb9ede3f8ef22393f5eefc"
)
DINO_REQUIREMENTS = "--extra-index-url https://download.pytorch.org/whl/cu117\ntorch==2.0.0\ntorchvision==0.15.0\nomegaconf\ntorchmetrics==0.10.3\nfvcore\niopath\nxformers==0.0.18\nsubmitit\n--extra-index-url https://pypi.nvidia.com\ncuml-cu11\n"


def checked_replace(path, expected_sha256, before, after):
    path = Path(path)
    source = path.read_bytes()
    before, after = (before.encode(), after.encode())
    digest = hashlib.sha256(source).hexdigest()
    if digest == expected_sha256:
        if source.count(before) != 1:
            raise ValueError(f"Expected exactly one patch location in {path}")
        updated = source.replace(before, after)
        path.write_bytes(updated)
        return "patched"
    if (
        source.count(after) == 1
        and hashlib.sha256(source.replace(after, before)).hexdigest() == expected_sha256
    ):
        return "already patched"
    raise ValueError(
        f"Unrecognized source in {path}; patch source hash mismatch (sha256={digest})"
    )


def checkout_source(name, destination, allowed_changes=()):
    url, revision = SOURCES[name]
    destination = Path(destination).resolve()
    if not destination.exists():
        destination.mkdir(parents=True)
        subprocess.run(["git", "init", str(destination)], check=True)
        subprocess.run(
            ["git", "-C", str(destination), "remote", "add", "origin", url], check=True
        )
        subprocess.run(
            ["git", "-C", str(destination), "fetch", "--depth=1", "origin", revision],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
        )

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(destination), *args], text=True
        ).strip()

    if (
        git("rev-parse", "HEAD") != revision
        or git("remote", "get-url", "origin") != url
    ):
        raise ValueError(f"{destination} is not the pinned {name} source")
    changed = set(git("diff", "HEAD", "--name-only").splitlines())
    if changed - set(allowed_changes):
        raise ValueError(f"Unexpected changes in {destination}: {sorted(changed)}")
    return destination


def install_source(path, build_isolation=True):
    command = [sys.executable, "-m", "pip", "install", "--no-deps"]
    if not build_isolation:
        command.append("--no-build-isolation")
    command.append(str(path))
    subprocess.run(command, check=True)


def configure_libero(source, directory):
    import yaml

    benchmark = Path(source).resolve() / "libero/libero"
    directory = Path(directory).resolve()
    values = {
        "benchmark_root": str(benchmark),
        "bddl_files": str(benchmark / "bddl_files"),
        "init_states": str(benchmark / "init_files"),
        "datasets": str(benchmark.parent / "datasets"),
        "assets": str(benchmark / "assets"),
    }
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "config.yaml"
    if target.exists() and yaml.safe_load(target.read_text()) != values:
        raise ValueError(f"Existing LIBERO resource configuration differs: {target}")
    target.write_text(yaml.safe_dump(values, sort_keys=False))
    print(f"Set LIBERO_CONFIG_PATH={directory} when running this benchmark.")


def install_teachers(directory):
    cotracker = checkout_source(
        "cotracker", directory / "cotracker", [COTRACKER_FILE, "setup.py"]
    )
    status = checked_replace(
        cotracker / COTRACKER_FILE,
        COTRACKER_SHA256,
        "coords_init = coords.view(B * T, N, 2)",
        "coords_init = coords.reshape(B * T, N, 2)",
    )
    (cotracker / "MTWAM_PATCH_NOTICE").write_text(
        "MT-WAM modification: replace one view with reshape in cotracker3_offline.py to support non-contiguous batched coordinates. Original copyright and CC BY-NC 4.0 license are retained.\n"
    )
    print(f"CoTracker: {status}")
    checked_replace(
        cotracker / "setup.py",
        "efe939de63c26ad3045fcde411893429f83d4cdfe1909a184897cf9aa803e5ab",
        'packages=find_packages(exclude="notebooks"),',
        'packages=find_packages(exclude="notebooks"),\n    license_files=("LICENSE.md", "MTWAM_PATCH_NOTICE"),',
    )
    dino = checkout_source(
        "dinov2", directory / "dinov2", ["requirements.txt", "setup.py"]
    )
    checked_replace(
        dino / "requirements.txt",
        DINO_REQUIREMENTS_SHA256,
        DINO_REQUIREMENTS,
        "torch==2.7.1\ntorchvision==0.22.1\nomegaconf==2.3.0\n",
    )
    (dino / "MTWAM_PATCH_NOTICE").write_text(
        "MT-WAM modification: requirements.txt specifies the frozen backbone inference dependencies used by MT-WAM. DINOv2 model code is unchanged. Original copyright and Apache-2.0 license are retained.\n"
    )
    checked_replace(
        dino / "setup.py",
        "8ec72d3be027aee099a7b7b1c36c874be6d7ade4dd9eb69baff3975b586820ae",
        'license_files=("LICENSE",),',
        'license_files=("LICENSE", "MTWAM_PATCH_NOTICE"),',
    )
    install_source(cotracker)
    install_source(dino)


def patch_robotwin_dependencies():
    for name, version in (("sapien", "3.0.0b1"), ("mplib", "0.2.1")):
        if importlib.metadata.version(name) != version:
            raise ValueError(
                f"Install {name}=={version} before applying the RoboTwin patches"
            )
    sapien = (
        Path(importlib.util.find_spec("sapien").origin).parent
        / "wrapper/urdf_loader.py"
    )
    mplib = Path(importlib.util.find_spec("mplib").origin).parent / "planner.py"
    replacements = [
        (
            sapien,
            'with open(urdf_file, "r") as f:',
            'with open(urdf_file, "r", encoding="utf-8") as f:',
        ),
        (
            sapien,
            'with open(srdf_file, "r") as f:',
            'with open(srdf_file, "r", encoding="utf-8") as f:',
        ),
        (
            mplib,
            "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
            "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
        ),
    ]
    updates = {}
    for path, before, after in replacements:
        content = updates.get(path, path.read_text())
        if content.count(before) == 1 and content.count(after) == 0:
            updates[path] = content.replace(before, after)
        elif content.count(before) == 0 and content.count(after) == 1:
            updates[path] = content
        else:
            raise ValueError(f"Unknown patch location in {path}; no files modified")
    for path, content in updates.items():
        path.write_text(content)


def main():
    parser = argparse.ArgumentParser(
        description="Install pinned teacher or benchmark source dependencies."
    )
    parser.add_argument(
        "component", choices=["teachers", "libero", "libero-plus", "robotwin"]
    )
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).resolve().parents[1] / ".deps"
    )
    args = parser.parse_args()
    directory = args.directory.expanduser().resolve()
    if args.component == "teachers":
        install_teachers(directory)
    elif args.component == "robotwin":
        if sys.platform != "linux":
            raise RuntimeError(
                "RoboTwin installation requires Linux and the CUDA toolkit."
            )
        subprocess.run(["git", "lfs", "version"], check=True)
        patch_robotwin_dependencies()
        source = checkout_source("curobo", directory / "curobo")
        install_source(source, build_isolation=False)
    else:
        source = checkout_source(args.component, directory / args.component)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(source)],
            check=True,
        )
        configure_libero(source, directory / f"{args.component}-config")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()
