import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

REPOSITORY = "TianxingChen/RoboTwin2.0"
REVISION = "9dc9299c163db059931898a9f0852098a61155a1"
ARCHIVES = {
    "background_texture.zip": (
        10970687027,
        "54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070",
    ),
    "embodiments.zip": (
        219859313,
        "6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6",
    ),
    "objects.zip": (
        3737778549,
        "6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27",
    ),
}


def verify_archive(path, expected_size, expected_digest):
    path = Path(path)
    if path.stat().st_size != expected_size:
        raise ValueError(f"Archive size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_digest:
        raise ValueError(f"Archive SHA256 mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            name = Path(item.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe archive member: {item.filename}")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Archive CRC failure: {bad}")


def configure_embodiments(assets, final_root):
    files = sorted((assets / "embodiments").rglob("*_tmp.yml"))
    if not files:
        raise FileNotFoundError(f"No embodiment configuration templates in {assets}")
    for template in files:
        content = template.read_text(encoding="utf-8")
        content = content.replace("${ASSETS_PATH}", str(final_root)).replace(
            "$ASSETS_PATH", str(final_root)
        )
        template.with_name(template.name.replace("_tmp.yml", ".yml")).write_text(
            content, encoding="utf-8"
        )


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned RoboTwin runtime assets."
    )
    parser.add_argument(
        "--robotwin-root", type=Path, default=project / "third_party/RoboTwin"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=project / ".deps/robotwin-archives"
    )
    args = parser.parse_args()
    root = args.robotwin_root.expanduser().resolve()
    if not (root / "task_config/_eval_step_limit.yml").is_file():
        raise FileNotFoundError(f"RoboTwin task configuration is missing in {root}")
    assets = root / "assets"
    identity = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "archives": {name: digest for name, (_, digest) in ARCHIVES.items()},
    }
    marker = assets / ".mtwam_assets.json"
    if assets.exists():
        if (
            marker.is_file()
            and json.loads(marker.read_text()) == identity
            and all(((assets / Path(name).stem).is_dir() for name in ARCHIVES))
        ):
            configure_embodiments(assets, root)
            print(f"Pinned assets already installed: {assets}")
            return
        raise FileExistsError(
            f"Existing assets are not identified as this pinned snapshot: {assets}"
        )
    if shutil.which("unzip") is None:
        raise RuntimeError("Install unzip before downloading RoboTwin assets.")
    cache = args.cache_dir.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    paths = []
    for filename, (size, digest) in ARCHIVES.items():
        path = Path(
            hf_hub_download(
                repo_id=REPOSITORY,
                repo_type="dataset",
                revision=REVISION,
                filename=filename,
                local_dir=str(cache),
            )
        )
        verify_archive(path, size, digest)
        paths.append(path)
    with tempfile.TemporaryDirectory(prefix=".mtwam-assets-", dir=root) as temporary:
        staged = Path(temporary) / "assets"
        staged.mkdir()
        for path in paths:
            subprocess.run(["unzip", "-q", str(path), "-d", str(staged)], check=True)
        for filename in ARCHIVES:
            if not (staged / Path(filename).stem).is_dir():
                raise FileNotFoundError(
                    f"Archive did not provide {Path(filename).stem}"
                )
        for path in staged.rglob("*"):
            if path.is_symlink() and (not path.resolve().is_relative_to(staged)):
                raise ValueError(
                    f"Asset symlink escapes the installation directory: {path}"
                )
        configure_embodiments(staged, root)
        (staged / marker.name).write_text(json.dumps(identity, indent=2) + "\n")
        staged.rename(assets)
    print(f"Installed verified assets: {assets}")


if __name__ == "__main__":
    main()
