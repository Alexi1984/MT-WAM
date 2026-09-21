import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tokenize
import zipfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def python_comments(source):
    return [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    ]


def check_python(name, source, manifest):
    tree = ast.parse(source, filename=name, feature_version=(3, 10))
    comments = python_comments(source)
    expected = manifest["legal_comment_sha256"].get(name)
    digest = (
        hashlib.sha256("\n".join(comments).encode()).hexdigest() if comments else None
    )
    require(digest == expected, f"{name}: unexpected or modified comment/attribution")
    for node in ast.walk(tree):
        require(
            not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ),
            f"{name}: standalone string or docstring",
        )
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        require(
            all(module.split(".")[0] != "fastwam" for module in modules),
            f"{name}: retired package import",
        )


def check_yaml(name, source):
    spans = []
    for token in yaml.scan(source):
        if isinstance(token, yaml.ScalarToken):
            start = token.start_mark.index
            if token.style in ("|", ">"):
                start = source.find("\n", start) + 1
            spans.append((start, token.end_mark.index))
    for match in re.finditer("#", source):
        require(
            any(start <= match.start() < end for start, end in spans),
            f"{name}: YAML comment",
        )
    yaml.safe_load(source)
    require(
        "fastwam." not in source and "FASTWAM_" not in source,
        f"{name}: retired configuration target",
    )


def check_shell(name, source):
    quote = None
    escaped = False
    for index, char in enumerate(source):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
        elif char == "#" and (index == 0 or source[index - 1] in " \t\r\n;|&()"):
            raise ValueError(f"{name}: Shell comment")


def check_file(name, data, manifest):
    require(len(data) <= MAX_BYTES, f"{name}: file exceeds the release size limit")
    path = Path(name)
    require(
        path.suffix.lower()
        not in {
            ".pt",
            ".pth",
            ".safetensors",
            ".mp4",
            ".png",
            ".jpg",
            ".pdf",
            ".zip",
            ".gz",
            ".log",
        },
        f"{name}: runtime output or binary asset",
    )
    source = data.decode("utf-8")
    if path.suffix == ".py":
        check_python(name, source, manifest)
    elif path.suffix in (".yaml", ".yml"):
        check_yaml(name, source)
    elif path.suffix == ".sh":
        check_shell(name, source)
    elif path.suffix == ".toml":
        require(not python_comments(source), f"{name}: TOML comment")
    elif path.suffix == ".json":
        json.loads(source)
    if name not in {
        "README.md",
        "NOTICE",
        "LICENSE",
        "scripts/check_release.py",
    } and not name.startswith("licenses/"):
        require(
            not re.search(r"/(?:Users|home|root|share-\d+)/[^\s\"\']+", source),
            f"{name}: machine-specific path",
        )
        require(
            not re.search(r"DYNBRANCH|FASTWAM_|(?:fastwam|FastWAM)[./]", source),
            f"{name}: private or retired namespace",
        )
    if name in manifest["legal_file_sha256"]:
        require(
            hashlib.sha256(data).hexdigest() == manifest["legal_file_sha256"][name],
            f"{name}: legal file changed without an audit update",
        )


def source_files(root):
    if (root / ".git").exists():
        output = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ]
        )
        return {
            name
            for name in output.decode().split("\0")
            if name and (root / name).is_file()
        }
    ignored = {
        ".git",
        ".venv",
        ".deps",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not any(
            part in ignored or part.endswith(".egg-info")
            for part in path.relative_to(root).parts
        )
        and path.name not in {"PKG-INFO", "setup.cfg"}
    }


def check_source(root, manifest):
    expected = set(manifest["files"])
    actual = source_files(root)
    require(
        actual == expected,
        f"Release file set mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}",
    )
    require(
        [
            name
            for name in sorted(actual)
            if name.lower().endswith((".md", ".rst", ".pdf", ".docx", ".tex"))
        ]
        == ["README.md"],
        "README.md must be the only explanatory document",
    )
    for name in sorted(actual):
        path = root / name
        require(not path.is_symlink(), f"{name}: symlink cannot be exported")
        require("fastwam" not in name.lower(), f"{name}: retired file name")
        check_file(name, path.read_bytes(), manifest)
        if path.suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True)
    return {
        "source_files": len(actual),
        "legal_comment_files": len(manifest["legal_comment_sha256"]),
        "status": "passed",
    }


def archive_members(path):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = [item.filename for item in archive.infolist() if not item.is_dir()]
            require(len(names) == len(set(names)), f"{path}: duplicate ZIP members")
            return {name: archive.read(name) for name in names}
    with tarfile.open(path, "r:gz") as archive:
        items = archive.getmembers()
        require(
            all(item.isfile() or item.isdir() for item in items),
            f"{path}: nonregular archive member",
        )
        files = [item for item in items if item.isfile()]
        require(
            len(files) == len({item.name for item in files}),
            f"{path}: duplicate tar members",
        )
        require(
            len({Path(item.name).parts[0] for item in files}) == 1,
            f"{path}: multiple source roots",
        )
        return {
            Path(*Path(item.name).parts[1:]).as_posix(): archive.extractfile(
                item
            ).read()
            for item in files
        }


def check_archive(path, root, manifest):
    members = archive_members(path)
    require(
        all(
            not Path(name).is_absolute() and ".." not in Path(name).parts
            for name in members
        ),
        f"{path}: unsafe member path",
    )
    if path.suffix == ".whl":
        expected = {
            name.removeprefix("src/"): name
            for name in manifest["files"]
            if name.startswith("src/mtwam/")
        }
        prefixes = {name.split("/")[0] for name in members if ".dist-info/" in name}
        require(len(prefixes) == 1, f"{path}: invalid wheel metadata layout")
        prefix = prefixes.pop()
        for name in manifest["legal_file_sha256"]:
            if name.startswith("third_party/"):
                continue
            expected[prefix + "/licenses/" + name] = name
        allowed_extra = {
            prefix + "/" + name
            for name in ("METADATA", "WHEEL", "RECORD", "top_level.txt")
        }
    else:
        expected = {name: name for name in manifest["files"]}
        allowed_extra = {"PKG-INFO", "setup.cfg"} | {
            "src/mt_wam.egg-info/" + name
            for name in (
                "PKG-INFO",
                "SOURCES.txt",
                "dependency_links.txt",
                "requires.txt",
                "top_level.txt",
            )
        }
    require(
        set(expected) <= set(members),
        f"{path}: missing package files {sorted(set(expected) - set(members))}",
    )
    require(
        set(members) <= set(expected) | allowed_extra,
        f"{path}: unexpected package files {sorted(set(members) - set(expected) - allowed_extra)}",
    )
    for name, original in expected.items():
        require(
            members[name] == (root / original).read_bytes(),
            f"{path}: package content differs from {original}",
        )
        check_file(original, members[name], manifest)
    return {"archive": path.name, "files": len(members), "status": "passed"}


def main():
    parser = argparse.ArgumentParser(
        description="Check release file boundaries, annotations, names and built packages."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, action="append", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "scripts/release_manifest.json").read_text())
    result = check_source(root, manifest)
    result["archives"] = [
        check_archive(path.resolve(), root, manifest) for path in args.archive
    ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
