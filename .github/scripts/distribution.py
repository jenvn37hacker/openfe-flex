"""Build-output verification, checksums and wheel installation for CI/release.

Uses the standard library only; it never publishes or changes Git history.
"""
import argparse
import ast
from email.parser import Parser
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib
import zipfile


def project_version():
    root = Path(__file__).resolve().parents[2]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    module = ast.parse((root / "src/openfe_flex/__init__.py").read_text())
    declared = [ast.literal_eval(node.value) for node in module.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)]
    if declared != [version]:
        raise ValueError("Package __version__ does not match pyproject.toml")
    return version


def artifacts(directory, version):
    directory = Path(directory)
    wheel = directory / f"openfe_flex-{version}-py3-none-any.whl"
    source = directory / f"openfe_flex-{version}.tar.gz"
    actual = {path.name for path in directory.glob("*.whl")} | {
        path.name for path in directory.glob("*.tar.gz")}
    if actual != {wheel.name, source.name}:
        raise ValueError("Require exactly the expected universal wheel and source archive")
    with zipfile.ZipFile(wheel) as archive:
        metadata_paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("Expected one wheel metadata record")
        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode())
        if metadata["Name"] != "openfe-flex" or metadata["Version"] != version:
            raise ValueError("Wheel name/version does not match project")
        if "openfe_flex/engine.py" not in archive.namelist():
            raise ValueError("Wheel is missing its engine")
        if not any(name.endswith("/licenses/LICENSE") for name in archive.namelist()):
            raise ValueError("Wheel is missing the retained MIT license")
    with tarfile.open(source) as archive:
        names = set(archive.getnames())
        prefix = f"openfe_flex-{version}/"
        for name in ["LICENSE", "UPSTREAM.json", "docs/algorithm_alignment.md",
                     "tests/test_paper_core.py", "examples/classification.py"]:
            if prefix + name not in names:
                raise ValueError(f"Source archive is missing {name}")
    return sorted([wheel, source])


def checksums(paths):
    return "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "install"])
    parser.add_argument("--directory", type=Path, default=Path("dist"))
    parser.add_argument("--tag", help="Release tag; must equal v<project version>")
    parser.add_argument("--extras", choices=["", "catboost", "lightgbm", "all"], default="")
    args = parser.parse_args()
    version = project_version()
    if args.tag is not None and args.tag != f"v{version}":
        parser.error(f"Tag must be v{version}")
    paths = artifacts(args.directory, version)
    manifest = args.directory / "SHA256SUMS"
    if args.command == "prepare":
        manifest.write_text(checksums(paths), encoding="utf-8")
    elif not manifest.exists() or manifest.read_text(encoding="utf-8") != checksums(paths):
        raise ValueError("Distribution checksums are missing or do not match")
    if args.command == "install":
        wheel = next(path for path in paths if path.suffix == ".whl")
        extra = f"[{args.extras}]" if args.extras else ""
        requirement = f"openfe-flex{extra} @ {wheel.resolve().as_uri()}"
        subprocess.run([sys.executable, "-m", "pip", "install", requirement], check=True)
    print(f"Verified openfe-flex {version}: {', '.join(path.name for path in paths)}")


if __name__ == "__main__":
    main()
