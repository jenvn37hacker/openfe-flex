"""Install a verified OpenFE-Flex GitHub Release into this Python environment.

Standard library only. Default: latest stable Release plus CatBoost dependencies.
Use --version 0.4.0 to pin; --print-requirement emits a hash-pinned URL for pip.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REPOSITORY = "jenvn37hacker/openfe-flex"
VERSION = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\Z")


def read_url(url):
    request = Request(url, headers={"User-Agent":"openfe-flex-installer"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except HTTPError as error:
        if error.code == 404:
            raise RuntimeError("No published Release found. Wait for a successful master release or check --version.") from None
        raise RuntimeError(f"GitHub download failed with HTTP {error.code}; check connectivity/API rate limits.") from None


def resolve_release(version=None):
    if version is not None and not VERSION.fullmatch(version):
        raise ValueError("--version requires a stable major.minor.patch value")
    endpoint = "/releases/latest" if version is None else "/releases/tags/v" + version
    release = json.loads(read_url("https://api.github.com/repos/" + REPOSITORY + endpoint))
    tag = release.get("tag_name", "")
    if not tag.startswith("v") or not VERSION.fullmatch(tag[1:]):
        raise ValueError("Release has no stable version tag")
    if version is not None and tag != "v" + version:
        raise ValueError("Release version does not match the requested version")
    if release.get("draft") or release.get("prerelease"):
        raise ValueError("Only published stable Releases are supported")
    wheel_name = f"openfe_flex-{tag[1:]}-py3-none-any.whl"
    assets = release.get("assets", [])
    selected = {}
    for name in (wheel_name, "SHA256SUMS"):
        matches = [asset for asset in assets if asset.get("name") == name]
        if len(matches) != 1 or matches[0].get("state") != "uploaded":
            raise ValueError(f"Release has no unique uploaded {name}")
        expected_url = f"https://github.com/{REPOSITORY}/releases/download/{tag}/{name}"
        if matches[0].get("browser_download_url") != expected_url:
            raise ValueError("Unexpected Release download URL")
        selected[name] = matches[0]
    manifest = read_url(selected["SHA256SUMS"]["browser_download_url"]).decode("utf-8")
    hashes = {}
    for line in manifest.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if match is None or match[2] in hashes:
            raise ValueError("Invalid or ambiguous SHA256SUMS")
        hashes[match[2]] = match[1]
    if wheel_name not in hashes:
        raise ValueError("Wheel checksum is missing")
    asset = selected[wheel_name]
    if asset.get("digest") not in (None, "sha256:" + hashes[wheel_name]):
        raise ValueError("GitHub asset digest and manifest disagree")
    return tag, asset, hashes[wheel_name]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="Pin a stable version instead of latest")
    parser.add_argument("--extras", choices=("none", "catboost", "lightgbm", "all"), default="catboost")
    parser.add_argument("--print-requirement", action="store_true", help="Print a hash-pinned dependency; install nothing")
    args = parser.parse_args(argv)
    try:
        tag, asset, digest = resolve_release(args.version)
        extra = "" if args.extras == "none" else f"[{args.extras}]"
        if args.print_requirement:
            print(f"openfe-flex{extra} @ {asset['browser_download_url']}#sha256={digest}")
            return
        with TemporaryDirectory(prefix="openfe-flex-install-") as folder:
            content = read_url(asset["browser_download_url"])
            if sha256(content).hexdigest() != digest:
                raise ValueError("Downloaded wheel checksum does not match; installation stopped")
            if len(content) != asset.get("size"):
                raise ValueError("Downloaded wheel size does not match")
            wheel = Path(folder) / asset["name"]
            wheel.write_bytes(content)
            print(f"Installing OpenFE-Flex {tag} with {sys.executable}", flush=True)
            requirement = f"openfe-flex{extra} @ {wheel.resolve().as_uri()}"
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", requirement], check=True)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
