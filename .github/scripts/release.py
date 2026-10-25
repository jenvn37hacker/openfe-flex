"""Automatic version preparation and verified, retryable GitHub Releases.

Only CI calls prepare/publish. `plan` is read-only. No PyPI publication is used.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from distribution import artifacts, checksums, project_version

VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")


def run(root, *command):
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=180)
    if result.returncode:
        message = (result.stderr or result.stdout).strip()
        token = os.environ.get("GH_TOKEN")
        if token:
            message = message.replace(token, "***")
        raise RuntimeError(f"{command[0]} failed: {message}")
    return result.stdout.strip()


def git(root, *args):
    return run(root, "git", *args)


def version_tuple(value):
    match = VERSION.fullmatch(value)
    if match is None:
        raise ValueError("Releases require a stable major.minor.patch version")
    return tuple(map(int, match.groups()))


def bump_version(value, bump):
    major, minor, patch = version_tuple(value)
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def tag_commit(root, tag):
    result = subprocess.run(["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
                            cwd=root, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def plan(root, *, bump="auto"):
    if bump not in ("auto", "patch", "minor", "major"):
        raise ValueError("Unknown version bump")
    current = project_version(root)
    version_tuple(current)
    head = git(root, "rev-parse", "HEAD")
    tags = [tag for tag in git(root, "tag", "--list", "v*").splitlines()
            if VERSION.fullmatch(tag[1:])]
    if tags:
        latest = max(tags, key=lambda tag: version_tuple(tag[1:]))
        if latest != f"v{current}":
            raise ValueError("Source version differs from latest tag; pull CI's version commit first")
        git(root, "merge-base", "--is-ancestor", latest, head)
    existing = tag_commit(root, f"v{current}")
    if existing == head and bump == "auto":
        # Retrying a failed build/test/upload reuses the prepared version.
        return dict(version=current, tag=f"v{current}", ref=head, release=True, skip=False)
    next_version = current if not tags and bump == "auto" else bump_version(current, bump)
    return dict(version=next_version, tag=f"v{next_version}", ref=head, release=True, skip=False)


def stamp(root, version):
    root = Path(root)
    current = project_version(root)
    if current == version:
        return False
    version_tuple(version)
    replacements = [
        (root / "pyproject.toml", rf'(?m)^version = "{re.escape(current)}"$', f'version = "{version}"'),
        (root / "src/openfe_flex/__init__.py", rf'(?m)^__version__ = "{re.escape(current)}"$',
         f'__version__ = "{version}"'),
    ]
    contents = []
    for path, pattern, replacement in replacements:
        text, count = re.subn(pattern, replacement, path.read_text(encoding="utf-8"))
        if count != 1:
            raise ValueError(f"Cannot uniquely update version in {path.name}")
        contents.append((path, text))
    for path, text in contents:
        path.write_text(text, encoding="utf-8")
    changelog = root / "CHANGELOG.md"
    text = changelog.read_text(encoding="utf-8")
    entry = f"## {version}\n\n- Automated release for updates through `{git(root, 'rev-parse', '--short', 'HEAD')}`.\n\n"
    if f"## {version}\n" not in text:
        text = text.replace("# Changelog\n\n", "# Changelog\n\n" + entry, 1)
        changelog.write_text(text, encoding="utf-8")
    return True


def prepare(root, *, event, trigger_sha, allow_release, bump="auto", push=True, client=None):
    root = Path(root)
    head = git(root, "rev-parse", "HEAD")
    current = project_version(root)
    idle = dict(version=current, tag=f"v{current}", ref=head, release=False, skip=False)
    if not allow_release:
        return idle
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Version preparation requires a clean tracked checkout")
    if event in ("push", "workflow_dispatch") and trigger_sha and head != trigger_sha:
        parent = git(root, "rev-parse", "HEAD^")
        recovery = (parent == trigger_sha and tag_commit(root, f"v{current}") == head
                    and git(root, "log", "-1", "--format=%s") == f"chore(release): {current} [skip ci]")
        if not recovery:
            return idle | {"skip":True}
        # Explicit manual minor/major reruns must reuse their prepared version.
        bump = "auto"
    result = plan(root, bump=bump)
    if tag_commit(root, result["tag"]) == head and client is not None:
        published = client.request("GET", "/releases/tags/" + quote(result["tag"], safe=""), missing_ok=True)
        if published is not None and not published["draft"]:
            return result | {"release":False, "skip":True}
    if stamp(root, result["version"]):
        git(root, "add", "pyproject.toml", "src/openfe_flex/__init__.py", "CHANGELOG.md")
        git(root, "commit", "-m", f"chore(release): {result['version']} [skip ci]")
    result["ref"] = git(root, "rev-parse", "HEAD")
    existing = tag_commit(root, result["tag"])
    if existing is None:
        git(root, "tag", result["tag"], result["ref"])
    elif existing != result["ref"]:
        raise ValueError("An existing release tag points to a different commit")
    if push:
        # No force pushes; branch and tag land together or neither lands.
        git(root, "push", "--atomic", "origin", "HEAD:refs/heads/master", f"refs/tags/{result['tag']}")
    return result


class GitHub:
    def __init__(self, repository, token):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository) or not token:
            raise ValueError("GITHUB_REPOSITORY and GH_TOKEN are required")
        self.base = "https://api.github.com/repos/" + repository
        self.token = token

    def request(self, method, path, data=None, *, binary=False, missing_ok=False):
        url = path if path.startswith("https://") else self.base + path
        if urlparse(url).scheme != "https" or urlparse(url).netloc not in ("api.github.com", "uploads.github.com"):
            raise ValueError("Unexpected GitHub API host")
        body = data if binary else json.dumps(data).encode() if data is not None else None
        headers = {"Authorization":"Bearer " + self.token, "Accept":"application/vnd.github+json",
                   "User-Agent":"openfe-flex-release", "X-GitHub-Api-Version":"2022-11-28"}
        if body is not None:
            headers["Content-Type"] = "application/octet-stream" if binary else "application/json"
        try:
            with urlopen(Request(url, data=body, headers=headers, method=method), timeout=30) as response:
                content = response.read()
                return json.loads(content) if content else None
        except HTTPError as error:
            if error.code == 404 and missing_ok:
                return None
            raise RuntimeError(f"GitHub API {method} failed with HTTP {error.code}") from None


def publish_assets(client, tag, files):
    """Upload to a draft, verify asset digests, then make it public.

    Reruns never replace public assets. Incomplete drafts can be repaired.
    """
    release = client.request("GET", "/releases/tags/" + quote(tag, safe=""), missing_ok=True)
    expected = {path.name: path for path in files}
    if release is None:
        release = client.request("POST", "/releases", {
            "tag_name":tag, "name":f"OpenFE-Flex {tag}", "draft":True, "generate_release_notes":True})
    route = f"/releases/{release['id']}/assets?per_page=100"
    assets = {asset["name"]:asset for asset in client.request("GET", route)}
    if release["draft"]:
        upload = release["upload_url"].split("{", 1)[0]
        for name, path in expected.items():
            content = path.read_bytes()
            digest = "sha256:" + sha256(content).hexdigest()
            old = assets.get(name)
            if old and old.get("state") == "uploaded" and old.get("digest") == digest:
                continue
            if old:
                client.request("DELETE", f"/releases/assets/{old['id']}")
            client.request("POST", upload + "?" + urlencode({"name":name}), content, binary=True)
        assets = {asset["name"]:asset for asset in client.request("GET", route)}
    for name, path in expected.items():
        asset = assets.get(name)
        if (not asset or asset.get("state") != "uploaded" or asset.get("size") != path.stat().st_size
                or asset.get("digest") != "sha256:" + sha256(path.read_bytes()).hexdigest()):
            raise ValueError("Release assets are incomplete or have different digests; publication stopped")
    if release["draft"]:
        release = client.request("PATCH", f"/releases/{release['id']}", {"draft":False, "make_latest":"true"})
    return release["html_url"]


def outputs(result):
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            for key, value in result.items():
                stream.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")
    print(json.dumps(result, indent=2))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(f"Version: **{result['version']}** | Release: **{result['release']}**\n\n"
                         f"Source commit: `{result['ref']}`\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "prepare", "publish"))
    parser.add_argument("--directory", type=Path, default=Path("dist"))
    parser.add_argument("--bump", choices=("auto", "patch", "minor", "major"), default="auto")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.command == "plan":
        outputs(plan(root, bump=args.bump))
    else:
        client = GitHub(os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GH_TOKEN", ""))
        if args.command == "prepare":
            outputs(prepare(root, event=os.environ.get("GITHUB_EVENT_NAME", ""),
                            trigger_sha=os.environ.get("GITHUB_SHA"),
                            allow_release=os.environ.get("GITHUB_REF") == "refs/heads/master"
                                          and os.environ.get("GITHUB_EVENT_NAME") in ("push", "workflow_dispatch"),
                            bump=args.bump, client=client))
        else:
            version = project_version(root)
            tag = os.environ["RELEASE_TAG"]
            if tag != f"v{version}" or tag_commit(root, tag) != git(root, "rev-parse", "HEAD"):
                raise ValueError("Release tag must identify the tested source commit")
            files = artifacts(args.directory, version)
            manifest = args.directory / "SHA256SUMS"
            if not manifest.exists() or manifest.read_text(encoding="utf-8") != checksums(files):
                raise ValueError("Distribution checksums do not match")
            print(publish_assets(client, tag, files + [manifest]))


if __name__ == "__main__":
    main()
