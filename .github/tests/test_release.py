"""Exercise automatic versions in temporary Git repositories and a local remote."""
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import parse_qs, urlparse

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("openfe_release", SCRIPTS / "release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class FakeGitHub:
    def __init__(self):
        self.release = None
        self.assets = {}
        self.uploads = 0
        self.fail_upload_once = False

    def request(self, method, route, data=None, **kwargs):
        if route.startswith("/releases/tags/"):
            return self.release
        if method == "POST" and route == "/releases":
            self.release = dict(data, id=1, upload_url="https://uploads.github.com/repos/test/repo/releases/1/assets{?name}",
                                html_url="https://github.com/test/repo/releases/tag/" + data["tag_name"])
            return self.release
        if method == "GET":
            return list(self.assets.values())
        if method == "POST":
            name = parse_qs(urlparse(route).query)["name"][0]
            self.uploads += 1
            self.assets[name] = dict(id=self.uploads, name=name, state="uploaded", size=len(data),
                                     digest="sha256:" + sha256(data).hexdigest())
            if self.fail_upload_once:
                self.fail_upload_once = False
                raise RuntimeError("simulated lost upload response")
            return self.assets[name]
        if method == "DELETE":
            identity = int(route.rsplit("/", 1)[1])
            self.assets = {k:v for k,v in self.assets.items() if v["id"] != identity}
            return None
        if method == "PATCH":
            self.release.update(data)
            return self.release
        raise AssertionError((method, route))


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "work"
        self.root.mkdir()
        self.remote = Path(self.temp.name) / "remote.git"
        release.run(self.root, "git", "init", "-b", "master")
        release.git(self.root, "config", "user.name", "Release Test")
        release.git(self.root, "config", "user.email", "test@example.invalid")
        (self.root / "src/openfe_flex").mkdir(parents=True)
        (self.root / "src/openfe_flex/__init__.py").write_text('__version__ = "0.4.0"\n')
        (self.root / "pyproject.toml").write_text('[project]\nname = "openfe-flex"\nversion = "0.4.0"\n')
        (self.root / "CHANGELOG.md").write_text("# Changelog\n\n## 0.4.0\n\n- First version.\n")
        self.commit("initial")
        release.run(self.root, "git", "init", "--bare", "-b", "master", str(self.remote))
        release.git(self.root, "remote", "add", "origin", str(self.remote))
        release.git(self.root, "push", "origin", "HEAD:master")

    def commit(self, message):
        release.git(self.root, "add", ".")
        release.git(self.root, "commit", "-m", message)
        return release.git(self.root, "rev-parse", "HEAD")

    def change(self):
        (self.root / "change.txt").write_text("new functionality")
        return self.commit("change implementation")

    def prepare(self, **kwargs):
        return release.prepare(self.root, event="push", trigger_sha=release.git(self.root, "rev-parse", "HEAD"),
                               allow_release=True, **kwargs)

    def test_plan_is_read_only_and_first_release_uses_current_version(self):
        head = release.git(self.root, "rev-parse", "HEAD")
        planned = release.plan(self.root)
        self.assertEqual(planned["version"], "0.4.0")
        self.assertEqual(release.git(self.root, "tag"), "")
        self.assertEqual(release.git(self.root, "status", "--porcelain"), "")
        result = self.prepare()
        self.assertEqual(result["ref"], head)
        self.assertEqual(release.git(self.remote, "rev-parse", "refs/tags/v0.4.0"), head)

    def test_patch_bump_synchronizes_files_and_retry_reuses_version(self):
        self.prepare()
        trigger = self.change()
        result = self.prepare()
        self.assertEqual(result["version"], "0.4.1")
        self.assertEqual(release.project_version(self.root), "0.4.1")
        self.assertIn("## 0.4.1", (self.root / "CHANGELOG.md").read_text())
        count = release.git(self.root, "rev-list", "--count", "HEAD")
        retried = release.prepare(self.root, event="push", trigger_sha=trigger, allow_release=True)
        self.assertEqual(retried, result)
        self.assertEqual(release.git(self.root, "rev-list", "--count", "HEAD"), count)
        self.assertEqual(release.git(self.remote, "rev-parse", "refs/heads/master"), result["ref"])

    def test_published_version_is_skipped_on_retry(self):
        result = self.prepare()
        client = FakeGitHub()
        client.release = {"draft":False}
        retried = self.prepare(client=client)
        self.assertTrue(retried["skip"])
        self.assertFalse(retried["release"])
        self.assertEqual(retried["version"], result["version"])

    def test_nonrelease_branch_and_stale_push_do_not_stamp(self):
        head = release.git(self.root, "rev-parse", "HEAD")
        idle = release.prepare(self.root, event="pull_request", trigger_sha=head, allow_release=False)
        self.assertFalse(idle["release"])
        self.assertEqual(release.git(self.root, "tag"), "")
        self.change()
        stale = release.prepare(self.root, event="push", trigger_sha=head, allow_release=True)
        self.assertTrue(stale["skip"])
        self.assertEqual(release.git(self.root, "tag"), "")

    def test_minor_major_and_invalid_source(self):
        self.assertEqual(release.bump_version("0.4.2", "minor"), "0.5.0")
        self.assertEqual(release.bump_version("0.4.2", "major"), "1.0.0")
        self.prepare()
        self.change()
        result = self.prepare(bump="minor", push=False)
        self.assertEqual(result["version"], "0.5.0")
        self.assertEqual(release.project_version(self.root), "0.5.0")
        with self.assertRaises(ValueError):
            release.version_tuple("01.2.3")

    def test_dirty_checkout_is_rejected(self):
        (self.root / "CHANGELOG.md").write_text("changed")
        with self.assertRaisesRegex(ValueError, "clean tracked"):
            self.prepare()
        self.assertEqual(release.git(self.root, "tag"), "")

    def test_manual_minor_on_tagged_head_and_dispatch_retry(self):
        first = self.prepare()
        trigger = first["ref"]
        result = release.prepare(self.root,event="workflow_dispatch",trigger_sha=trigger,
                                 allow_release=True,bump="minor")
        self.assertEqual(result["version"],"0.5.0")
        retried = release.prepare(self.root,event="workflow_dispatch",trigger_sha=trigger,
                                  allow_release=True,bump="minor")
        self.assertEqual(retried,result)

    def test_atomic_push_does_not_force_remote_or_leave_partial_tag(self):
        other = Path(self.temp.name) / "other"
        release.run(self.root, "git", "clone", str(self.remote), str(other))
        release.git(other, "config", "user.name", "Other")
        release.git(other, "config", "user.email", "other@example.invalid")
        (other / "other.txt").write_text("a newer remote commit")
        release.git(other, "add", ".")
        release.git(other, "commit", "-m", "advance remote")
        release.git(other, "push", "origin", "master")
        remote_head = release.git(self.remote, "rev-parse", "refs/heads/master")
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.assertEqual(release.git(self.remote, "rev-parse", "refs/heads/master"), remote_head)
        self.assertIsNone(release.tag_commit(self.remote, "v0.4.0"))


class PublicationTests(unittest.TestCase):
    def test_interrupted_draft_upload_retries_and_public_assets_are_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            files = [Path(folder) / name for name in ("wheel.whl", "source.tar.gz", "SHA256SUMS")]
            for path in files:
                path.write_bytes(path.name.encode())
            client = FakeGitHub()
            client.fail_upload_once = True
            with self.assertRaises(RuntimeError):
                release.publish_assets(client, "v0.4.0", files)
            self.assertTrue(client.release["draft"])
            url = release.publish_assets(client, "v0.4.0", files)
            self.assertFalse(client.release["draft"])
            self.assertEqual(client.uploads, 3)
            self.assertEqual(release.publish_assets(client, "v0.4.0", files), url)
            self.assertEqual(client.uploads, 3)
            files[0].write_bytes(b"different build")
            with self.assertRaises(ValueError):
                release.publish_assets(client, "v0.4.0", files)
            self.assertEqual(client.uploads, 3)

    def test_malformed_client_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            release.GitHub("bad repository", "token")
        client = release.GitHub("owner/repo", "token")
        with self.assertRaises(ValueError):
            client.request("GET", "https://example.invalid/steal")
