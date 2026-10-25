"""Installer tests use fake GitHub responses and never execute pip."""
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/install_latest.py"
spec = importlib.util.spec_from_file_location("openfe_installer", SCRIPT)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.data = b"a test wheel"
        self.digest = sha256(self.data).hexdigest()
        self.name = "openfe_flex-0.4.0-py3-none-any.whl"
        self.prefix = "https://github.com/jenvn37hacker/openfe-flex/releases/download/v0.4.0/"
        self.release = {"tag_name":"v0.4.0", "draft":False, "prerelease":False, "assets":[
            {"name":self.name, "state":"uploaded", "browser_download_url":self.prefix+self.name,
             "size":len(self.data), "digest":"sha256:"+self.digest},
            {"name":"SHA256SUMS", "state":"uploaded", "browser_download_url":self.prefix+"SHA256SUMS"},
        ]}
        self.manifest = f"{self.digest}  {self.name}\n".encode()

    def read(self, url):
        if url.startswith("https://api.github.com/"):
            return json.dumps(self.release).encode()
        return self.manifest if url.endswith("SHA256SUMS") else self.data

    def test_latest_requirement_has_resolved_version_extras_and_hash_without_pip(self):
        with patch.object(installer, "read_url", side_effect=self.read), patch.object(installer.subprocess, "run") as run:
            output = io.StringIO()
            with redirect_stdout(output):
                installer.main(["--print-requirement", "--extras", "lightgbm"])
            self.assertEqual(output.getvalue().strip(),
                             f"openfe-flex[lightgbm] @ {self.prefix+self.name}#sha256={self.digest}")
            run.assert_not_called()

    def test_pin_and_verified_install_use_current_python(self):
        with patch.object(installer, "read_url", side_effect=self.read) as read, patch.object(installer.subprocess, "run") as run:
            installer.main(["--version", "0.4.0"])
            self.assertIn("/releases/tags/v0.4.0", read.call_args_list[0].args[0])
            args = run.call_args.args[0]
            self.assertEqual(args[:5], [installer.sys.executable,"-m","pip","install","--upgrade"])
            self.assertTrue(args[5].startswith("openfe-flex[catboost] @ file:"))

    def test_corrupted_wheel_stops_before_pip(self):
        good_digest = self.digest
        self.data = b"corrupted"
        with patch.object(installer, "read_url", side_effect=self.read), patch.object(installer.subprocess, "run") as run:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                installer.main([])
            run.assert_not_called()
        self.assertNotEqual(sha256(self.data).hexdigest(), good_digest)

    def test_missing_duplicate_or_redirected_asset_is_rejected(self):
        for bad in ("missing", "duplicate", "redirected", "draft"):
            self.setUp()
            if bad == "missing":
                self.release["assets"].pop()
            elif bad == "duplicate":
                self.release["assets"].append(dict(self.release["assets"][0]))
            elif bad == "redirected":
                self.release["assets"][0]["browser_download_url"] = "https://example.invalid/wheel"
            else:
                self.release["draft"] = True
            with self.subTest(bad=bad), patch.object(installer, "read_url", side_effect=self.read):
                with self.assertRaises(ValueError):
                    installer.resolve_release()

    def test_bad_version_and_ambiguous_checksum_are_rejected(self):
        with self.assertRaises(ValueError):
            installer.resolve_release("../../bad")
        self.manifest += self.manifest
        with patch.object(installer, "read_url", side_effect=self.read):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                installer.resolve_release()
