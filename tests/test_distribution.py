"""Release checks reject wrong versions, incomplete artifacts and corruption."""
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SCRIPT=Path(__file__).resolve().parents[1]/".github/scripts/distribution.py"
spec=importlib.util.spec_from_file_location("distribution_checks",SCRIPT)
distribution=importlib.util.module_from_spec(spec)
spec.loader.exec_module(distribution)


class DistributionTests(unittest.TestCase):
    def fixtures(self,directory,metadata_version=None,license=True):
        version=distribution.project_version()
        wheel=directory/f"openfe_flex-{version}-py3-none-any.whl"
        with zipfile.ZipFile(wheel,"w") as archive:
            archive.writestr(f"openfe_flex-{version}.dist-info/METADATA",
                             f"Name: openfe-flex\nVersion: {metadata_version or version}\n")
            archive.writestr("openfe_flex/engine.py","")
            if license:
                archive.writestr(f"openfe_flex-{version}.dist-info/licenses/LICENSE","MIT")
        with tarfile.open(directory/f"openfe_flex-{version}.tar.gz","w:gz") as archive:
            for name in ("LICENSE","UPSTREAM.json","docs/algorithm_alignment.md",
                         "tests/test_paper_core.py","examples/classification.py"):
                member=tarfile.TarInfo(f"openfe_flex-{version}/{name}")
                member.size=1
                archive.addfile(member,io.BytesIO(b"x"))

    def test_valid_artifacts_and_checksum_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);self.fixtures(directory)
            with patch("sys.argv",[str(SCRIPT),"prepare","--directory",folder]):
                distribution.main()
            with patch("sys.argv",[str(SCRIPT),"verify","--directory",folder]):
                distribution.main()
            archive=next(directory.glob("*.tar.gz"))
            with archive.open("ab") as handle:
                handle.write(b"tampered")
            with patch("sys.argv",[str(SCRIPT),"verify","--directory",folder]):
                with self.assertRaisesRegex(ValueError,"checksums"):
                    distribution.main()

    def test_reject_wrong_wheel_version_and_missing_license(self):
        for metadata_version,license in (("0.0.0",True),(None,False)):
            with tempfile.TemporaryDirectory() as folder:
                self.fixtures(Path(folder),metadata_version,license)
                with self.assertRaises(ValueError):
                    distribution.artifacts(Path(folder),distribution.project_version())

    def test_reject_stale_wheel_and_wrong_tag(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);self.fixtures(directory)
            with patch("sys.argv",[str(SCRIPT),"prepare","--directory",folder,"--tag=v0.0.0"]):
                with self.assertRaises(SystemExit):
                    distribution.main()
            (directory/"stale.whl").write_bytes(b"old")
            with self.assertRaises(ValueError):
                distribution.artifacts(directory,distribution.project_version())
