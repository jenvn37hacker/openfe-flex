# Automatic GitHub Releases

The repository is `https://github.com/jenvn37hacker/openfe-flex`. GitHub Releases
host the universal wheel; PyPI publication is not configured.

## Trigger and version policy

- A push to `master` prepares a release automatically. A first release uses the
  source version (0.4.0 at introduction); later updates bump patch automatically.
- Other branches and pull requests only build/test the source snapshot.
- Workflow dispatch on master accepts `auto`, `patch`, `minor` or `major`.
  Explicit bumps can create a new version even when HEAD is already tagged.
- Version files must agree. If tags and source are inconsistent, CI stops rather
  than overwriting a tag. Pull the CI-generated version commit before a new push.

The preparation job creates a bot commit only if stamping changes the version,
then atomically pushes master and its tag. It never force-pushes. The original
trigger SHA is checked so queued stale pushes do not release newer unrelated
commits. The bot uses `GITHUB_TOKEN` and `[skip ci]` to avoid recursive workflows.

## Build/test/publication

Every downstream job checks out the prepared commit SHA, not a moving branch.
Build validates metadata, source contents and version alignment and generates
`SHA256SUMS`. Tests install the wheel into clean environments on:

- Ubuntu 22.04 and 24.04 x86-64;
- Windows x64;
- macOS Intel and Apple Silicon;
- each with Python 3.11, 3.12 and 3.13 (15 combinations).

Core-only, LightGBM-only and both-model installations are exercised. A failed job
prevents publication. The release job checks the actual tag commit and artifact
checksums again, uploads the wheel/source/checksums to a draft, verifies asset
digests and sizes, then makes the Release public and latest.

The tag/version commit may exist before tests finish; it is not evidence of a
published Release. Pip consumers use published Releases, not branch heads or tags.

## Reruns and interruptions

Rerunning a failed workflow uses the same version when the prepared commit/tag
already exist, including a retry of an explicit minor/major dispatch. A newer
unrelated master commit makes the old preparation skip instead of stamping it.
Retrying a partial upload resumes the unpublished draft, replacing only corrupt
or incomplete draft assets. Public assets are immutable: an inconsistent existing
public Release is an error, never silently replaced.

An already-published version at HEAD is skipped on an automatic retry. A new human
commit to master triggers the next patch release. If preparation lost an atomic
push race, rerun against the current master; no partial tag is pushed to origin.

## Install in another project

Download `scripts/install_latest.py` from the repository or copy it into the
consuming project. Run it with that project's Python environment:

```bash
.venv/bin/python install_latest.py --extras catboost
# Reproducible experiment:
.venv/bin/python install_latest.py --version 0.4.0 --extras catboost
# Generate an exact, hash-pinned pip requirement instead of installing:
.venv/bin/python install_latest.py --print-requirement > requirements-openfe.txt
.venv/bin/python -m pip install -r requirements-openfe.txt
```

Use `.venv\Scripts\python.exe` on Windows. The installer resolves the Release's
actual versioned wheel filename, verifies SHA-256 before calling pip and supports
`none`, `catboost`, `lightgbm` and `all` extras. It never falls back to an untested
branch if the Release is missing or incomplete. Public GitHub API rate limits
apply. A missing first Release means CI has not published successfully yet.

## Local checks

```bash
python -m unittest discover -s .github/tests -v
python .github/scripts/release.py plan
```

The tests use temporary Git repositories, a local bare remote and fake GitHub
responses. They do not push to GitHub, create real Releases or execute installer
pip operations. `plan` is read-only; `prepare` and `publish` are CI operations.
