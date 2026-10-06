import os
import re
import subprocess
import sys

from conftest import REPO_ROOT
from version import __version__, __version_info__, _SEMVER


def test_version_is_semver():
    match = _SEMVER.match(__version__)
    assert match is not None
    assert __version_info__ == (int(match["major"]), int(match["minor"]), int(match["patch"]))


def test_semver_pattern_rejects_invalid_versions():
    for bad in ["3", "3.0", "v3.0.0", "03.0.0", "3.0.0.1", "3.0.x"]:
        assert _SEMVER.match(bad) is None, bad
    for good in ["0.1.0", "3.0.0", "3.1.0-rc.1", "3.1.0+build.5"]:
        assert _SEMVER.match(good) is not None, good


def test_changelog_has_entry_for_current_version():
    with open(os.path.join(REPO_ROOT, "CHANGELOG.md")) as f:
        changelog = f.read()
    assert re.search(rf"^## \[?{re.escape(__version__)}\]?\b", changelog, re.MULTILINE), (
        f"CHANGELOG.md has no section for {__version__}"
    )


def test_main_version_flag_needs_no_mode():
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "main"), "--version"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert f"SCALAR tagger {__version__}" in result.stdout
