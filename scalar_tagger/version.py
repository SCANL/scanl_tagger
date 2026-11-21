"""Single source of truth for the SCALAR software version.

SCALAR follows Semantic Versioning 2.0.0 (https://semver.org). See the
"Versioning" section of README.md for what counts as a major, minor, or
patch change, and record every change in CHANGELOG.md.
"""

import re

__version__ = "3.0.0"

_SEMVER = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?(?:\+(?P<build>[0-9A-Za-z.-]+))?$"
)

_match = _SEMVER.match(__version__)
if _match is None:
    raise ValueError(f"__version__ is not a valid semantic version: {__version__!r}")

__version_info__ = (
    int(_match["major"]),
    int(_match["minor"]),
    int(_match["patch"]),
)
