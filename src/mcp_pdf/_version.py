"""Single source of truth for the package version.

Its own module on purpose. Computing this in ``__init__.py`` and importing it
from ``server.py`` does technically work, because ``__init__`` binds
``__version__`` before it imports ``server``, so the partially-initialized
package already carries the attribute. But that only holds while those two
statements stay in that order, which is a landmine for whoever tidies the
imports later. A leaf module has no cycle to get the order wrong in.

The version itself comes from installed package metadata, which hatchling
fills from ``pyproject.toml``. Hardcoding it is what let three copies drift
apart: ``__init__.py`` claimed 0.1.0, ``server_info`` reported 2.0.12, and
``main()``'s fallback said 2.1.0, while the package was actually on 2.3.1.

Versioning is CalVer, ``YYYY.MM.DD``, with a PEP 440 post-release segment for
same-day fixes (``2026.09.21.1``). The scheme communicates *when* the package
was last tested against the PDF libraries and system tools it wraps, which is
the question you actually have when a PDF behaves strangely.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("mcp-pdf")
except PackageNotFoundError:  # source tree with no install
    __version__ = "0+unknown"

__all__ = ["__version__"]
