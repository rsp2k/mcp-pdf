"""
Shared utility functions for official mixins
"""

from typing import Optional, List


def parse_pages_parameter(pages: Optional[str]) -> Optional[List[int]]:
    """Parse pages parameter from string to list of 0-based page numbers

    Supports formats:
    - Single page: "5"
    - Comma-separated: "1,3,5"
    - Ranges: "1-10" or "11-30"
    - Mixed: "1,3-5,7,10-15"

    Args:
        pages: Page specification string (1-based page numbers)

    Returns:
        List of 0-based page indices, or None if pages is None
    """
    if not pages:
        return None

    try:
        result = []
        parts = pages.split(',')

        for part in parts:
            part = part.strip()

            # Handle range (e.g., "1-10" or "11-30")
            if '-' in part:
                range_parts = part.split('-')
                if len(range_parts) == 2:
                    start = int(range_parts[0].strip())
                    end = int(range_parts[1].strip())
                    # Convert 1-based to 0-based and create range
                    result.extend(range(start - 1, end))
                else:
                    return None
            # Handle single page
            else:
                result.append(int(part) - 1)

        return result
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# PyMuPDF document properties
#
# Three document facts were read from attribute names PyMuPDF does not have,
# via getattr-with-default or a bare except, so each returned a plausible
# constant forever: pdf_version was always the string "Unknown",
# is_linearized always False, and embedded_file_count always 0. Because the
# values looked reasonable nothing ever surfaced as an error, and downstream
# checks guarded on them (`if isinstance(pdf_version, (int, float))`,
# `if pdf_version < 1.4`) could never fire.
#
# Centralised here so the correct spelling lives in exactly one place.
# ---------------------------------------------------------------------------

def pdf_version(doc) -> Optional[str]:
    """The PDF spec version, e.g. "1.7", or None if unavailable.

    There is no `Document.pdf_version` attribute. The version is exposed
    through metadata's "format" key as a string like "PDF 1.3".
    """
    try:
        fmt = (doc.metadata or {}).get("format") or ""
    except Exception:
        return None
    # "PDF 1.3" -> "1.3"; tolerate anything unexpected by returning it whole.
    parts = fmt.split()
    if len(parts) == 2 and parts[0].upper() == "PDF":
        return parts[1]
    return fmt or None


def pdf_version_float(doc) -> Optional[float]:
    """pdf_version() as a float for comparisons, or None if unparseable."""
    v = pdf_version(doc)
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def is_linearized(doc) -> bool:
    """Whether the PDF is linearized ("fast web view").

    The attribute is `is_fast_webaccess`, not `is_linearized` and not
    `is_fast_web_view`. Both misspellings were in use.
    """
    try:
        value = getattr(doc, "is_fast_webaccess", False)
        return bool(value() if callable(value) else value)
    except Exception:
        return False


def embedded_file_count(doc) -> int:
    """Number of embedded file attachments.

    The method is `embfile_count()`, not `embedded_file_count()`.
    """
    try:
        value = getattr(doc, "embfile_count", None)
        if value is None:
            return 0
        return int(value() if callable(value) else value)
    except Exception:
        return 0
