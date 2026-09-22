"""XFA (Adobe LiveCycle / dynamic form) support.

Dynamic XFA PDFs are a hard wall for open-source rendering: the visible form
layout + all fields live in an XFA program that only Adobe's runtime executes.
PyMuPDF / pdfium / pikepdf / MuPDF all correctly see the single "Open in Adobe
Reader" placeholder page. What IS recoverable without rendering is the **XFA
template XML** (field names, captions, UI widget types), which is enough to
build a form *schema* and drive conversational filling pipelines.

This module:

1. Detects dynamic-XFA vs static-XFA vs not-XFA (``is_xfa_pdf``).
2. Extracts the XFA template + datasets streams via pypdf (no new deps).
3. Parses fields with stdlib ``xml.etree.ElementTree`` (namespace-agnostic).
4. Classifies fields against the **zipForm producer profile** (the most common
   XFA producer in the wild: Lone Wolf / zipForm Plus uses it for real-estate
   forms): shared semantic ``Global_Info-*`` fields, opaque positional
   ``p01tf022``-style codes, plumbing internals (``tbd7``, ``folder_id``, ...).
5. Canonicalizes shared names to a producer-neutral vocabulary so values
   collected for one form populate cross-form.

Design rule this module holds to, learned from a v2.3.0 review: **absence and
failure never share a representation.** "I read the file and it has no XFA" and
"I could not read the file" are different facts, so they get different return
shapes. Same for an empty template versus an unparseable one, and an unknown
unit versus a missing one. Every function here either answers or says it
could not, and never reports a failure as a confident negative.

The substantive parsing + classification logic is lifted from a working
prototype contributed by:

    Ryan Malloy <ryan@supported.systems> (via cdh-accessory-use-permit)

Originally proven on the Idaho REALTORS (IAR) standardized form library
(RE-05..RE-54), 456 fields extracted from RE-21, 17 identical canonical shared
fields verified across RE-21 and RE-16.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader
from pypdf.generic import ArrayObject


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back on any garbage."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# An XFA template is XML describing a form. Real ones run from a few KB to a
# couple of MB. The cap exists because expat's entity-amplification guard is
# relative to input size (roughly 100x past an 8 MiB threshold), so bounding
# the input is what actually bounds the expansion. Rejecting a large template
# is also much cheaper than the ~9s of CPU expat burns before refusing one.
MAX_XFA_TEMPLATE_BYTES = _env_int("MCP_PDF_MAX_XFA_TEMPLATE_BYTES", 16 * 1024 * 1024)

# Deepest <subform> nesting we will descend. Real forms sit in the low tens.
# Python's own recursion limit trips somewhere past ~1000 frames here, and a
# RecursionError escaping mid-walk gives the caller a shapeless error, so we
# stop first and say so.
MAX_XFA_DEPTH = _env_int("MCP_PDF_MAX_XFA_DEPTH", 100)

# Fields serialized inline before we truncate. Guards the MCP response budget:
# 456 fields (the real RE-21 case) is ~57 KB and fine, but a multi-form package
# with 50k fields serializes to ~6 MB, which no client can accept.
MAX_INLINE_FIELDS = _env_int("MCP_PDF_MAX_XFA_INLINE_FIELDS", 5000)

VALID_PROFILES = ("zipform", "generic")


class XfaError(Exception):
    """Base for XFA failures that callers should see rather than infer."""


class XfaTemplateParseError(XfaError):
    """The XFA template stream exists but its XML did not parse.

    Distinct from an empty template: this one means we found the bytes and
    could not read them, which is a different diagnosis and a different fix.
    """


class XfaTemplateTooLargeError(XfaError):
    """The XFA template exceeds MAX_XFA_TEMPLATE_BYTES."""


class XfaDepthExceededError(XfaError):
    """The template nests deeper than MAX_XFA_DEPTH."""


# ---------------------------------------------------------------------------
# XFA detection
# ---------------------------------------------------------------------------

def is_xfa_pdf(pdf_path: str) -> Dict[str, Any]:
    """Detect whether a PDF is XFA, and whether it's dynamic or static.

    Dynamic XFA is the failure case for every open-source renderer: its only
    rendered output is the "Open in Adobe Reader" placeholder page. Static XFA
    has a usable PDF representation alongside the XFA layer and renders fine.

    Returns ``{is_xfa, xfa_type, has_acroform, detection_failed, reason}``.

    Three outcomes, deliberately distinguishable:

    * **Read it, found XFA**  -> ``is_xfa=True``, ``xfa_type`` in
      ``{"dynamic", "static"}``, ``detection_failed=False``.
    * **Read it, no XFA**     -> ``is_xfa=False``, ``xfa_type=None``,
      ``detection_failed=False``.
    * **Could not read it**   -> ``is_xfa=None``, ``xfa_type=None``,
      ``detection_failed=True``, ``reason`` describing the failure.

    Callers MUST check ``detection_failed`` before trusting ``is_xfa``. The
    third state exists because reporting a damaged file as "not an XFA form"
    sends callers down the AcroForm path, where they hit a cryptic
    "document closed" instead of being told the file is unreadable. A
    truncated or mail-mangled form package is the common case, not an exotic
    one, so it gets a real answer.
    """
    def failed(reason: str) -> Dict[str, Any]:
        return {
            "is_xfa": None,
            "xfa_type": None,
            "has_acroform": None,
            "detection_failed": True,
            "reason": reason,
        }

    # Everything touching the file stays inside the guard, including the
    # trailer and AcroForm lookups. Those can raise on their own (an
    # encrypted PDF raises FileNotDecryptedError from the /Root access, not
    # from the constructor), and in v2.3.0 they sat outside it.
    try:
        reader = PdfReader(pdf_path)
        root = reader.trailer["/Root"]
        acro = root.get("/AcroForm")
        if acro is None:
            return {
                "is_xfa": False,
                "xfa_type": None,
                "has_acroform": False,
                "detection_failed": False,
                "reason": None,
            }

        xfa = acro.get_object().get("/XFA")
        if xfa is None:
            return {
                "is_xfa": False,
                "xfa_type": None,
                "has_acroform": True,
                "detection_failed": False,
                "reason": None,
            }

        # /NeedsRendering true in the catalog is the canonical dynamic-XFA
        # marker. Without it the XFA layer is supplementary to a renderable
        # PDF (static XFA).
        needs_rendering = bool(root.get("/NeedsRendering", False))
    except Exception as e:
        return failed(f"{type(e).__name__}: {e}")

    return {
        "is_xfa": True,
        "xfa_type": "dynamic" if needs_rendering else "static",
        "has_acroform": True,
        "detection_failed": False,
        "reason": None,
    }


# ---------------------------------------------------------------------------
# XFA packet extraction
# ---------------------------------------------------------------------------

def _local(tag: str) -> str:
    """Strip the XML namespace from a tag (``{ns}field`` -> ``field``)."""
    return tag.rsplit("}", 1)[-1]


def _part_name(obj: Any) -> str:
    """Normalize an XFA array key to a plain packet name.

    Spec-conformant producers use a text string (``template``), but a
    ``NameObject`` stringifies with a leading slash (``/template``), which
    would make every ``parts.get("template")`` miss and silently yield an
    empty schema. Strip it.
    """
    name = str(obj)
    return name[1:] if name.startswith("/") else name


def extract_xfa_parts(pdf_path: str) -> Dict[str, bytes]:
    """Return the XFA packet parts (template, datasets, config, ...) as bytes.

    The XFA entry is either a single stream or a flat array of name/stream
    pairs (``[preamble, <stream>, config, <stream>, template, <stream>, ...]``).

    Insertion order is preserved, so ``list(result)`` reflects the order the
    packets appear in the file rather than an alphabetical rewrite of it.
    """
    reader = PdfReader(pdf_path)
    acro = reader.trailer["/Root"].get("/AcroForm")
    if acro is None:
        return {}
    xfa = acro.get_object().get("/XFA")
    if xfa is None:
        return {}
    xfa = xfa.get_object()

    parts: Dict[str, bytes] = {}
    if isinstance(xfa, ArrayObject):
        items = [x.get_object() for x in xfa]
        for i in range(0, len(items) - 1, 2):
            name = _part_name(items[i])
            try:
                parts[name] = items[i + 1].get_data()
            except Exception:
                # The paired value was not a readable stream. Record the
                # packet as present-but-unreadable rather than dropping it,
                # so a caller seeing an empty schema can tell the difference
                # between "no template packet" and "template packet broke".
                parts[name] = b""
    else:
        try:
            parts["template"] = xfa.get_data()
        except Exception:
            parts["template"] = b""
    return parts


# ---------------------------------------------------------------------------
# XFA field parsing
# ---------------------------------------------------------------------------

@dataclass
class XfaField:
    """A field as it appears in the XFA template, before classification."""

    name: str
    caption: Optional[str] = None
    ui_type: Optional[str] = None   # textEdit, checkButton, choiceList, dateTimeEdit, signature
    is_excl_group: bool = False     # <exclGroup> is XFA's radio-group construct
    bbox: Optional[Dict[str, Any]] = None   # best-effort design-time geometry in pt


def _first_caption(field_el: ET.Element) -> Optional[str]:
    """The field's own caption text (caption/value/text), if any.

    Read only from direct children so nested fields don't bleed their captions
    upward. Keeps scanning past a caption whose text is empty.
    """
    for child in field_el:
        if _local(child.tag) != "caption":
            continue
        texts = [t.text or "" for t in child.iter() if _local(t.tag) == "text"]
        caption = " ".join(t.strip() for t in texts if t.strip())
        if caption:
            return caption
    return None


def _ui_type(field_el: ET.Element) -> Optional[str]:
    """The field's UI widget kind (textEdit / checkButton / choiceList / ...)."""
    for child in field_el:
        if _local(child.tag) == "ui":
            for widget in child:
                return _local(widget.tag)
    return None


# Unit conversions to points (PDF user-space units). "mp" is millipoints,
# which is legal XFA and was missing in v2.3.0, so "72000mp" came back as
# 72000.0 instead of 72.0.
_UNIT_TO_PT = {
    "pt": 1.0,
    "mp": 0.001,
    "in": 72.0,
    "mm": 72.0 / 25.4,
    "cm": 72.0 / 2.54,
    "pc": 12.0,
}

# Guards against a template declaring an absurd dimension that then serializes
# into JSON as 7.2e+22. A US Letter page is 792pt tall; 1e6 is already absurd.
_MAX_DIM_PT = 1_000_000.0

# Accepts "1.5in", ".5in", "+1in", "36MM", "10 pt", "72000mp".
_DIM_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([a-zA-Z]+)?\s*$")


def _parse_dim(value: Optional[str]) -> Optional[float]:
    """Parse an XFA dimension like ``"1.5in"`` or ``"36mm"`` to points.

    Returns None when the value is absent, malformed, carries a unit we do not
    know, or is implausibly large. An unknown unit returns None rather than
    silently assuming points, because v2.3.0's ``.get(unit, 1.0)`` fallback
    turned ``"1.5inch"`` into 1.5pt, a 72x error delivered with no warning.
    """
    if not value:
        return None
    m = _DIM_RE.match(value)
    if not m:
        return None
    try:
        num = float(m.group(1))
    except ValueError:
        return None
    unit = (m.group(2) or "pt").lower()
    factor = _UNIT_TO_PT.get(unit)
    if factor is None:
        return None
    result = num * factor
    if abs(result) > _MAX_DIM_PT:
        return None
    return result


def _bbox_from_field(field_el: ET.Element,
                     parent_offset: Tuple[float, float],
                     page: Optional[int]) -> Optional[Dict[str, Any]]:
    """Compute best-effort design-time bbox in points.

    XFA <field> elements carry their own (x, y, w, h) relative to the
    enclosing <subform>; parent offsets accumulate as we walk down, and reset
    at each <pageArea> so coordinates stay page-relative.

    Coordinates are DESIGN-TIME and not authoritative for dynamic XFA, whose
    growable and optional subforms reflow at render time. Treat as a hint.

    ``x_pt`` / ``y_pt`` are present only when the field actually declared x/y.
    v2.3.0 emitted the bare parent offset for a field that declared only w/h,
    which is a concrete-looking answer where "unknown" is the truth.
    """
    fx = _parse_dim(field_el.get("x"))
    fy = _parse_dim(field_el.get("y"))
    fw = _parse_dim(field_el.get("w"))
    fh = _parse_dim(field_el.get("h"))
    if fx is None and fy is None and fw is None and fh is None:
        return None

    px, py = parent_offset
    bbox: Dict[str, Any] = {"page": page, "origin": "top-left"}
    if fx is not None:
        bbox["x_pt"] = round(px + fx, 2)
    if fy is not None:
        bbox["y_pt"] = round(py + fy, 2)
    if fw is not None:
        bbox["w_pt"] = round(fw, 2)
    if fh is not None:
        bbox["h_pt"] = round(fh, 2)
    return bbox


@dataclass
class _WalkState:
    """Mutable bookkeeping threaded through the recursive walk."""

    fields: List[XfaField] = field(default_factory=list)
    skipped_unnamed: int = 0
    page: Optional[int] = None
    max_depth_seen: int = 0


def _iter_fields(el: ET.Element,
                 state: _WalkState,
                 parent_offset: Tuple[float, float] = (0.0, 0.0),
                 with_bbox: bool = False,
                 depth: int = 0) -> None:
    """Walk the XFA tree depth-first, appending to ``state``.

    Tracks parent <subform> x/y offsets so each field's bbox accumulates
    relative to the current page origin, and counts <pageArea> boundaries so
    a bbox can report which page it belongs to.

    Raises XfaDepthExceededError past MAX_XFA_DEPTH rather than letting a
    RecursionError escape mid-walk, because a RecursionError reaches the
    caller as a shapeless error with none of the contract keys on it.
    """
    if depth > MAX_XFA_DEPTH:
        raise XfaDepthExceededError(
            f"XFA template nests deeper than {MAX_XFA_DEPTH} levels"
        )
    state.max_depth_seen = max(state.max_depth_seen, depth)

    tag = _local(el.tag)

    # A new pageArea restarts the coordinate space and advances the page
    # counter. Without this, offsets accumulate monotonically across pages and
    # y_pt becomes uninterpretable on anything multi-page.
    if tag == "pageArea":
        state.page = 1 if state.page is None else state.page + 1
        parent_offset = (0.0, 0.0)

    # Accumulate offset when descending through a <subform> with x/y.
    if tag == "subform":
        sx = _parse_dim(el.get("x"))
        sy = _parse_dim(el.get("y"))
        # An unparseable offset is not zero. Treating it as zero silently
        # shifts every descendant, so we leave the offset alone and let the
        # affected fields simply not report coordinates they cannot support.
        parent_offset = (
            parent_offset[0] + (sx if sx is not None else 0.0),
            parent_offset[1] + (sy if sy is not None else 0.0),
        )

    if tag in ("field", "exclGroup"):
        name = el.get("name")
        if not name:
            # Unnamed fields are legal XFA but unusable as a schema key.
            # Count them so the caller can see something was skipped.
            state.skipped_unnamed += 1
            return

        is_excl = tag == "exclGroup"
        ui = _ui_type(el)
        if is_excl and ui is None:
            # An exclGroup's <ui> lives on its member <field> elements, not on
            # the group. Look one level down so a radio group is typed as a
            # radio group instead of falling through to the text default.
            for child in el:
                if _local(child.tag) == "field":
                    ui = _ui_type(child)
                    if ui:
                        break

        state.fields.append(XfaField(
            name=name,
            caption=_first_caption(el),
            ui_type=ui,
            is_excl_group=is_excl,
            bbox=_bbox_from_field(el, parent_offset, state.page) if with_bbox else None,
        ))
        # XFA fields can technically nest, but the prototype's policy was to
        # not recurse INTO field elements (captions don't bleed up). Match it.
        return

    for child in el:
        _iter_fields(child, state, parent_offset, with_bbox, depth + 1)


def parse_fields(template_xml: bytes,
                 with_bbox: bool = False) -> Tuple[List[XfaField], Dict[str, Any]]:
    """Parse the XFA <template> into fields plus walk diagnostics.

    Returns ``(fields, diagnostics)`` where diagnostics carries
    ``skipped_unnamed`` and ``max_depth_seen``.

    Raises:
        XfaTemplateTooLargeError: template exceeds MAX_XFA_TEMPLATE_BYTES.
        XfaTemplateParseError: the bytes are present but are not parseable XML.
        XfaDepthExceededError: nesting exceeds MAX_XFA_DEPTH.

    An empty ``template_xml`` returns an empty field list, which is the
    genuine "nothing here" answer. Unparseable bytes raise, because in v2.3.0
    both cases returned ``[]`` and the caller was told the form had no fields
    when in fact the template was corrupt.
    """
    if not template_xml:
        return [], {"skipped_unnamed": 0, "max_depth_seen": 0}

    if len(template_xml) > MAX_XFA_TEMPLATE_BYTES:
        raise XfaTemplateTooLargeError(
            f"XFA template is {len(template_xml)} bytes, over the "
            f"{MAX_XFA_TEMPLATE_BYTES} byte limit. Raise "
            f"MCP_PDF_MAX_XFA_TEMPLATE_BYTES to process it anyway."
        )

    try:
        root = ET.fromstring(template_xml)
    except ET.ParseError as e:
        # Covers malformed XML and expat's entity-amplification refusal. Both
        # mean "found the template, could not read it", which is the fact the
        # caller needs.
        raise XfaTemplateParseError(f"XFA template XML did not parse: {e}") from e

    state = _WalkState()
    _iter_fields(root, state, (0.0, 0.0), with_bbox=with_bbox)
    return state.fields, {
        "skipped_unnamed": state.skipped_unnamed,
        "max_depth_seen": state.max_depth_seen,
    }


def fields_from_pdf(pdf_path: str,
                    with_bbox: bool = False) -> Tuple[List[XfaField], Dict[str, Any]]:
    """Convenience: extract the XFA template from a PDF and parse its fields."""
    parts = extract_xfa_parts(pdf_path)
    return parse_fields(parts.get("template") or b"", with_bbox=with_bbox)


# ---------------------------------------------------------------------------
# Field classification (producer profiles)
# ---------------------------------------------------------------------------

# XFA ui widget -> the portable vocabulary shared with extract_form_data:
# text / checkbox / radio / dropdown / date / signature, plus button and
# unknown for the edges. Anything absent from this map becomes "unknown",
# NOT "text": presenting a guess as a fact is how a barcode or image picker
# ends up rendered as a free-text box that silently accepts bad input.
_UI_TO_TYPE = {
    "textEdit": "text",
    "numericEdit": "text",
    "passwordEdit": "text",
    "checkButton": "checkbox",
    "choiceList": "dropdown",
    "dateTimeEdit": "date",
    "signature": "signature",
    "button": "button",
}


@dataclass
class XfaProfile:
    """A producer-specific naming profile for classifying XFA fields.

    Different XFA producers (Lone Wolf zipForm, Adobe LiveCycle, custom) use
    different conventions for positional codes and plumbing internals. The
    profile parameterizes the classification heuristics so callers can extend
    or replace them per producer.

    ``plumbing_exact`` entries are compared case-insensitively and must be
    stored lowercased. Use ``_resolve_profile`` or lowercase them yourself.
    """

    name: str = "custom"
    # Field-name prefix that marks shared cross-form semantic fields.
    shared_prefix: str = "Global_Info-"
    # Exact names to drop as plumbing. Stored lowercased; matched lowercased.
    plumbing_exact: set = field(default_factory=set)
    # Regex patterns dropped as plumbing. Matched with .search (substring).
    plumbing_patterns: List[re.Pattern] = field(default_factory=list)
    # Regex patterns identifying opaque positional codes. Matched with .match
    # (anchored at the start).
    positional_patterns: List[re.Pattern] = field(default_factory=list)


# zipForm profile: Lone Wolf / zipForm Plus producer conventions, verified
# against the IAR real-estate form library.
_ZIPFORM_PROFILE = XfaProfile(
    name="zipform",
    shared_prefix="Global_Info-",
    plumbing_exact={
        "folder_id", "file_id", "security_key", "forms_package_id",
        "last_clicked_field", "preparer_info", "clauses", "serial", "serial_2",
    },
    plumbing_patterns=[
        re.compile(r"^tbd\d+$", re.I),
        re.compile(r"^serial", re.I),
        re.compile(r"invisibind", re.I),
    ],
    positional_patterns=[
        # p<page><type><index>, e.g. p01tf022 (text), p01cb001 (checkbox)
        re.compile(r"^p\d+[a-z]{2,4}\d+$", re.I),
    ],
)

# Generic profile: only the shared-prefix convention, no producer-specific
# plumbing/positional patterns. Callers add their own via extra_* parameters.
_GENERIC_PROFILE = XfaProfile(name="generic", shared_prefix="Global_Info-")

_BASE_PROFILES = {
    "zipform": _ZIPFORM_PROFILE,
    "generic": _GENERIC_PROFILE,
}


def _resolve_profile(profile: str,
                     extra_plumbing_exact: Optional[List[str]] = None,
                     extra_plumbing_patterns: Optional[List[str]] = None,
                     extra_positional_patterns: Optional[List[str]] = None) -> XfaProfile:
    """Compose an XfaProfile from the named base plus caller extensions.

    The profile name is matched case-insensitively, so ``"zipForm"`` (how the
    product is actually spelled, including in this repo's own docs) resolves
    to the zipform profile instead of silently falling through to generic.

    Raises:
        ValueError: the name is not a known profile. v2.3.0 fell back to
            generic for any unrecognized string, which quietly promoted
            producer internals like ``Global_Info-Invisibind-Test`` into the
            shared cross-form vocabulary: the exact invariant the profile
            exists to protect.
    """
    key = (profile or "").strip().lower()
    base = _BASE_PROFILES.get(key)
    if base is None:
        raise ValueError(
            f"Unknown XFA profile {profile!r}. Valid profiles: "
            f"{', '.join(VALID_PROFILES)}"
        )

    # Lowercase on insert to match the lowercased lookup in _is_plumbing.
    # v2.3.0 inserted caller casing verbatim, so any mixed-case entry was
    # unreachable and the parameter silently did nothing.
    plumbing_exact = set(base.plumbing_exact)
    plumbing_exact.update(n.lower() for n in (extra_plumbing_exact or []))

    plumbing_patterns = list(base.plumbing_patterns)
    for pat in (extra_plumbing_patterns or []):
        plumbing_patterns.append(re.compile(pat, re.I))

    positional_patterns = list(base.positional_patterns)
    for pat in (extra_positional_patterns or []):
        positional_patterns.append(re.compile(pat, re.I))

    extended = any([
        extra_plumbing_exact, extra_plumbing_patterns, extra_positional_patterns
    ])
    return XfaProfile(
        name=f"{base.name}+custom" if extended else base.name,
        shared_prefix=base.shared_prefix,
        plumbing_exact=plumbing_exact,
        plumbing_patterns=plumbing_patterns,
        positional_patterns=positional_patterns,
    )


def _is_plumbing(name: str, profile: XfaProfile) -> bool:
    if name.lower() in profile.plumbing_exact:
        return True
    return any(p.search(name) for p in profile.plumbing_patterns)


def _is_positional(name: str, profile: XfaProfile) -> bool:
    return any(p.match(name) for p in profile.positional_patterns)


def canonicalize(name: str, profile: XfaProfile, separator: str = "_") -> str:
    """Canonicalize a (shared-prefixed) name to a producer-neutral identifier.

    ``Global_Info-Seller-Broker-Entity-Name`` -> ``seller_broker_entity_name``,
    or with ``separator="."`` -> ``seller.broker.entity.name``.
    """
    core = name[len(profile.shared_prefix):] if name.startswith(profile.shared_prefix) else name
    core = core.replace("-", "_")
    # camelCase -> snake (insert _ at lowercase|digit -> uppercase boundary)
    core = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", core)
    core = re.sub(r"[^A-Za-z0-9_]+", "_", core)
    snake = re.sub(r"_+", "_", core).strip("_").lower()
    return snake.replace("_", separator) if separator != "_" else snake


def _field_type_for(ui_type: Optional[str], is_excl_group: bool = False) -> str:
    """Map an XFA UI widget to the portable vocabulary.

    ``exclGroup`` is XFA's mutually-exclusive choice construct, so it maps to
    ``radio`` regardless of the member widgets' own type. In v2.3.0 it fell
    through to ``text``, which rendered a two-option radio group as a
    free-text box that would accept any value.
    """
    if is_excl_group:
        return "radio"
    return _UI_TO_TYPE.get(ui_type or "", "unknown")


@dataclass
class ClassifiedField:
    original: str
    field_type: str
    category: str            # "shared" | "positional" | "other"
    canonical_name: Optional[str] = None   # only set when category == "shared"
    label: Optional[str] = None            # <caption> if present, else derived for shared
    ui_type: Optional[str] = None
    bbox: Optional[Dict[str, Any]] = None


def classify_fields(fields: List[XfaField],
                    profile: XfaProfile,
                    separator: str = "_") -> Tuple[List[ClassifiedField], List[str]]:
    """Filter plumbing and split the rest into shared / positional / other.

    Plumbing is checked FIRST, before the shared prefix, so a producer-internal
    field that happens to carry the shared prefix (zipForm's
    ``Global_Info-Invisibind-*``) is dropped rather than promoted into the
    cross-form vocabulary. That ordering is load-bearing.

    Returns ``(classified, dropped_plumbing_names)``.
    """
    classified: List[ClassifiedField] = []
    dropped: List[str] = []

    for f in fields:
        if _is_plumbing(f.name, profile):
            dropped.append(f.name)
            continue

        if f.name.startswith(profile.shared_prefix):
            canonical = canonicalize(f.name, profile, separator)
            derived = re.sub(r"[_.\-]+", " ", canonical).strip().title()
            label = f.caption or derived
            category = "shared"
        elif _is_positional(f.name, profile):
            canonical = None
            label = f.caption  # opaque positional fields rarely have captions
            category = "positional"
        else:
            canonical = None
            label = f.caption
            category = "other"

        classified.append(ClassifiedField(
            original=f.name,
            field_type=_field_type_for(f.ui_type, f.is_excl_group),
            category=category,
            canonical_name=canonical,
            label=label,
            ui_type=f.ui_type,
            bbox=f.bbox,
        ))

    return classified, dropped


def find_canonical_collisions(classified: List[ClassifiedField]) -> Dict[str, List[str]]:
    """Shared fields whose distinct XFA names canonicalize to the same key.

    The canonical name is the cross-form join key: collect a value once and it
    populates that slot on every form. So two different fields mapping to one
    canonical name means one human answer written into two semantically
    different slots. ``canonicalize`` normalizes case plus ``-`` and ``_``, so
    ``Global_Info-Buyer-Name`` and ``Global_Info-Buyer_Name`` converge.

    Returns ``{canonical_name: [original, ...]}`` for canonical names claimed
    by more than one field. Empty dict is the healthy case.
    """
    by_canonical: Dict[str, List[str]] = defaultdict(list)
    for cf in classified:
        if cf.category == "shared" and cf.canonical_name:
            by_canonical[cf.canonical_name].append(cf.original)
    return {k: v for k, v in by_canonical.items() if len(v) > 1}


# ---------------------------------------------------------------------------
# Top-level entry point used by the MCP tool
# ---------------------------------------------------------------------------

def _error_response(message: str,
                    detection: Optional[Dict[str, Any]] = None,
                    **extra: Any) -> Dict[str, Any]:
    """Build a failure response that still carries the contract keys.

    ``is_xfa`` / ``xfa_type`` are on every response, success or failure, so a
    caller doing ``result["is_xfa"]`` never hits a KeyError precisely when the
    call failed.
    """
    response: Dict[str, Any] = {
        "success": False,
        "is_xfa": detection.get("is_xfa") if detection else None,
        "xfa_type": detection.get("xfa_type") if detection else None,
        "error": message,
    }
    response.update(extra)
    return response


def extract_xfa_schema(pdf_path: str,
                       profile: str = "zipform",
                       extra_plumbing_exact: Optional[List[str]] = None,
                       extra_plumbing_patterns: Optional[List[str]] = None,
                       extra_positional_patterns: Optional[List[str]] = None,
                       canonical_separator: str = "_",
                       include_design_time_bbox: bool = False,
                       max_inline_fields: Optional[int] = None) -> Dict[str, Any]:
    """Full extraction pipeline used by the MCP tool.

    Every response carries ``success``, ``is_xfa`` and ``xfa_type``. On
    success it also carries:

    - ``xfa_parts`` in file order, and ``field_count`` (named fields found in
      the template, before plumbing is dropped)
    - ``fields``: every non-plumbing field, each with ``original`` (the
      round-trip key for filling), ``field_type``, ``category``, and
      ``canonical_name`` only where ``category == "shared"``
    - ``categories``: counts per category. Note **four** categories, not
      three: ``shared``, ``positional``, ``other``, and separately
      ``plumbing_fields_dropped``. ``other`` is the default bucket for a name
      matching neither the shared prefix nor a positional pattern, so on a
      non-zipForm producer it usually holds the majority of fields. Callers
      branching on category must handle it.
    - ``shared_fields``: the cross-form vocabulary, ``{original: canonical}``
    - ``canonical_collisions``: ``{canonical: [original, ...]}`` where two
      shared fields claim one canonical name. Empty is healthy.
    - ``profile_used``, ``canonical_separator``, ``warnings``
    """
    warnings: List[str] = []

    detection = is_xfa_pdf(pdf_path)

    # Read failure is its own outcome. Reporting it as "not XFA" sends the
    # caller to the AcroForm path, where a damaged form package produces a
    # cryptic "document closed" instead of "this file is unreadable".
    if detection.get("detection_failed"):
        return _error_response(
            f"Could not read the PDF well enough to determine whether it is "
            f"XFA: {detection.get('reason')}",
            detection,
            detection_failed=True,
        )

    if not detection["is_xfa"]:
        return _error_response(
            "Not an XFA PDF. Use extract_form_data for AcroForm fields.",
            detection,
            has_acroform=detection.get("has_acroform"),
        )

    try:
        resolved_profile = _resolve_profile(
            profile,
            extra_plumbing_exact,
            extra_plumbing_patterns,
            extra_positional_patterns,
        )
    except ValueError as e:
        return _error_response(str(e), detection)

    try:
        parts = extract_xfa_parts(pdf_path)
    except Exception as e:
        return _error_response(
            f"Could not read the XFA packets: {type(e).__name__}: {e}", detection
        )

    template = parts.get("template") or b""
    if not template:
        return _error_response(
            "No readable XFA template packet. The packet is absent or its "
            "stream could not be decoded, so no field schema is recoverable.",
            detection,
            xfa_parts=list(parts),
        )

    try:
        raw_fields, diagnostics = parse_fields(
            template, with_bbox=include_design_time_bbox
        )
    except XfaError as e:
        # Covers unparseable XML, oversize templates, and excessive nesting.
        # Each is a real diagnosis, and none of them is "this form has no
        # fields", which is what v2.3.0 reported for all three.
        return _error_response(
            str(e), detection, xfa_parts=list(parts), error_kind=type(e).__name__
        )

    classified, dropped = classify_fields(
        raw_fields, resolved_profile, canonical_separator
    )

    shared_map = {
        cf.original: cf.canonical_name
        for cf in classified
        if cf.category == "shared"
    }

    collisions = find_canonical_collisions(classified)
    if collisions:
        warnings.append(
            f"{len(collisions)} canonical name(s) are claimed by more than one "
            f"shared field. The canonical name is the cross-form join key, so "
            f"filling one would write the same value into different slots. See "
            f"canonical_collisions."
        )

    if diagnostics["skipped_unnamed"]:
        warnings.append(
            f"{diagnostics['skipped_unnamed']} field element(s) had no name "
            f"attribute and were skipped; they cannot be used as schema keys."
        )

    if include_design_time_bbox:
        warnings.append(
            "Design-time bbox included. Coordinates come from the template, "
            "not from a render, so growable or optional subforms will move "
            "them. Geometry is page-relative with a top-left origin, which "
            "is the opposite convention to extract_form_data's bottom-up "
            "coordinates. Treat as a hint, not a source of truth."
        )

    categories = {"shared": 0, "positional": 0, "other": 0}
    for cf in classified:
        categories[cf.category] += 1

    if not classified:
        warnings.append(
            "The template parsed but yielded no non-plumbing fields. Either it "
            "genuinely declares none, or every field matched this profile's "
            "plumbing rules (check plumbing_fields_dropped)."
        )
    elif categories["other"] > categories["shared"] + categories["positional"]:
        warnings.append(
            f"{categories['other']} of {len(classified)} fields landed in the "
            f"'other' category, meaning they matched neither the shared prefix "
            f"nor a positional pattern for profile '{resolved_profile.name}'. "
            f"If this form is not zipForm-produced, supply "
            f"extra_positional_patterns for its naming convention."
        )

    # Bound the serialized response. The repo's house convention is
    # file-first output for anything unbounded; this is the interim guard so a
    # 50k-field package cannot blow the MCP response budget silently.
    limit = max_inline_fields if max_inline_fields is not None else MAX_INLINE_FIELDS
    truncated = len(classified) > limit
    emitted = classified[:limit] if truncated else classified
    if truncated:
        warnings.append(
            f"Response truncated to {limit} of {len(classified)} fields. Raise "
            f"max_inline_fields (or MCP_PDF_MAX_XFA_INLINE_FIELDS) to see more. "
            f"Counts in 'categories' and 'shared_fields' cover ALL fields, not "
            f"just the emitted ones."
        )

    return {
        "success": True,
        "is_xfa": True,
        "xfa_type": detection["xfa_type"],
        "xfa_parts": list(parts),
        "field_count": len(raw_fields),
        "field_count_after_classification": len(classified),
        "fields": [_classified_to_dict(cf) for cf in emitted],
        "fields_truncated": truncated,
        "categories": categories,
        "shared_fields": shared_map,
        "canonical_collisions": collisions,
        "plumbing_fields_dropped": dropped,
        "profile_used": resolved_profile.name,
        "canonical_separator": canonical_separator,
        "warnings": warnings,
    }


def _classified_to_dict(cf: ClassifiedField) -> Dict[str, Any]:
    """Serialize ClassifiedField, omitting optional keys that are None."""
    out: Dict[str, Any] = {
        "original": cf.original,
        "field_type": cf.field_type,
        "category": cf.category,
    }
    if cf.canonical_name is not None:
        out["canonical_name"] = cf.canonical_name
    if cf.label:
        out["label"] = cf.label
    if cf.ui_type:
        out["ui_type"] = cf.ui_type
    if cf.bbox:
        out["bbox"] = cf.bbox
    return out
