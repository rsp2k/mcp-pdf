"""XFA (Adobe LiveCycle / dynamic form) support.

Dynamic XFA PDFs are a hard wall for open-source rendering: the visible form
layout + all fields live in an XFA program that only Adobe's runtime executes.
PyMuPDF / pdfium / pikepdf / MuPDF all correctly see the single "Open in Adobe
Reader" placeholder page. What IS recoverable without rendering is the **XFA
template XML** — field names, captions, UI widget types — which is enough to
build a form *schema* and drive conversational filling pipelines.

This module:

1. Detects dynamic-XFA vs static-XFA vs not-XFA (``is_xfa_pdf``).
2. Extracts the XFA template + datasets streams via pypdf (no new deps).
3. Parses fields with stdlib ``xml.etree.ElementTree`` (namespace-agnostic).
4. Classifies fields against the **zipForm producer profile** (the most common
   XFA producer in the wild — Lone Wolf / zipForm Plus uses it for real-estate
   forms): shared semantic ``Global_Info-*`` fields, opaque positional
   ``p01tf022``-style codes, plumbing internals (``tbd7``, ``folder_id``, ...).
5. Canonicalizes shared names to a producer-neutral vocabulary so values
   collected for one form populate cross-form.

The substantive parsing + classification logic is lifted from a working
prototype contributed by:

    Ryan Malloy <ryan@supported.systems> (via cdh-accessory-use-permit)

Originally proven on the Idaho REALTORS (IAR) standardized form library
(RE-05..RE-54), 456 fields extracted from RE-21, 17 identical canonical shared
fields verified across RE-21 ∩ RE-16. Coordinated via the agent-thread
``docs/agent-threads/xfa-form-support/``.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader
from pypdf.generic import ArrayObject


# ---------------------------------------------------------------------------
# XFA detection
# ---------------------------------------------------------------------------

def is_xfa_pdf(pdf_path: str) -> Dict[str, Any]:
    """Detect whether a PDF is XFA, and whether it's dynamic or static.

    Dynamic XFA is the failure case for every open-source renderer — its only
    rendered output is the "Open in Adobe Reader" placeholder page. Static XFA
    has a usable PDF representation alongside the XFA layer and can be rendered
    normally.

    Returns ``{is_xfa, xfa_type, has_acroform}`` where ``xfa_type`` is one of
    ``"dynamic"``, ``"static"``, or ``None`` (not XFA).
    """
    try:
        reader = PdfReader(pdf_path)
    except Exception:
        return {"is_xfa": False, "xfa_type": None, "has_acroform": False}

    root = reader.trailer["/Root"]
    acro = root.get("/AcroForm")
    if acro is None:
        return {"is_xfa": False, "xfa_type": None, "has_acroform": False}

    acro_obj = acro.get_object()
    xfa = acro_obj.get("/XFA")
    if xfa is None:
        return {"is_xfa": False, "xfa_type": None, "has_acroform": True}

    # /NeedsRendering true in the catalog is the canonical dynamic-XFA marker.
    # Without it, the XFA layer is supplementary to a renderable PDF (static XFA).
    needs_rendering = bool(root.get("/NeedsRendering", False))

    return {
        "is_xfa": True,
        "xfa_type": "dynamic" if needs_rendering else "static",
        "has_acroform": True,
    }


# ---------------------------------------------------------------------------
# XFA packet extraction
# ---------------------------------------------------------------------------

def _local(tag: str) -> str:
    """Strip the XML namespace from a tag (``{ns}field`` -> ``field``)."""
    return tag.rsplit("}", 1)[-1]


def extract_xfa_parts(pdf_path: str) -> Dict[str, bytes]:
    """Return the XFA packet parts (template, datasets, config, ...) as bytes.

    The XFA entry is either a single stream or a flat array of name/stream
    pairs (``[preamble, <stream>, config, <stream>, template, <stream>, ...]``).
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
            name = str(items[i])
            try:
                parts[name] = items[i + 1].get_data()
            except Exception:
                continue
    else:
        try:
            parts["template"] = xfa.get_data()
        except Exception:
            pass
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
    bbox: Optional[Dict[str, float]] = None   # best-effort design-time geometry in pt


def _first_caption(field_el: ET.Element) -> Optional[str]:
    """The field's own caption text (caption/value/text), if any.

    Read only from direct children so nested fields don't bleed their captions
    upward.
    """
    for child in field_el:
        if _local(child.tag) != "caption":
            continue
        texts = [t.text or "" for t in child.iter() if _local(t.tag) == "text"]
        caption = " ".join(t.strip() for t in texts if t.strip())
        return caption or None
    return None


def _ui_type(field_el: ET.Element) -> Optional[str]:
    """The field's UI widget kind (textEdit / checkButton / choiceList / ...)."""
    for child in field_el:
        if _local(child.tag) == "ui":
            for widget in child:
                return _local(widget.tag)
    return None


# Unit conversions to points (PDF user-space units).
_UNIT_TO_PT = {
    "pt": 1.0,
    "in": 72.0,
    "mm": 72.0 / 25.4,
    "cm": 72.0 / 2.54,
}


def _parse_dim(value: Optional[str]) -> Optional[float]:
    """Parse an XFA dimension like ``"1.5in"`` or ``"36mm"`` to points."""
    if not value:
        return None
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*([a-z]+)?\s*$", value)
    if not m:
        return None
    num = float(m.group(1))
    unit = (m.group(2) or "pt").lower()
    return num * _UNIT_TO_PT.get(unit, 1.0)


def _bbox_from_field(field_el: ET.Element,
                     parent_offset: Tuple[float, float]) -> Optional[Dict[str, float]]:
    """Compute best-effort design-time bbox in points.

    XFA <field> elements carry their own (x, y, w, h) attributes relative to
    the enclosing <subform>; we accumulate parent offsets as we walk down.
    Coordinates are DESIGN-TIME and not authoritative for dynamic XFA
    (growable/optional subforms reflow them at render time).
    """
    fx = _parse_dim(field_el.get("x"))
    fy = _parse_dim(field_el.get("y"))
    fw = _parse_dim(field_el.get("w"))
    fh = _parse_dim(field_el.get("h"))
    if fx is None and fy is None and fw is None and fh is None:
        return None
    px, py = parent_offset
    return {
        "x_pt": round((px + (fx or 0.0)), 2),
        "y_pt": round((py + (fy or 0.0)), 2),
        "w_pt": round(fw, 2) if fw is not None else None,
        "h_pt": round(fh, 2) if fh is not None else None,
        "note": "Design-time geometry. Reflows in dynamic XFA — not authoritative.",
    }


def _iter_fields(el: ET.Element,
                 parent_offset: Tuple[float, float] = (0.0, 0.0),
                 with_bbox: bool = False) -> List[XfaField]:
    """Walk the XFA tree depth-first, emitting XfaField per field/exclGroup.

    Tracks parent <subform> x/y offsets so each field's bbox is accumulated
    relative to the page origin (best-effort — see _bbox_from_field).
    """
    out: List[XfaField] = []
    tag = _local(el.tag)

    # Accumulate offset when descending through a <subform> with x/y.
    if tag == "subform":
        sx = _parse_dim(el.get("x")) or 0.0
        sy = _parse_dim(el.get("y")) or 0.0
        parent_offset = (parent_offset[0] + sx, parent_offset[1] + sy)

    if tag in ("field", "exclGroup"):
        name = el.get("name")
        if name:
            out.append(XfaField(
                name=name,
                caption=_first_caption(el),
                ui_type=_ui_type(el),
                bbox=_bbox_from_field(el, parent_offset) if with_bbox else None,
            ))
        # XFA fields can technically nest, but the prototype's policy was to
        # not recurse INTO field elements (captions don't bleed up). Match that.
        return out

    for child in el:
        out.extend(_iter_fields(child, parent_offset, with_bbox))
    return out


def parse_fields(template_xml: bytes,
                 with_bbox: bool = False) -> List[XfaField]:
    """Extract every field/exclGroup from the XFA <template> with name + caption.

    Set ``with_bbox=True`` for best-effort design-time geometry on each field.
    """
    try:
        root = ET.fromstring(template_xml)
    except ET.ParseError:
        return []
    return _iter_fields(root, (0.0, 0.0), with_bbox=with_bbox)


def fields_from_pdf(pdf_path: str, with_bbox: bool = False) -> List[XfaField]:
    """Convenience: extract the XFA template from a PDF and parse its fields."""
    parts = extract_xfa_parts(pdf_path)
    template = parts.get("template") or b""
    return parse_fields(template, with_bbox=with_bbox)


# ---------------------------------------------------------------------------
# Field classification (producer profiles)
# ---------------------------------------------------------------------------

# XFA ui widget -> portable six-term vocabulary shared with extract_form_data.
_UI_TO_TYPE = {
    "textEdit": "text",
    "numericEdit": "text",
    "checkButton": "checkbox",
    "choiceList": "dropdown",
    "dateTimeEdit": "date",
    "signature": "signature",
}


@dataclass
class XfaProfile:
    """A producer-specific naming profile for classifying XFA fields.

    Different XFA producers (Lone Wolf zipForm, Adobe LiveCycle, custom) use
    different conventions for positional codes and plumbing internals. The
    profile parameterizes the classification heuristics so callers can extend
    or replace them per producer.
    """

    name: str = "generic"
    # Field-name prefix that marks shared cross-form semantic fields.
    shared_prefix: str = "Global_Info-"
    # Exact names to drop as plumbing.
    plumbing_exact: set = field(default_factory=set)
    # Regex patterns (full or partial match) to drop as plumbing.
    plumbing_patterns: List[re.Pattern] = field(default_factory=list)
    # Regex patterns identifying opaque positional codes.
    positional_patterns: List[re.Pattern] = field(default_factory=list)


# zipForm profile — Lone Wolf / zipForm Plus producer conventions, verified
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

# Generic profile — only the shared-prefix convention, no producer-specific
# plumbing/positional patterns. Callers add their own via extra_* parameters.
_GENERIC_PROFILE = XfaProfile(name="generic", shared_prefix="Global_Info-")


def _resolve_profile(profile: str,
                     extra_plumbing_exact: Optional[List[str]],
                     extra_plumbing_patterns: Optional[List[str]],
                     extra_positional_patterns: Optional[List[str]]) -> XfaProfile:
    """Compose an XfaProfile from the named base + caller extensions."""
    base = _ZIPFORM_PROFILE if profile == "zipform" else _GENERIC_PROFILE

    plumbing_exact = set(base.plumbing_exact)
    plumbing_exact.update(extra_plumbing_exact or [])

    plumbing_patterns = list(base.plumbing_patterns)
    for pat in (extra_plumbing_patterns or []):
        plumbing_patterns.append(re.compile(pat, re.I))

    positional_patterns = list(base.positional_patterns)
    for pat in (extra_positional_patterns or []):
        positional_patterns.append(re.compile(pat, re.I))

    return XfaProfile(
        name=f"{base.name}+custom" if any([
            extra_plumbing_exact, extra_plumbing_patterns, extra_positional_patterns
        ]) else base.name,
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

    ``Global_Info-Seller-Broker-Entity-Name`` -> ``seller_broker_entity_name``
    (with separator=".") -> ``seller.broker.entity.name``.
    """
    core = name[len(profile.shared_prefix):] if name.startswith(profile.shared_prefix) else name
    core = core.replace("-", "_")
    # camelCase -> snake (insert _ at lowercase|digit -> uppercase boundary)
    core = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", core)
    core = re.sub(r"[^A-Za-z0-9_]+", "_", core)
    snake = re.sub(r"_+", "_", core).strip("_").lower()
    return snake.replace("_", separator) if separator != "_" else snake


def _field_type_for(ui_type: Optional[str]) -> str:
    return _UI_TO_TYPE.get(ui_type or "", "text")


@dataclass
class ClassifiedField:
    original: str
    field_type: str
    category: str            # "shared" | "positional" | "other"
    canonical_name: Optional[str] = None   # only set when category == "shared"
    label: Optional[str] = None            # human label (derived for shared, harvested from <caption> if present)
    ui_type: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None


def classify_fields(fields: List[XfaField],
                    profile: XfaProfile,
                    separator: str = "_") -> Tuple[List[ClassifiedField], List[str]]:
    """Filter plumbing and split the rest into shared / positional / other.

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
            label = f.caption or canonical.replace("_", " ").replace(".", " ").title()
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
            field_type=_field_type_for(f.ui_type),
            category=category,
            canonical_name=canonical,
            label=label,
            ui_type=f.ui_type,
            bbox=f.bbox,
        ))

    return classified, dropped


# ---------------------------------------------------------------------------
# Top-level entry point used by the MCP tool
# ---------------------------------------------------------------------------

def extract_xfa_schema(pdf_path: str,
                       profile: str = "zipform",
                       extra_plumbing_exact: Optional[List[str]] = None,
                       extra_plumbing_patterns: Optional[List[str]] = None,
                       extra_positional_patterns: Optional[List[str]] = None,
                       canonical_separator: str = "_",
                       include_design_time_bbox: bool = False) -> Dict[str, Any]:
    """Full extraction pipeline used by the MCP tool.

    Returns a structured dict matching the response shape agreed in
    ``docs/agent-threads/xfa-form-support/004-*.md``:

    - ``is_xfa``, ``xfa_type``, ``xfa_parts`` — provenance / detection
    - ``field_count`` — raw count BEFORE plumbing drop
    - ``fields`` — every non-plumbing field with original + classification
    - ``shared_fields`` — the cross-form vocabulary (original → canonical)
    - ``plumbing_fields_dropped`` — names that were dropped, for auditability
    - ``profile_used`` — which profile (incl. +custom suffix if extended)
    - ``warnings`` — anything the caller should know
    """
    detection = is_xfa_pdf(pdf_path)
    warnings: List[str] = []

    if not detection["is_xfa"]:
        return {
            "is_xfa": False,
            "xfa_type": None,
            "error": "Not an XFA PDF. Use extract_form_data for AcroForm fields.",
        }

    parts = extract_xfa_parts(pdf_path)
    template = parts.get("template") or b""

    resolved_profile = _resolve_profile(
        profile, extra_plumbing_exact, extra_plumbing_patterns, extra_positional_patterns,
    )

    raw_fields = parse_fields(template, with_bbox=include_design_time_bbox)
    classified, dropped = classify_fields(raw_fields, resolved_profile, canonical_separator)

    shared_map = {
        cf.original: cf.canonical_name
        for cf in classified
        if cf.category == "shared"
    }

    if include_design_time_bbox:
        warnings.append(
            "Design-time bbox included. Coordinates can drift from Adobe's "
            "rendered positions on dynamic XFA forms (growable/optional subforms)."
        )
    if not classified:
        warnings.append(
            "No non-plumbing fields recovered. The XFA template may be empty, "
            "use a producer profile we don't recognize, or be a static-XFA "
            "template that ships fields via AcroForm instead."
        )

    return {
        "is_xfa": True,
        "xfa_type": detection["xfa_type"],
        "xfa_parts": sorted(parts.keys()),
        "field_count": len(raw_fields),
        "field_count_after_classification": len(classified),
        "fields": [_classified_to_dict(cf) for cf in classified],
        "shared_fields": shared_map,
        "plumbing_fields_dropped": dropped,
        "profile_used": resolved_profile.name,
        "canonical_separator": canonical_separator,
        "warnings": warnings,
    }


def _classified_to_dict(cf: ClassifiedField) -> Dict[str, Any]:
    """Serialize ClassifiedField. Drops None-valued optional keys for cleanliness."""
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
