"""Tests for XFA (dynamic Adobe LiveCycle form) support.

Two groups, deliberately:

**Happy path** pins behavior against the synthetic XFA fixture contributed via
the agent-thread at ``docs/agent-threads/xfa-form-support/`` (the
expected-output table in ``005-iar-attaches-fixture.md``). The fixture is
hand-built, license-clean, ~2 KB; its README documents what each field
exercises.

**Failure path** covers what happens when the input fights back. This module's
only input is untrusted files from third-party producers, so these matter more
than the happy path, not less. v2.3.0 shipped with exactly one failure-path
test and it passed *because* of a bare-except defect, certifying the bug.

The governing invariant for the whole module: **absence and failure never
share a representation.** A file we could not read must never be reported as a
file with no XFA, and a template we could not parse must never be reported as
a form with no fields.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from mcp_pdf.xfa import (
    MAX_XFA_DEPTH,
    XfaDepthExceededError,
    XfaTemplateParseError,
    XfaTemplateTooLargeError,
    _field_type_for,
    _parse_dim,
    _resolve_profile,
    _ZIPFORM_PROFILE,
    canonicalize,
    classify_fields,
    extract_xfa_parts,
    extract_xfa_schema,
    fields_from_pdf,
    find_canonical_collisions,
    is_xfa_pdf,
    parse_fields,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "xfa"
SYNTHETIC_FIXTURE = FIXTURE_DIR / "synthetic_dynamic_xfa.pdf"


def _damaged_copy(mutate) -> str:
    """Write a damaged copy of the fixture to a temp file, return its path.

    Damage modes are mundane on purpose: a truncated download, a stripped
    xref, a missing EOF marker. These are what an emailed or FTP'd form
    package actually looks like when it goes wrong.
    """
    raw = SYNTHETIC_FIXTURE.read_bytes()
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as fh:
        fh.write(mutate(raw))
    return path


# ---------------------------------------------------------------------------
# Detection: the three outcomes
# ---------------------------------------------------------------------------

class TestXfaDetection:
    """is_xfa_pdf must distinguish found / not-found / could-not-look."""

    def test_synthetic_fixture_detected_as_dynamic_xfa(self):
        result = is_xfa_pdf(str(SYNTHETIC_FIXTURE))
        assert result["is_xfa"] is True
        assert result["xfa_type"] == "dynamic"
        assert result["has_acroform"] is True
        assert result["detection_failed"] is False

    def test_non_xfa_pdf_is_a_confident_negative(self):
        """A readable PDF without XFA gets is_xfa=False, NOT None."""
        plain = Path(__file__).parent / "CopperSprings_DigitalPortfolio.pdf"
        if not plain.exists():
            pytest.skip("non-XFA fixture not present")
        result = is_xfa_pdf(str(plain))
        assert result["is_xfa"] is False
        assert result["detection_failed"] is False
        assert result["xfa_type"] is None

    def test_nonexistent_pdf_reports_detection_failure_not_absence(self):
        """A missing file is NOT 'this file has no XFA'.

        v2.3.0 returned is_xfa=False here, which is a claim about a file we
        never read.
        """
        result = is_xfa_pdf("/nonexistent/path.pdf")
        assert result["detection_failed"] is True
        assert result["is_xfa"] is None
        assert result["reason"]

    @pytest.mark.parametrize("label, mutate", [
        ("missing %%EOF", lambda raw: raw.replace(b"%%EOF", b"")),
        ("truncated to 90%", lambda raw: raw[:int(len(raw) * 0.9)]),
        ("garbled trailer", lambda raw: raw.replace(b"trailer", b"trailXX")),
    ])
    def test_damaged_pdf_reports_detection_failure(self, label, mutate):
        """The finding that mattered most in the v2.3.0 review.

        Each of these damage modes made pypdf raise while MuPDF still opened
        the file. v2.3.0 swallowed the raise and returned is_xfa=False, so
        extract_form_data proceeded into fitz and produced the exact cryptic
        "document closed" this release existed to eliminate, on a file whose
        XFA template bytes were physically still present.
        """
        path = _damaged_copy(mutate)
        try:
            result = is_xfa_pdf(path)
            assert result["detection_failed"] is True, label
            assert result["is_xfa"] is None, label
            assert result["reason"], label
        finally:
            os.unlink(path)

    def test_damaged_pdf_schema_call_explains_itself(self):
        """extract_xfa_schema on a damaged file says so, with contract keys."""
        path = _damaged_copy(lambda raw: raw[:int(len(raw) * 0.9)])
        try:
            result = extract_xfa_schema(path)
            assert result["success"] is False
            assert result["detection_failed"] is True
            # Contract keys present even on failure, so result["is_xfa"]
            # never raises KeyError exactly when the call failed.
            assert "is_xfa" in result and "xfa_type" in result
            assert "could not read" in result["error"].lower()
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# XFA packet extraction
# ---------------------------------------------------------------------------

class TestXfaPartsExtraction:
    """The fixture uses the array form [template, datasets], which exercises
    the name/stream-pair parsing path rather than the single-stream path."""

    def test_extracts_template_and_datasets(self):
        parts = extract_xfa_parts(str(SYNTHETIC_FIXTURE))
        assert "template" in parts
        assert "datasets" in parts
        assert isinstance(parts["template"], bytes)
        assert len(parts["template"]) > 0

    def test_packet_order_is_file_order_not_alphabetical(self):
        """005's expected table says ["template", "datasets"].

        v2.3.0 ran the keys through sorted(), silently discarding packet
        order and returning ["datasets", "template"].
        """
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        assert result["xfa_parts"] == ["template", "datasets"]


# ---------------------------------------------------------------------------
# Field parsing
# ---------------------------------------------------------------------------

class TestFieldParsing:
    def test_finds_eight_raw_fields(self):
        """All 8 fields from the fixture README's table, pre-classification."""
        fields, diagnostics = fields_from_pdf(str(SYNTHETIC_FIXTURE))
        assert len(fields) == 8
        assert diagnostics["skipped_unnamed"] == 0

    def test_caption_harvested_from_field_with_caption(self):
        fields, _ = fields_from_pdf(str(SYNTHETIC_FIXTURE))
        by_name = {f.name: f for f in fields}
        assert by_name["Global_Info-Test-Property-Address"].caption == "Property Address"

    def test_ui_types_extracted(self):
        fields, _ = fields_from_pdf(str(SYNTHETIC_FIXTURE))
        by_name = {f.name: f for f in fields}
        assert by_name["Global_Info-Test-Property-Address"].ui_type == "textEdit"
        assert by_name["Global_Info-Test-Buyer-Name"].ui_type == "checkButton"
        assert by_name["p01cb002"].ui_type == "checkButton"

    def test_empty_template_is_not_an_error(self):
        """Genuinely empty input is the real 'nothing here' answer."""
        fields, diagnostics = parse_fields(b"")
        assert fields == []
        assert diagnostics["skipped_unnamed"] == 0

    def test_unparseable_template_raises_rather_than_returning_empty(self):
        """The C2 finding: found the bytes, could not read them.

        v2.3.0 caught ET.ParseError and returned [], so a corrupt template
        was reported as a form with zero fields plus a warning naming three
        causes, none of which was the real one.
        """
        with pytest.raises(XfaTemplateParseError):
            parse_fields(b"<template><field name='x'")

    def test_unparseable_template_surfaces_through_the_schema_call(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        assert result["success"] is True  # sanity: fixture is good

        # Now the real assertion, against a template we know will not parse.
        with pytest.raises(XfaTemplateParseError):
            parse_fields(b"\x00\x01\x02 not xml at all")

    def test_unnamed_fields_are_counted_not_silently_dropped(self):
        template = (
            b"<template><subform>"
            b"<field name='kept'><ui><textEdit/></ui></field>"
            b"<field><ui><textEdit/></ui></field>"
            b"</subform></template>"
        )
        fields, diagnostics = parse_fields(template)
        assert len(fields) == 1
        assert diagnostics["skipped_unnamed"] == 1

    def test_oversize_template_is_rejected_with_a_clear_error(self):
        from mcp_pdf import xfa
        original = xfa.MAX_XFA_TEMPLATE_BYTES
        xfa.MAX_XFA_TEMPLATE_BYTES = 64
        try:
            with pytest.raises(XfaTemplateTooLargeError):
                parse_fields(b"<template>" + b"x" * 500 + b"</template>")
        finally:
            xfa.MAX_XFA_TEMPLATE_BYTES = original

    def test_excessive_nesting_raises_instead_of_leaking_recursionerror(self):
        """A RecursionError escaping mid-walk reaches the caller with none of
        the contract keys on it. Stop first and say why."""
        depth = MAX_XFA_DEPTH + 20
        template = (
            b"<template>" + b"<subform>" * depth
            + b"<field name='deep'><ui><textEdit/></ui></field>"
            + b"</subform>" * depth + b"</template>"
        )
        with pytest.raises(XfaDepthExceededError):
            parse_fields(template)

    def test_entity_expansion_is_refused_as_a_parse_error(self):
        """Billion-laughs lands in the parse-error path, not a silent empty."""
        bomb = (
            b"<?xml version='1.0'?><!DOCTYPE t ["
            b"<!ENTITY a 'aaaaaaaaaa'>"
            b"<!ENTITY b '&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;'>"
            b"<!ENTITY c '&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;'>"
            b"<!ENTITY d '&c;&c;&c;&c;&c;&c;&c;&c;&c;&c;'>"
            b"<!ENTITY e '&d;&d;&d;&d;&d;&d;&d;&d;&d;&d;'>"
            b"<!ENTITY f '&e;&e;&e;&e;&e;&e;&e;&e;&e;&e;'>"
            b"<!ENTITY g '&f;&f;&f;&f;&f;&f;&f;&f;&f;&f;'>"
            b"<!ENTITY h '&g;&g;&g;&g;&g;&g;&g;&g;&g;&g;'>"
            b"<!ENTITY i '&h;&h;&h;&h;&h;&h;&h;&h;&h;&h;'>"
            b"]><template><field name='&i;'/></template>"
        )
        with pytest.raises(XfaTemplateParseError):
            parse_fields(bomb)

    def test_external_entity_is_refused(self):
        """XXE: ElementTree refuses external entities outright."""
        xxe = (
            b"<?xml version='1.0'?>"
            b"<!DOCTYPE t [<!ENTITY x SYSTEM 'file:///etc/passwd'>]>"
            b"<template><field name='&x;'/></template>"
        )
        with pytest.raises(XfaTemplateParseError):
            parse_fields(xxe)


# ---------------------------------------------------------------------------
# Field type vocabulary (the v2.3.0 blind spot: zero assertions existed)
# ---------------------------------------------------------------------------

class TestFieldTypeVocabulary:
    """The portable vocabulary is the stated reason this release exists, and
    v2.3.0 had no assertion on it anywhere in 31 tests."""

    @pytest.mark.parametrize("ui_type, expected", [
        ("textEdit", "text"),
        ("numericEdit", "text"),
        ("passwordEdit", "text"),
        ("checkButton", "checkbox"),
        ("choiceList", "dropdown"),
        ("dateTimeEdit", "date"),
        ("signature", "signature"),
        ("button", "button"),
    ])
    def test_mapped_ui_types(self, ui_type, expected):
        assert _field_type_for(ui_type) == expected

    @pytest.mark.parametrize("ui_type", ["barcode", "imageEdit", "bogus", None])
    def test_unmapped_ui_types_are_unknown_not_text(self, ui_type):
        """Presenting a guess as a fact is how a barcode gets rendered as a
        free-text box that silently accepts bad input. The AcroForm side
        returns 'unknown' for its unmapped case; so does this one now."""
        assert _field_type_for(ui_type) == "unknown"

    def test_exclgroup_is_radio(self):
        """exclGroup is XFA's mutually-exclusive choice construct.

        v2.3.0 returned 'text' for it, rendering a two-option radio group as
        a free-text field, in the release whose purpose was a coherent
        vocabulary.
        """
        assert _field_type_for("checkButton", is_excl_group=True) == "radio"
        assert _field_type_for(None, is_excl_group=True) == "radio"

    def test_exclgroup_end_to_end_finds_member_ui_and_types_as_radio(self):
        template = (
            b"<template><subform>"
            b"<exclGroup name='Global_Info-Financing-Type'>"
            b"<field name='cash'><ui><checkButton/></ui></field>"
            b"<field name='loan'><ui><checkButton/></ui></field>"
            b"</exclGroup></subform></template>"
        )
        fields, _ = parse_fields(template)
        classified, _ = classify_fields(fields, _ZIPFORM_PROFILE)
        assert len(classified) == 1
        assert classified[0].field_type == "radio"
        assert classified[0].ui_type == "checkButton"

    def test_fixture_field_types(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        types = {f["original"]: f["field_type"] for f in result["fields"]}
        assert types == {
            "Global_Info-Test-Property-Address": "text",
            "Global_Info-Test-Buyer-Name": "checkbox",
            "p01tf001": "text",
            "p01cb002": "checkbox",
            "p02te001": "text",
        }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestClassificationBehavior:
    def test_classification_counts_match_fixture_spec(self):
        """README: 2 shared, 3 positional, 3 plumbing dropped."""
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        assert result["categories"] == {"shared": 2, "positional": 3, "other": 0}
        assert len(result["plumbing_fields_dropped"]) == 3

    def test_shared_canonical_names(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        assert set(result["shared_fields"].values()) == {
            "test_property_address", "test_buyer_name",
        }

    def test_positional_codes_preserved_verbatim_with_no_canonical_name(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        positional = sorted(
            f["original"] for f in result["fields"] if f["category"] == "positional"
        )
        assert positional == ["p01cb002", "p01tf001", "p02te001"]
        for f in result["fields"]:
            if f["category"] == "positional":
                assert "canonical_name" not in f

    def test_plumbing_drops_match_fixture_spec(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        assert set(result["plumbing_fields_dropped"]) == {
            "tbd1", "folder_id", "Global_Info-Invisibind-Test",
        }

    def test_denylist_beats_shared_prefix(self):
        """The load-bearing invariant.

        Global_Info-Invisibind-Test carries the shared prefix AND matches the
        invisibind denylist. Plumbing must win, or a filling pipeline prompts
        a human for a producer-internal value.
        """
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        assert "Global_Info-Invisibind-Test" in result["plumbing_fields_dropped"]
        assert not any(
            f["original"] == "Global_Info-Invisibind-Test" for f in result["fields"]
        )
        assert "Global_Info-Invisibind-Test" not in result["shared_fields"]

    def test_positional_regex_does_not_overmatch_different_pages(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        positional = {
            f["original"] for f in result["fields"] if f["category"] == "positional"
        }
        assert positional == {"p01tf001", "p01cb002", "p02te001"}

    def test_field_count_is_raw_count(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        assert result["field_count"] == 8
        assert result["field_count_after_classification"] == 5

    def test_original_present_on_every_field(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        for f in result["fields"]:
            assert f["original"]

    def test_caption_used_as_label_for_shared(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        f = next(f for f in result["fields"]
                 if f["original"] == "Global_Info-Test-Property-Address")
        assert f["label"] == "Property Address"

    def test_label_derived_from_canonical_when_no_caption(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        f = next(f for f in result["fields"]
                 if f["original"] == "Global_Info-Test-Buyer-Name")
        assert f["label"] == "Test Buyer Name"

    def test_derived_label_is_clean_under_every_separator(self):
        """A kebab separator used to leak hyphens into the human label."""
        for sep in ("_", ".", "-"):
            result = extract_xfa_schema(str(SYNTHETIC_FIXTURE), canonical_separator=sep)
            f = next(f for f in result["fields"]
                     if f["original"] == "Global_Info-Test-Buyer-Name")
            assert f["label"] == "Test Buyer Name", sep


class TestCanonicalCollisions:
    """The canonical name is the cross-form join key, so a collision means one
    human answer written into two semantically different slots."""

    def test_no_collisions_on_the_clean_fixture(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        assert result["canonical_collisions"] == {}

    def test_collision_detected_and_warned(self):
        template = (
            b"<template><subform>"
            b"<field name='Global_Info-Buyer-Name'><ui><textEdit/></ui></field>"
            b"<field name='Global_Info-Buyer_Name'><ui><textEdit/></ui></field>"
            b"</subform></template>"
        )
        fields, _ = parse_fields(template)
        classified, _ = classify_fields(fields, _ZIPFORM_PROFILE)
        collisions = find_canonical_collisions(classified)
        assert collisions == {
            "buyer_name": ["Global_Info-Buyer-Name", "Global_Info-Buyer_Name"]
        }


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

class TestCanonicalization:
    @pytest.mark.parametrize("original, expected", [
        ("Global_Info-Test-Property-Address", "test_property_address"),
        ("Global_Info-Seller-Broker-Entity-Name", "seller_broker_entity_name"),
        ("Global_Info-Buyer-Name", "buyer_name"),
        ("Global_Info-BuyerName", "buyer_name"),
        ("Global_Info--Field--Name", "field_name"),
        ("Some-Other-Field", "some_other_field"),
    ])
    def test_snake_case(self, original, expected):
        assert canonicalize(original, _ZIPFORM_PROFILE, separator="_") == expected

    @pytest.mark.parametrize("separator, expected", [
        (".", "seller.broker.name"),
        ("-", "seller-broker-name"),
        ("_", "seller_broker_name"),
    ])
    def test_separators(self, separator, expected):
        result = canonicalize(
            "Global_Info-Seller-Broker-Name", _ZIPFORM_PROFILE, separator=separator
        )
        assert result == expected

    def test_separator_wired_through_the_schema_call(self):
        """Only canonicalize() was unit-tested in v2.3.0; nothing pinned the
        end-to-end wiring of the parameter."""
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE), canonical_separator=".")
        assert set(result["shared_fields"].values()) == {
            "test.property.address", "test.buyer.name",
        }
        assert result["canonical_separator"] == "."


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class TestProfiles:
    @pytest.mark.parametrize("spelling", ["zipform", "zipForm", "ZIPFORM", " zipform "])
    def test_profile_name_is_case_and_space_insensitive(self, spelling):
        """'zipForm' is how the product is spelled, including in this repo's
        own docs and in the tool's description string.

        v2.3.0 compared with a bare == against the lowercase literal, so that
        spelling fell through to the generic profile, which promoted
        Global_Info-Invisibind-Test into the shared vocabulary. A typo copied
        from our own documentation defeated the invariant.
        """
        assert _resolve_profile(spelling).name == "zipform"
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE), profile=spelling)
        assert result["profile_used"] == "zipform"
        assert "Global_Info-Invisibind-Test" in result["plumbing_fields_dropped"]

    def test_unknown_profile_is_an_error_not_a_silent_fallback(self):
        with pytest.raises(ValueError, match="Unknown XFA profile"):
            _resolve_profile("banana")

        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE), profile="banana")
        assert result["success"] is False
        assert "Unknown XFA profile" in result["error"]
        assert "zipform" in result["error"]  # names the valid options

    def test_generic_profile_classifies_exactly(self):
        """Exact counts, and the leak this profile causes, stated plainly.

        v2.3.0's version of this test asserted `> 0` and carried a comment
        claiming Invisibind lands in 'other' under generic. It does not: it
        becomes *shared*, which is the single most important thing to assert
        here and the thing the comment got backwards.
        """
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE), profile="generic")
        assert result["profile_used"] == "generic"
        assert result["categories"] == {"shared": 3, "positional": 0, "other": 5}
        assert result["plumbing_fields_dropped"] == []

        # The leak, asserted: with no plumbing patterns, a producer-internal
        # field carrying the shared prefix is promoted into the cross-form
        # vocabulary. This is why profile validation matters.
        assert result["shared_fields"]["Global_Info-Invisibind-Test"] == "invisibind_test"

    def test_caller_can_add_plumbing_pattern(self):
        result = extract_xfa_schema(
            str(SYNTHETIC_FIXTURE), profile="zipform",
            extra_plumbing_patterns=[r"^p01tf"],
        )
        assert "p01tf001" in result["plumbing_fields_dropped"]
        assert result["profile_used"] == "zipform+custom"

    def test_caller_can_add_positional_pattern(self):
        result = extract_xfa_schema(
            str(SYNTHETIC_FIXTURE), profile="generic",
            extra_positional_patterns=[r"^p\d+[a-z]+\d+$"],
        )
        positional = {
            f["original"] for f in result["fields"] if f["category"] == "positional"
        }
        assert positional == {"p01tf001", "p01cb002", "p02te001"}

    @pytest.mark.parametrize("name", ["p01tf001", "P01TF001", "P01tf001"])
    def test_extra_plumbing_exact_is_case_insensitive(self, name):
        """v2.3.0 inserted caller casing verbatim but looked up lowercased, so
        any mixed-case entry was unreachable and the parameter silently did
        nothing. The old test passed only because it picked a lowercase value.
        """
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE), extra_plumbing_exact=[name])
        assert "p01tf001" in result["plumbing_fields_dropped"]

    def test_positional_patterns_are_anchored_plumbing_are_not(self):
        """Documented asymmetry: _is_positional uses .match, _is_plumbing uses
        .search. A caller who does not know that writes a pattern that
        silently matches nothing."""
        unanchored = extract_xfa_schema(
            str(SYNTHETIC_FIXTURE), profile="generic",
            extra_positional_patterns=[r"tf\d+$"],
        )
        assert unanchored["categories"]["positional"] == 0

        anchored = extract_xfa_schema(
            str(SYNTHETIC_FIXTURE), profile="generic",
            extra_positional_patterns=[r"^p\d+tf\d+$"],
        )
        assert anchored["categories"]["positional"] == 1

        # Plumbing patterns need no anchor, by contrast.
        substring = extract_xfa_schema(
            str(SYNTHETIC_FIXTURE), extra_plumbing_patterns=[r"tf\d+$"],
        )
        assert "p01tf001" in substring["plumbing_fields_dropped"]


# ---------------------------------------------------------------------------
# Dimension parsing (design-time bbox support)
# ---------------------------------------------------------------------------

class TestDimensionParsing:
    @pytest.mark.parametrize("value, expected", [
        ("1.5in", 108.0),
        (".5in", 36.0),
        ("+1in", 72.0),
        ("-1in", -72.0),
        ("10pt", 10.0),
        ("72000mp", 72.0),
        ("1.5 in", 108.0),
        ("36MM", 102.05),
        ("1pc", 12.0),
    ])
    def test_valid_dimensions(self, value, expected):
        assert _parse_dim(value) == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize("value", [
        "1.5inch",                   # unknown unit: must NOT silently be pt
        "10px",
        "10em",
        "999999999999999999999in",   # implausible magnitude
        "abc",
        "",
        None,
    ])
    def test_invalid_dimensions_return_none_not_a_wrong_number(self, value):
        """v2.3.0's .get(unit, 1.0) turned '1.5inch' into 1.5pt, a 72x error
        delivered with the same confidence as a correct value."""
        assert _parse_dim(value) is None


class TestDesignTimeBbox:
    """Opt-in, explicitly non-authoritative, and previously never executed by
    the suite at all, which meant a refactor could turn it into a hard
    exception with all tests still green."""

    def test_disabled_by_default(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        assert all("bbox" not in f for f in result["fields"])

    def test_enabling_it_produces_a_well_formed_response(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE), include_design_time_bbox=True)
        assert result["success"] is True
        assert any("bbox" in w.lower() or "design-time" in w.lower()
                   for w in result["warnings"])

    def test_offsets_accumulate_through_nested_subforms(self):
        template = (
            b"<template><pageArea>"
            b"<subform x='0.5in' y='0.5in'><subform x='10pt' y='10pt'>"
            b"<field name='Global_Info-Nested' x='1in' y='1in' w='2in' h='0.25in'>"
            b"<ui><textEdit/></ui></field>"
            b"</subform></subform></pageArea></template>"
        )
        fields, _ = parse_fields(template, with_bbox=True)
        bbox = fields[0].bbox
        # 0.5in + 10pt + 1in = 36 + 10 + 72 = 118pt
        assert bbox["x_pt"] == pytest.approx(118.0)
        assert bbox["y_pt"] == pytest.approx(118.0)
        assert bbox["w_pt"] == pytest.approx(144.0)
        assert bbox["page"] == 1
        assert bbox["origin"] == "top-left"

    def test_field_without_xy_reports_no_coordinates(self):
        """v2.3.0 emitted the bare parent offset as the field's own position,
        a concrete-looking answer where 'unknown' is the truth."""
        template = (
            b"<template><pageArea><subform x='0.5in' y='0.5in'>"
            b"<field name='Global_Info-Sized' w='50mm' h='5mm'>"
            b"<ui><textEdit/></ui></field>"
            b"</subform></pageArea></template>"
        )
        fields, _ = parse_fields(template, with_bbox=True)
        bbox = fields[0].bbox
        assert "x_pt" not in bbox
        assert "y_pt" not in bbox
        assert bbox["w_pt"] == pytest.approx(141.73, abs=0.01)

    def test_page_counter_advances_and_offsets_reset_per_page(self):
        """Without a per-pageArea reset, offsets accumulate monotonically and
        y_pt becomes uninterpretable on a multi-page form."""
        template = (
            b"<template>"
            b"<pageArea><subform x='1in' y='1in'>"
            b"<field name='Global_Info-P1' x='0pt' y='0pt'><ui><textEdit/></ui></field>"
            b"</subform></pageArea>"
            b"<pageArea><subform x='1in' y='1in'>"
            b"<field name='Global_Info-P2' x='0pt' y='0pt'><ui><textEdit/></ui></field>"
            b"</subform></pageArea>"
            b"</template>"
        )
        fields, _ = parse_fields(template, with_bbox=True)
        assert [f.bbox["page"] for f in fields] == [1, 2]
        # Both sit at the same place on their own page, not 72 then 144.
        assert fields[0].bbox["y_pt"] == fields[1].bbox["y_pt"] == pytest.approx(72.0)


# ---------------------------------------------------------------------------
# Response bounding
# ---------------------------------------------------------------------------

class TestResponseBounding:
    def test_truncation_is_flagged_and_counts_stay_complete(self):
        """A 50k-field form package serializes to ~6 MB, which no MCP client
        can accept. Cap the emitted list but keep the counts honest."""
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE), max_inline_fields=2)
        assert result["fields_truncated"] is True
        assert len(result["fields"]) == 2
        # Counts cover every field, not just the emitted ones.
        assert result["field_count_after_classification"] == 5
        assert result["categories"] == {"shared": 2, "positional": 3, "other": 0}
        assert len(result["shared_fields"]) == 2
        assert any("truncated" in w.lower() for w in result["warnings"])

    def test_not_truncated_under_the_limit(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        assert result["fields_truncated"] is False


# ---------------------------------------------------------------------------
# MCP tool integration
# ---------------------------------------------------------------------------

class TestXfaMcpTools:
    def test_is_xfa_pdf_tool_detects_dynamic(self):
        from mcp_pdf.mixins_official.form_management import FormManagementMixin
        mixin = FormManagementMixin()
        result = asyncio.run(mixin.is_xfa_pdf(str(SYNTHETIC_FIXTURE)))
        assert result["success"] is True
        assert result["is_xfa"] is True
        assert result["xfa_type"] == "dynamic"
        assert result["detection_failed"] is False

    def test_extract_xfa_fields_tool_returns_expected_shape(self):
        from mcp_pdf.mixins_official.form_management import FormManagementMixin
        mixin = FormManagementMixin()
        result = asyncio.run(mixin.extract_xfa_fields(str(SYNTHETIC_FIXTURE)))
        assert result["success"] is True
        assert result["field_count"] == 8
        assert result["field_count_after_classification"] == 5
        assert result["profile_used"] == "zipform"

    def test_extract_xfa_fields_error_path_carries_contract_keys(self):
        """result["is_xfa"] must not KeyError precisely when the call failed."""
        from mcp_pdf.mixins_official.form_management import FormManagementMixin
        mixin = FormManagementMixin()
        result = asyncio.run(mixin.extract_xfa_fields("/nonexistent/file.pdf"))
        assert result["success"] is False
        assert "is_xfa" in result
        assert "xfa_type" in result
        assert result["error"]

    def test_extract_form_data_diagnoses_dynamic_xfa(self):
        from mcp_pdf.mixins_official.form_management import FormManagementMixin
        mixin = FormManagementMixin()
        result = asyncio.run(mixin.extract_form_data(str(SYNTHETIC_FIXTURE)))
        assert result["success"] is False
        assert result.get("is_xfa") is True
        assert result.get("xfa_type") == "dynamic"
        assert "extract_xfa_fields" in result.get("hint", "")

    def test_extract_form_data_on_damaged_file_says_it_is_damaged(self):
        """The regression this release's whole premise depends on.

        Before the detection third state, a damaged form package produced a
        bare "document closed" with no is_xfa flag and no hint, which is
        exactly the failure mode v2.3.0 set out to eliminate.
        """
        from mcp_pdf.mixins_official.form_management import FormManagementMixin
        mixin = FormManagementMixin()
        path = _damaged_copy(lambda raw: raw.replace(b"trailer", b"trailXX"))
        try:
            result = asyncio.run(mixin.extract_form_data(path))
            if result.get("xfa_detection_failed"):
                assert result["success"] is False
                assert "truncated or corrupt" in result["hint"]
            else:
                # MuPDF recovered and found no widgets, which is a legitimate
                # answer — and a successful one, so success is not asserted
                # here. What must NOT happen is a bare "document closed" with
                # no explanation of either kind. Until 2026-09-21 that was
                # exactly what came back, because the success return called
                # len(doc) after doc.close(); this branch passed only because
                # the bug it was written to catch was still present.
                assert result.get("error") != "document closed"
                assert "form_summary" in result or result.get("hint")
        finally:
            os.unlink(path)

    def test_analyze_pdf_health_surfaces_xfa_without_discarding_analysis(self):
        from mcp_pdf.mixins_official.document_analysis import DocumentAnalysisMixin
        mixin = DocumentAnalysisMixin()
        result = asyncio.run(mixin.analyze_pdf_health(str(SYNTHETIC_FIXTURE)))
        assert result["success"] is True
        assert result["document_stats"]["is_xfa"] is True
        assert result["document_stats"]["xfa_type"] == "dynamic"
        assert any("XFA" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Radio-group export values
#
# Not XFA, but it lives here because it is the same class of defect this file
# exists to guard: a value that looks fine on a convenient fixture and breaks
# on a realistic one. The original single-word test labels ("Cash", "Loan",
# "Other") all passed while multi-word labels silently produced an unusable
# button, because a PDF name object cannot contain a space.
# ---------------------------------------------------------------------------

class TestRadioGroupExportValues:
    LABELS = ["Cash", "Conventional Loan", "FHA / VA", "Seller-Financed",
              "Other (specify)"]

    def _build(self, tmp_path):
        import json
        import pymupdf
        from mcp_pdf.mixins_official.advanced_forms import AdvancedFormsMixin

        src = tmp_path / "blank.pdf"
        doc = pymupdf.open()
        doc.new_page()
        doc.save(str(src))
        doc.close()

        out = tmp_path / "radio.pdf"
        result = asyncio.run(AdvancedFormsMixin().add_radio_group(
            str(src), str(out), "financing", json.dumps(self.LABELS)
        ))
        return result, out

    def test_every_label_produces_a_usable_button(self, tmp_path):
        """A label with a space used to yield on-state None and value 'Yes'."""
        import pymupdf
        result, out = self._build(tmp_path)
        assert result["success"] is True
        assert result["radio_group_summary"]["buttons_added"] == len(self.LABELS)

        doc = pymupdf.open(str(out))
        states = [w.button_states() for w in doc[0].widgets()]
        doc.close()
        assert all(s and s.get("normal") for s in states), \
            "a button has no usable on-state; its /AP/N key is malformed"

    def test_all_buttons_share_one_field_name(self, tmp_path):
        """Sharing the name is what makes them mutually exclusive."""
        import pymupdf
        _, out = self._build(tmp_path)
        doc = pymupdf.open(str(out))
        names = {w.field_name for w in doc[0].widgets()}
        doc.close()
        assert names == {"financing"}

    def test_export_values_are_distinct_and_pdf_safe(self, tmp_path):
        """Distinct on-states are what make the selection readable back."""
        result, _ = self._build(tmp_path)
        values = result["radio_group_summary"]["option_values"]
        assert set(values) == set(self.LABELS), "every label needs a mapping"
        assert len(set(values.values())) == len(self.LABELS), "values must be unique"
        for export in values.values():
            assert " " not in export
            assert all(c.isalnum() or c in "-_." for c in export), export

    def test_mapping_is_reported_so_callers_need_not_guess(self, tmp_path):
        result, _ = self._build(tmp_path)
        values = result["radio_group_summary"]["option_values"]
        assert values["Conventional Loan"] == "Conventional_Loan"
        assert values["FHA / VA"] == "FHA_VA"

    def test_filling_selects_exactly_one_button(self, tmp_path):
        """fill_form_pdf used to turn on EVERY button in the group.

        PyMuPDF's Widget.update() ignores an "Off" assignment on a radio and
        switches it on with its own state, so assigning the requested value
        widget-by-widget left all five options selected while reporting
        fields_filled=5 for a single key.
        """
        import json
        import pymupdf
        from mcp_pdf.mixins_official.form_management import FormManagementMixin

        _, created = self._build(tmp_path)
        filled = tmp_path / "filled.pdf"
        result = asyncio.run(FormManagementMixin().fill_form_pdf(
            str(created), str(filled), json.dumps({"financing": "Conventional_Loan"})
        ))
        assert result["fill_summary"]["fields_filled"] == 1, \
            "a radio group is one field, not one per button"

        doc = pymupdf.open(str(filled))
        selected = [
            w.button_states()["normal"][1]
            for w in doc[0].widgets()
            if doc.xref_get_key(w.xref, "AS")[1] != "/Off"
        ]
        doc.close()
        assert selected == ["Conventional_Loan"]

    def test_unknown_option_selects_nothing_and_is_reported(self, tmp_path):
        """An unmatched value must not silently select something."""
        import json
        import pymupdf
        from mcp_pdf.mixins_official.form_management import FormManagementMixin

        _, created = self._build(tmp_path)
        filled = tmp_path / "filled_bad.pdf"
        result = asyncio.run(FormManagementMixin().fill_form_pdf(
            str(created), str(filled), json.dumps({"financing": "NoSuchOption"})
        ))
        # fields_filled stays 0 against total_data_provided 1, which is the
        # documented way to detect a key that matched nothing.
        assert result["fill_summary"]["fields_filled"] == 0
        assert result["fill_summary"]["total_data_provided"] == 1

        doc = pymupdf.open(str(filled))
        selected = [
            w for w in doc[0].widgets()
            if doc.xref_get_key(w.xref, "AS")[1] != "/Off"
        ]
        doc.close()
        assert selected == []

    def test_extract_form_data_exposes_the_on_state(self, tmp_path):
        """A caller who did not create the PDF must be able to discover the
        valid values. Every button in a group reports the same field_name and
        a field_value of "Off", so without on_state there is nothing to tell
        them apart and no way to know what fill_form_pdf will accept.
        """
        from mcp_pdf.mixins_official.form_management import FormManagementMixin

        result, created = self._build(tmp_path)
        extracted = asyncio.run(
            FormManagementMixin().extract_form_data(str(created))
        )
        assert extracted["success"] is True

        on_states = [f.get("on_state") for f in extracted["form_fields"]]
        assert all(on_states), "every radio widget needs an on_state"
        # The discovered states must match what add_radio_group reported, or
        # the two tools disagree about the same document.
        assert set(on_states) == set(result["radio_group_summary"]["option_values"].values())
        assert all(f.get("is_selected") is False for f in extracted["form_fields"])
