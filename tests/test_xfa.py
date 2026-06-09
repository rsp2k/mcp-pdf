"""Tests for XFA (dynamic Adobe LiveCycle form) support.

Pins behavior against the synthetic XFA fixture contributed via the
agent-thread at ``docs/agent-threads/xfa-form-support/`` (specifically the
expected-output table in ``005-iar-attaches-fixture.md``). Each assertion
here corresponds to a row in that table.

The fixture is hand-built, license-clean, and small (~2 KB). Its README at
``tests/fixtures/xfa/README.md`` documents what each field exercises.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mcp_pdf.xfa import (
    canonicalize,
    classify_fields,
    extract_xfa_parts,
    extract_xfa_schema,
    fields_from_pdf,
    is_xfa_pdf,
    parse_fields,
    _ZIPFORM_PROFILE,
    _GENERIC_PROFILE,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "xfa"
SYNTHETIC_FIXTURE = FIXTURE_DIR / "synthetic_dynamic_xfa.pdf"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestXfaDetection:
    """is_xfa_pdf and the catalog /NeedsRendering flag."""

    def test_synthetic_fixture_detected_as_dynamic_xfa(self):
        result = is_xfa_pdf(str(SYNTHETIC_FIXTURE))
        assert result["is_xfa"] is True
        assert result["xfa_type"] == "dynamic"
        assert result["has_acroform"] is True

    def test_nonexistent_pdf_returns_not_xfa(self):
        result = is_xfa_pdf("/nonexistent/path.pdf")
        assert result["is_xfa"] is False
        assert result["xfa_type"] is None


# ---------------------------------------------------------------------------
# XFA packet extraction (array-form parsing)
# ---------------------------------------------------------------------------

class TestXfaPartsExtraction:
    """The fixture uses the array form [template, datasets] — exercises the
    name/stream-pair parsing path, not the single-stream path."""

    def test_extracts_template_and_datasets(self):
        parts = extract_xfa_parts(str(SYNTHETIC_FIXTURE))
        assert "template" in parts
        assert "datasets" in parts
        assert isinstance(parts["template"], bytes)
        assert isinstance(parts["datasets"], bytes)

    def test_template_is_parseable_xml(self):
        parts = extract_xfa_parts(str(SYNTHETIC_FIXTURE))
        # If <template> isn't well-formed XML, parse_fields silently returns []
        # — which would also pass if the template were empty. So assert content
        # AND that parse_fields actually finds fields.
        assert len(parts["template"]) > 0
        fields = parse_fields(parts["template"])
        assert len(fields) > 0


# ---------------------------------------------------------------------------
# Field parsing
# ---------------------------------------------------------------------------

class TestFieldParsing:
    """parse_fields against the fixture's 8-field template."""

    def test_finds_eight_raw_fields(self):
        """All 8 fields from fixture/README.md table, before classification."""
        fields = fields_from_pdf(str(SYNTHETIC_FIXTURE))
        assert len(fields) == 8

    def test_caption_harvested_from_field_with_caption(self):
        """The fixture has one in-field <caption> on Test-Property-Address."""
        fields = fields_from_pdf(str(SYNTHETIC_FIXTURE))
        property_address = next(
            f for f in fields if f.name == "Global_Info-Test-Property-Address"
        )
        assert property_address.caption == "Property Address"

    def test_ui_types_extracted(self):
        fields = fields_from_pdf(str(SYNTHETIC_FIXTURE))
        by_name = {f.name: f for f in fields}
        # textEdit -> text in vocabulary; checkButton -> checkbox
        assert by_name["Global_Info-Test-Property-Address"].ui_type == "textEdit"
        assert by_name["Global_Info-Test-Buyer-Name"].ui_type == "checkButton"
        assert by_name["p01cb002"].ui_type == "checkButton"


# ---------------------------------------------------------------------------
# Classification (the critical behaviors)
# ---------------------------------------------------------------------------

class TestClassificationBehavior:
    """All the rows from the fixture README's expected-output table."""

    def test_classification_counts_match_fixture_spec(self):
        """README: 2 shared, 3 positional, 3 plumbing dropped."""
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        shared = [f for f in result["fields"] if f["category"] == "shared"]
        positional = [f for f in result["fields"] if f["category"] == "positional"]
        assert len(shared) == 2
        assert len(positional) == 3
        assert len(result["plumbing_fields_dropped"]) == 3

    def test_shared_canonical_names(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        canonical = set(result["shared_fields"].values())
        assert canonical == {"test_property_address", "test_buyer_name"}

    def test_positional_codes_preserved_verbatim(self):
        """Opaque codes have no canonical_name; original is preserved as-is."""
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        positional = sorted(
            f["original"] for f in result["fields"] if f["category"] == "positional"
        )
        assert positional == ["p01cb002", "p01tf001", "p02te001"]
        # Also: no canonical_name on positional fields (response shape contract)
        for f in result["fields"]:
            if f["category"] == "positional":
                assert "canonical_name" not in f

    def test_plumbing_drops_match_fixture_spec(self):
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        assert set(result["plumbing_fields_dropped"]) == {
            "tbd1", "folder_id", "Global_Info-Invisibind-Test",
        }

    def test_denylist_beats_shared_prefix(self):
        """The most important invariant: Global_Info-Invisibind-Test has the
        shared prefix AND matches the invisibind denylist regex — plumbing
        MUST win. If this regresses, value-collection pipelines would silently
        try to fill a producer-internal field as a user-facing form field."""
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        # It must be in plumbing_dropped, NOT in fields, NOT in shared_fields.
        assert "Global_Info-Invisibind-Test" in result["plumbing_fields_dropped"]
        assert not any(
            f["original"] == "Global_Info-Invisibind-Test"
            for f in result["fields"]
        )
        assert "Global_Info-Invisibind-Test" not in result["shared_fields"]

    def test_positional_regex_does_not_overmatch_different_pages(self):
        """The fixture deliberately includes p01tf001, p01cb002, AND p02te001 —
        the regex must accept all three different page prefixes (01 vs 02) and
        type codes (tf, cb, te). If the regex anchored to a specific page or
        type, this would silently miss fields."""
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        positional = {
            f["original"] for f in result["fields"] if f["category"] == "positional"
        }
        assert positional == {"p01tf001", "p01cb002", "p02te001"}

    def test_field_count_is_raw_count(self):
        """field_count reports the raw field count BEFORE plumbing drops,
        for auditability — separate from field_count_after_classification."""
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        assert result["field_count"] == 8
        assert result["field_count_after_classification"] == 5

    def test_original_present_on_every_field(self):
        """Round-trip key contract from 004: every field carries 'original'."""
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        for f in result["fields"]:
            assert "original" in f
            assert f["original"]

    def test_caption_used_as_label_for_shared(self):
        """When a shared field has an in-field <caption>, use it as label
        instead of deriving from the canonical name."""
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE))
        property_address = next(
            f for f in result["fields"]
            if f["original"] == "Global_Info-Test-Property-Address"
        )
        assert property_address["label"] == "Property Address"


# ---------------------------------------------------------------------------
# Canonicalization (pure function)
# ---------------------------------------------------------------------------

class TestCanonicalization:
    """The mechanical name transform — separate from classification."""

    @pytest.mark.parametrize("original, expected", [
        ("Global_Info-Test-Property-Address", "test_property_address"),
        ("Global_Info-Seller-Broker-Entity-Name", "seller_broker_entity_name"),
        ("Global_Info-Buyer-Name", "buyer_name"),
        # camelCase boundaries get an underscore
        ("Global_Info-BuyerName", "buyer_name"),
        # Special chars collapse to single underscores
        ("Global_Info--Field--Name", "field_name"),
        # Without the prefix, just normalizes
        ("Some-Other-Field", "some_other_field"),
    ])
    def test_snake_case(self, original, expected):
        assert canonicalize(original, _ZIPFORM_PROFILE, separator="_") == expected

    def test_dotted_separator(self):
        result = canonicalize(
            "Global_Info-Seller-Broker-Name", _ZIPFORM_PROFILE, separator=".",
        )
        assert result == "seller.broker.name"

    def test_kebab_separator(self):
        result = canonicalize(
            "Global_Info-Property-Address", _ZIPFORM_PROFILE, separator="-",
        )
        assert result == "property-address"


# ---------------------------------------------------------------------------
# Profiles (zipform vs generic + caller extensions)
# ---------------------------------------------------------------------------

class TestProfiles:
    def test_generic_profile_keeps_positional_as_other(self):
        """The generic profile has NO positional patterns, so p01tf001 etc.
        get classified as 'other' (not 'positional') without caller extension."""
        result = extract_xfa_schema(str(SYNTHETIC_FIXTURE), profile="generic")
        positional_count = sum(
            1 for f in result["fields"] if f["category"] == "positional"
        )
        other_count = sum(1 for f in result["fields"] if f["category"] == "other")
        assert positional_count == 0
        # generic profile still drops nothing as plumbing (no patterns), so
        # Invisibind/tbd1/folder_id all show up as 'other' too
        assert other_count > 0

    def test_caller_can_add_plumbing_pattern(self):
        """Custom plumbing patterns get appended to the profile's denylist."""
        result = extract_xfa_schema(
            str(SYNTHETIC_FIXTURE),
            profile="zipform",
            extra_plumbing_patterns=[r"^p01tf"],
        )
        # p01tf001 should now be dropped as plumbing
        assert "p01tf001" in result["plumbing_fields_dropped"]
        assert result["profile_used"].endswith("+custom")

    def test_caller_can_add_positional_pattern(self):
        """Custom positional patterns identify additional opaque codes."""
        # Generic profile + custom positional regex should categorize p01tf001
        result = extract_xfa_schema(
            str(SYNTHETIC_FIXTURE),
            profile="generic",
            extra_positional_patterns=[r"^p\d+[a-z]+\d+$"],
        )
        positional = [f for f in result["fields"] if f["category"] == "positional"]
        assert len(positional) == 3

    def test_caller_can_add_exact_plumbing_names(self):
        """Custom exact plumbing names extend the denylist."""
        result = extract_xfa_schema(
            str(SYNTHETIC_FIXTURE),
            extra_plumbing_exact=["p01tf001"],
        )
        assert "p01tf001" in result["plumbing_fields_dropped"]


# ---------------------------------------------------------------------------
# MCP tool integration (mixin-level, async)
# ---------------------------------------------------------------------------

class TestXfaMcpTools:
    """Verify the MCP tool entry points wire through to xfa.py correctly."""

    def test_is_xfa_pdf_tool_detects_dynamic(self):
        from mcp_pdf.mixins_official.form_management import FormManagementMixin
        mixin = FormManagementMixin()
        result = asyncio.run(mixin.is_xfa_pdf(str(SYNTHETIC_FIXTURE)))
        assert result["is_xfa"] is True
        assert result["xfa_type"] == "dynamic"

    def test_extract_xfa_fields_tool_returns_expected_shape(self):
        from mcp_pdf.mixins_official.form_management import FormManagementMixin
        mixin = FormManagementMixin()
        result = asyncio.run(mixin.extract_xfa_fields(str(SYNTHETIC_FIXTURE)))
        assert result["is_xfa"] is True
        assert result["field_count"] == 8
        assert result["field_count_after_classification"] == 5
        assert result["profile_used"] == "zipform"

    def test_extract_form_data_diagnoses_dynamic_xfa(self):
        """The big UX win from 003+004: instead of cryptic 'document closed',
        callers get a clear error + hint pointing to extract_xfa_fields."""
        from mcp_pdf.mixins_official.form_management import FormManagementMixin
        mixin = FormManagementMixin()
        result = asyncio.run(mixin.extract_form_data(str(SYNTHETIC_FIXTURE)))
        assert result["success"] is False
        assert result.get("is_xfa") is True
        assert result.get("xfa_type") == "dynamic"
        assert "extract_xfa_fields" in result.get("hint", "")
