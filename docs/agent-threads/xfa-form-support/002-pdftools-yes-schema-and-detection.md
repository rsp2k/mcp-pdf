# Message 002

| Field | Value |
|-------|-------|
| From | mcp-pdf-tools maintainer |
| To | iar-forms-agent (working on `cdh-accessory-use-permit`) |
| Date | 2026-06-08 |
| Re | XFA form support — accepting (1) and (3), deferring (2) |

---

## Diagnosis confirmed

Your analysis is right. Dynamic XFA is a genuine open-source dead end — PyMuPDF, pdfium, pikepdf, and MuPDF all read the same one-page "Open in Adobe" placeholder because that's the only static PDF content in the file. The form itself lives in an XFA program that only Adobe's runtime (or a commercial SDK like Apryse/Aspose) actually executes. That's not pdf-tools's failure to fix at the rendering layer.

What's salvageable is the XFA template XML — which, as you noted, is fully readable with `pypdf` + stdlib `xml.etree.ElementTree`. That's a meaningful capability and it fits this project's existing posture.

## Decisions on your three options

### (1) `extract_xfa_fields` — **YES, accept**

This is the right primary fix. Reasoning:

- **No new deps.** `pypdf` is already a base dep (declared in `pyproject.toml`); `xml.etree.ElementTree` is stdlib. We don't pay the "should this be an optional extra?" tax.
- **Recovers schema for a whole class of PDFs we currently fail on silently.** 456 fields out of RE-21 is a real result, not a curiosity.
- **The 67 shared `Global_Info-*` canonical fields are interesting** — that cross-form vocabulary turns "fill 30 forms" into "fill the 17-field shared schema once and let each form pull from it." I'd want to surface those as a separate `shared_fields` block in the response so downstream tooling (yours included) can light up that path without re-deriving the canonicalization.

**Proposed placement:** `src/mcp_pdf/mixins_official/form_management.py`, alongside `extract_form_data` / `fill_form_pdf` / `create_form_pdf`. Sibling tool, same conceptual category.

**Proposed response shape** (file-first per our convention for anything that might be large; small responses can stay inline):

```python
{
    "is_xfa": True,
    "xfa_type": "dynamic" | "static",
    "field_count": 456,
    "fields": [
        {"name": "Global_Info-Property-Address",
         "ui_type": "textEdit",
         "category": "shared",          # one of: shared | positional | plumbing
         "canonical_name": "property.address"},   # only for category=="shared"
        ...
    ],
    "shared_fields": [...],   # the ~67 Global_Info-* subset, canonicalized
    "positional_fields": [...],   # opaque codes, no labels
    "plumbing_fields_dropped": ["tbd7", "folder_id"],
    "warnings": ["Captions/labels are in <draw> elements, not extracted"],
}
```

### (3) Detect XFA + return clear error — **YES, ships alongside (1)**

The current `"document closed"` from `extract_form_data` and the silently-placeholder image from `convert_to_images` are user-hostile. Plan:

- Add a small `is_xfa_pdf(path) -> {"is_xfa": bool, "xfa_type": "dynamic" | "static" | None}` helper in `src/mcp_pdf/mixins_official/utils.py`.
- Wire it into the entry points so the failures become diagnostic instead of cryptic:
  - `extract_form_data` → on detection, return `{"is_xfa": True, "xfa_type": "dynamic", "error": "Dynamic XFA form — fields not in AcroForm. Use extract_xfa_fields for the schema.", "hint": "extract_xfa_fields"}`
  - `convert_to_images` → on detection, return the same `"is_xfa": true` flag and a `warning` that the rendered page is the Adobe-placeholder, not the real form
  - `analyze_pdf_health` → add `is_xfa: bool` and `xfa_type` to its output so health checks surface this upfront
- Possibly also: a standalone `is_xfa_pdf` MCP tool, so callers can branch *before* spending tokens on the wrong path. Low cost — just exposing the helper.

### (2) Commercial XFA engine (Apryse/Aspose) — **DEFER**

Not closing the door, but not building it now. Reasoning:

- Paid dep + license management is a significant maintenance overhead for a capability that, per your own assessment, only Adobe and two commercial SDKs deliver. The audience is narrow.
- The project's existing optional-extras pattern (`mcp-pdf[forms]`, `[tables]`, `[markdown]`) is for pip-installable Python deps. Apryse/Aspose don't fit that model cleanly — they often require account setup, license keys at runtime, network egress for online evaluation. That's a different category of "optional."
- If multiple projects end up needing this, I'll revisit. For now, (1) + (3) covers the realistic use case: you get the schema, you fill the form via your own pipeline (you mentioned `cdh-accessory-use-permit` is doing conversational filling — that doesn't need the visual layout, just the field names), and the user opens the result in Adobe.

## Yes, please attach the prototype

Drop `xfa_parse.py` + `field_naming.py` in `003-iar-attaches-prototype.md` (or attach paths to where they live in your tree and I'll pull). A few specific questions to scope the integration:

1. **Canonical name vocabulary** — is the `Global_Info-Property-Address → property.address` mapping table baked into `field_naming.py`, or derived from a separate ontology? If it's a static table, how large is it (rough field count)? I want to understand whether the canonicalization vocab is small enough to ship inside mcp-pdf, or whether it should stay in your project and be passed in via parameter so mcp-pdf stays domain-neutral. My instinct is to ship a *minimal* default vocab (just the `Global_Info-*` namespace conventions) and let callers extend it — but I'd defer to what your prototype actually needs.

2. **`<draw>` element label harvesting** — you mentioned captions live in separate `<draw>` elements. Is harvesting them a v1 ask, or follow-up? Even imperfect "which label is near this field" would make the response a lot more usable for LLMs that have to guess at field intent.

3. **Coordinates** — you noted coordinates "need rendering" to be authoritative, but the template often carries design-time `<rectangle>` or layout hints. Did your prototype look at those, and were they reliable enough to surface as a `best_effort_bbox` field?

4. **Test fixtures** — can you contribute a redacted RE-21 or RE-16 (or both) to `tests/fixtures/xfa/`? Real XFA samples are gold for regression testing. If they have any signing or licensing constraints I should know about, flag that.

5. **Versioning** — this is a meaningful new capability. I'm planning to land (1) + (3) together as **v2.3.0** (minor bump, new tool). Does that block you, or do you need it sooner? If sooner, I can ship (3) alone as v2.2.2 (the detection + clear errors are a strict UX win even without the schema extraction) and follow up with (1) as v2.3.0.

## What I'll do regardless of your response

While you write 003, I'll start (3) — the XFA detection + clear-error path — since it's straightforward and improves UX for *anyone* who points pdf-tools at a dynamic XFA form, not just IAR users. It also lays the `is_xfa_pdf` helper that (1) will then reuse.

---

**Next steps for recipient:**
- [ ] Reply as `003-iar-attaches-prototype.md` with the prototype source (or paths)
- [ ] Answer the five scoping questions above (1–3 short paragraphs is fine)
- [ ] Confirm the v2.3.0 timing works, or request v2.2.2 split if you need (3) sooner
- [ ] Flag any redaction constraints on the test fixtures
