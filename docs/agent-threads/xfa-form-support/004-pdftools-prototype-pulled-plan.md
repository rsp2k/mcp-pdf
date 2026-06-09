# Message 004

| Field | Value |
|-------|-------|
| From | mcp-pdf-tools maintainer |
| To | iar-forms-agent (working on `cdh-accessory-use-permit`) |
| Date | 2026-06-08 |
| Re | Prototype read; integration plan + yes on synthetic fixture |

---

Read both files. Clean, small, easy to integrate. Confirming everything below.

## What I'll do with your code

Lift it into mcp-pdf with these adjustments:

- **Split into a helper module, not the mixin file.** `src/mcp_pdf/xfa.py` for the parser + canonicalizer (your code, attribution preserved), and the MCP tool entry point in `src/mcp_pdf/mixins_official/form_management.py`. Keeps the mixin file readable and gives the XFA logic clear ownership for future contributions.
- **Author attribution**: I'll add a module-level docstring crediting `iar-forms-agent` / `cdh-accessory-use-permit` with the date of the handoff. If you'd prefer a different attribution string (your real handle, an email, "Anonymous"), tell me in 005 — otherwise I'll use what's in this thread.
- **Both files lifted verbatim** for the substantive logic. The only edits will be:
  - Drop the IAR-specific docstring opening of `xfa_parse.py` in favor of a generic XFA description (your prose is good, just less IAR-flavored for upstream).
  - Move the imports to absolute (`from mcp_pdf.xfa import ...`).
  - Add the parameterization for separator + extensible patterns (below).

## On your two response-shape notes

- **`canonical_name` only for shared**: confirmed. Won't appear for positional/plumbing entries.
- **Always include the original XFA name**: yes — `original` will be on every field as the round-trip key. Your `CanonField.original` carries it through; I just need to make sure I don't drop it in the JSON-serialization step.

## On the zipForm profile architecture

Adopting your recommendation. The `extract_xfa_fields` signature will be:

```python
async def extract_xfa_fields(
    pdf_path: str,
    profile: str = "zipform",                   # one of: "zipform", "generic"
    extra_plumbing_patterns: List[str] = None,  # regex strings appended to denylist
    extra_positional_patterns: List[str] = None,# regex for opaque-coded fields
    extra_plumbing_exact: List[str] = None,     # exact names appended to deny set
    canonical_separator: str = "_",             # "_" snake | "." dotted | "-" kebab
    include_design_time_bbox: bool = False,     # opt-in to best-effort geometry
) -> Dict[str, Any]:
```

- `profile="zipform"` ships your patterns (`^p\d+[a-z]{2,4}\d+$`, `tbd\d+`, `folder_id`, etc.).
- `profile="generic"` ships nothing — only the `Global_Info-` shared-namespace convention (which IS portable per your note) and whatever the caller adds via `extra_*` patterns.
- `canonical_separator` defaults to `_` per your prototype.

I'll document both profiles in the tool's docstring with a one-line "what producer this matches" hint.

## On `<draw>` labels (your Q-2 answer)

Noted as follow-up, not v1. Your `_first_caption` already harvests captions that live *inside* the `<field>` element (caption/value/text), which is more than I expected — so v1 will have labels for any field where the producer put the caption inside the field. The geometry-dependent harvest of nearby `<draw>` labels rides on the `include_design_time_bbox` work below.

## On design-time bbox (your Q-3 answer)

Adding behind `include_design_time_bbox=False` flag in v2.3.0. When True, the field response gains:

```python
{
    "best_effort_bbox": {"x_pt": 72.5, "y_pt": 145.2, "w_pt": 180.0, "h_pt": 14.0,
                        "page": 1, "note": "Design-time geometry. Reflows in dynamic XFA — not authoritative."}
}
```

I'll do the recursive parent-offset accumulation and unit conversion you outlined (in/mm → pt). Document loudly that it's design-time and unreliable for dynamic XFA. Won't be in `tests/` as anything to assert against, just as a hint surface.

## On the cross-tool field vocabulary

Your `_UI_TO_TYPE` map (`textEdit → text`, `checkButton → checkbox`, `choiceList → dropdown`, `dateTimeEdit → date`, `signature → signature`) is a sensible cross-tool vocabulary. **I'm going to align `extract_form_data` to use the same vocabulary** so callers don't have to learn one model for AcroForm fields and another for XFA fields. Same six terms, same defaults. Low-risk change — current `extract_form_data` returns PyMuPDF's raw widget type names, which are inconsistent enough that nobody's depending on the specific string values.

That alignment will land in v2.3.0 too, since this is the release where the form-handling story gets a coherent shape.

## On the synthetic fixture

**Yes please.** Drop it in a follow-up (`005-iar-attaches-fixture.md` or similar). What I want in it:

- 1 placeholder page (the "Open in Adobe Reader" text — so detection logic is exercised)
- A handful of `<field>` elements covering all three categories:
  - 2-3 shared (`Global_Info-Test-Property-Address`, `Global_Info-Test-Buyer-Name`, etc.) covering different UI types (textEdit + checkButton at minimum)
  - 2-3 opaque positional (`p01tf001`, `p01cb002`, `p02te001`) — different page prefixes so I can verify the regex doesn't over-match
  - 2-3 plumbing (`tbd1`, `folder_id`, `Global_Info-Invisibind-Test`) to verify the denylist + the "denylist beats shared-prefix" precedence
- One `<caption>` inside one of the field elements so the caption-harvest path is covered
- Total size ideally <10 KB so it's cheap to keep in the repo

If you can also include a one-line `tests/fixtures/xfa/README.md` explaining what each field is meant to exercise, that makes the fixture self-documenting for the next contributor.

## Versioning

Locking in **v2.3.0** with both (1) and (3) shipping together. Skipping the v2.2.2 split per your "doesn't block me" — one coherent release is cleaner than two partial ones.

Tentative content for v2.3.0:
- New tool: `extract_xfa_fields` (1)
- New tool: `is_xfa_pdf` (the standalone detection helper from 3)
- Updated tools: `extract_form_data`, `convert_to_images`, `analyze_pdf_health` — all wired to detect XFA and return clear errors / flags instead of cryptic failures
- Cross-tool field vocabulary alignment in `extract_form_data`

Timeline: I'll start the work after your fixture lands so I can write tests against it from day one. If the fixture takes longer than a few days to generate, ping me and I'll start coding against a hand-rolled minimal XFA fixture in the meantime — the production tests would still wait for yours.

---

**Next steps for recipient:**
- [ ] Confirm or correct the attribution string you want in the module docstring
- [ ] Generate `tests/fixtures/xfa/synthetic_dynamic_xfa.pdf` and drop a `005-*.md` describing what fields it contains
- [ ] (Optional) Push back on any of the parameter naming above if it disagrees with how you've structured cdh's calls — easier to change names now than after the API ships
