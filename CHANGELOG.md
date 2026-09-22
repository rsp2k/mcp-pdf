# Changelog

## Versioning

As of `2026.09.21` this project uses **CalVer**, `YYYY.MM.DD`, with a PEP 440
post-release segment for same-day fixes (`2026.09.21.1`, `.2`, ...).

The reason is what the number is actually useful for. This package is a thin
layer over PyMuPDF, pdfplumber, pypdf, Camelot, Tabula, Tesseract, pandoc and a
handful of system binaries, and nearly every surprising behavior traces back to
one of those rather than to code here. When a PDF misbehaves, the question you
have is "when was this last tested against those?", which a date answers and a
semver number does not.

Ordering is safe across the switch. PEP 440 compares the release tuple
numerically, so `(2026, 9, 21)` sorts after `(2, 3, 1)` on the first component
and `pip install --upgrade` behaves correctly for anyone on an older release.
Leading zeros normalize, so `2026.09.21` and `2026.9.21` are the same version.

Releases before `2026.09.21` used semver and are listed below under their
original numbers.

---

## 2026.09.22

Came out of auditing the tool schemas from a calling model's point of view. That
surfaced fourteen places where the code did not do what its own docstring said,
and fixing those turned up a dozen real bugs underneath.

### Breaking

- **`add_radio_group` buttons now share one field name.** They previously got
  `{group}_0`, `{group}_1`, ... and PDF enforces mutual exclusion only among
  widgets sharing a name, so the "group" was a column of independent
  checkboxes a user could tick all of. Anything storing the old per-button
  names needs updating; in exchange the field's value is now the selected
  option rather than a meaningless `"Yes"`.
- **`extract_form_data` field types** use the portable vocabulary shared with
  the XFA tools (`text`, `checkbox`, `radio`, `dropdown`, `date`, `signature`,
  plus `button` / `unknown`). `listbox` and `combobox` both report as
  `dropdown`; the original value survives as `field_type_raw`.
- **`is_xfa_pdf` has a third outcome.** `is_xfa` is `True`, `False`, or `None`,
  where `None` means detection failed. **Check `detection_failed` before
  trusting `is_xfa`.**
- **`detect_structure` rejects an unrecognised `strategies`** instead of
  running no detector and returning an empty structure with `success: true`.

### Fixed

Tools that reported success while doing nothing:

- `extract_form_data` returned `{"success": false, "error": "document closed"}`
  for **every valid PDF**, because its success return called `len(doc)` after
  `doc.close()`.
- `add_form_fields`, `add_radio_group`, `add_textarea_field` and
  `add_date_field` created **no widgets at all**. Each called `add_widget()` on
  a bare `Widget` whose `rect` is `None`, and each tool's own handler swallowed
  the `AttributeError` and reported success on a field-less PDF.
- `extract_text`'s `method` parameter: `"pymupdf"`, `"pdfplumber"` and
  `"pypdf"` hit an unimplemented stub returning empty text with
  `success: true` and a zero-byte file. All three are now real, and `"auto"`
  walks the PyMuPDF → pdfplumber → pypdf cascade the README always described
  but never had.
- `merge_pdfs` silently discarded every bookmark. They are now carried across
  with page targets rebased, and reported.
- `merge_pdfs_advanced(include_toc=True)` failed entirely on an invalid
  base-14 font name; `add_page_numbers` was accepted and never acted on.
- `optimize_pdf`'s `preserve_quality` was never read. Setting it `False` now
  recompresses images: a photo-heavy PDF goes from 0.2% to **93.8%** smaller.
- `fill_form_pdf` turned on **every** button in a radio group for one data key.
- `create_form_pdf` silently dropped every dropdown.

Tools reporting constants as measurements:

- `pdf_version` was always `"Unknown"`, `is_linearized` always `False` and
  `embedded_file_count` permanently `0`, all read from attribute names PyMuPDF
  does not have. The "old PDF version" warnings guarded by them could never
  fire.
- `is_scanned_pdf` divided image area in **pixels** by page area in **points**,
  so `image_coverage_percent` routinely exceeded 100 and `large_image_present`
  fired on any photo. Its `confidence` also came from a different threshold
  ladder than its verdict, so it could return `is_scanned: true` alongside
  `confidence: 0.2` labelled "likely text-based".
- `analyze_layout`'s `include_coordinates=False` silently disabled column
  detection, because the heuristic read back the coordinates it had just
  dropped.
- `classify_content`'s documented `"general"` fallback was unreachable; an
  unmatched document was reported as `"academic"` at confidence 0.0.
- `detect_structure`'s `"auto"` was byte-identical to `"all"`. It is now
  adaptive: bookmarks first, heuristics only when there is no usable outline.
- `batch_extract` reported `success: true` when every section failed.
- `markdown_to_pdf` had no `success` key on any of its nine return paths.

### Added

- `extract_charts` **extracts**. Pass `output_directory` to crop every flagged
  region to a PNG. Without it the behaviour is unchanged and nothing is
  written.
- `extract_form_data` reports `on_state` and `is_selected` for button widgets,
  which is how a caller discovers what value selects a radio option in a form
  they did not create.
- `add_radio_group` returns `option_values`, the label → export-value mapping
  (`"Conventional Loan"` → `"Conventional_Loan"`). A PDF name cannot contain a
  space, and building the on-state straight from the label produced a silently
  unusable button.
- All 54 tools carry MCP annotations (`readOnlyHint`, `destructiveHint`,
  `idempotentHint`, `openWorldHint`), and 17 enum-ish parameters became real
  JSON-Schema enums, so a wrong value is rejected rather than merely
  discouraged.
- Every parameter is documented. JSON-string parameters now state their exact
  shape with an example, which is what `split_pdf_by_pages` needed: it takes
  `["1-5","6-10"]`, and `"1"` or `"1-1"` both fail with different errors.

### Changed

- Median tool description went from ~60 characters to ~1,200. Seventeen tools
  had descriptions under 40 characters.
- Test suite 105 → 112.

---

## 2026.09.21.1

Dependency currency, and two declared floors that were fiction.

### Fixed

- **`import fitz` migrated to `import pymupdf`** across all 12 mixins.
  PyMuPDF renamed its module in 1.24.3 and the `fitz` shim now prints
  `The 'fitz' API is deprecated and will be removed in future` on import.
  Every user of `2026.09.21` saw that on server startup. It was invisible
  during development because `uv.lock` pinned PyMuPDF 1.26.3, whose shim is
  silent, while fresh installs resolved 1.28.2, whose shim warns. Verified all
  26 PyMuPDF attributes this package uses resolve identically on both module
  names before switching.

- **`PyMuPDF>=1.23.0` raised to `>=1.24.3`.** The old floor permitted versions
  that have no `pymupdf` module at all, so after the rename a resolver landing
  on 1.23.x would fail at import. Confirmed by inspecting wheel contents: 1.24.2
  ships only `fitz`, 1.24.3 ships both.

- **`fastmcp>=0.1.0` raised to `>=2.11.2`.** That floor was a fiction. This
  package imports `fastmcp.contrib.mcp_mixin`, which does not exist before 2.x,
  so the resolver was free to pick a version where the server could not import.

### Changed

- **FastMCP 2.11.2 to 4.0.5, PyMuPDF 1.26.3 to 1.28.2** in the lockfile. Both
  were behind what fresh installs actually resolved, which is the skew that hid
  the deprecation warning. All 54 tools still register and a 12-tool functional
  pass is clean under 4.0.5.

- `tests/test_server.py` reads tool names through whichever listing API is
  present. FastMCP 4.x replaced `get_tools()` (a name-keyed dict) with an async
  `list_tools()` returning `Sequence[Tool]`, so the suite spans both majors
  rather than pinning the project to one.

- PyMuPDF 1.28.2 also clears the three `SwigPyObject has no __module__`
  DeprecationWarnings that every test run has been emitting.

---

## 2026.09.21

First CalVer release. Contains one breaking change and a round of fixes from a
code review of the XFA work.

### Breaking

- **Removed the `mcp-pdf-legacy` and `mcp-pdf-modular` console scripts.** Only
  `mcp-pdf` remains. Both removed entry points ran stale code with live
  defects: `mcp-pdf-modular` used an older mixin package carrying an
  `UnboundLocalError`, and `mcp-pdf-legacy` had a `NameError` in
  `extract_links` (it called a `parse_page_ranges` that is defined nowhere,
  inside an `except ValueError` that cannot catch it). If you invoke either
  script, switch to `mcp-pdf`, which has 52 tools against the legacy server's
  39.

- **Removed `add_video_notes`.** It existed only in the legacy server and was
  broken on two independent paths: it referenced `note_width`/`note_height`
  85 lines before those were assigned, and it rejected a valid MP4 as
  "Invalid or corrupted video file" so it never reached that code anyway.
  Previous documentation described it as a working feature; that was wrong.
  Video annotation may return as a fresh implementation.

- **`extract_form_data` field types** now use the portable vocabulary shared
  with the XFA tools: `text`, `checkbox`, `radio`, `dropdown`, `date`,
  `signature`, plus `button` and `unknown`. `listbox` and `combobox` both
  report as `dropdown`. The original value is preserved on every field as
  `field_type_raw`, since the two differ on free-text entry and multi-select.

- **`is_xfa_pdf` gained a third outcome.** `is_xfa` is now `True`, `False`, or
  `None`, and `None` means detection failed. **Check `detection_failed` before
  trusting `is_xfa`.** Code that treated a falsy `is_xfa` as "this file has no
  XFA" will now see `None` for unreadable files, which is the point: see below.

### Fixed

The review found one defect repeated in six places. Every failure path had been
given a plausible success shape, so the module could not distinguish "I looked
and found nothing" from "I could not look."

- `is_xfa_pdf` caught bare `Exception` and returned `is_xfa=False`. On four
  ordinary damage modes (missing `%%EOF`, stripped xref, garbled trailer,
  truncation) pypdf raises while MuPDF still opens the file, so detection said
  "not XFA", `extract_form_data` proceeded, and the caller got the cryptic
  `"document closed"` that XFA support existed to eliminate, on a file whose
  XFA template was physically present.
- An unknown `profile` silently fell back to generic, which promotes
  producer-internal fields such as `Global_Info-Invisibind-*` into the shared
  cross-form vocabulary. `profile="zipForm"` (the product's real spelling) hit
  this. Profile names are now case-insensitive and unknown values are errors.
- `extra_plumbing_exact` stored caller casing but matched lowercased, so any
  mixed-case entry silently did nothing.
- A corrupt XFA template was reported as a form with zero fields, alongside a
  warning naming three possible causes that excluded the real one. It now
  raises `XfaTemplateParseError`.
- `exclGroup`, XFA's radio-group construct, typed as `text`, rendering a
  two-option radio group as a free-text field. Now `radio`.
- Unmapped XFA UI types defaulted to `text` while the AcroForm side returned
  `unknown`, so two tools declared aligned on one vocabulary disagreed on the
  unknown case, and XFA's direction presented a guess as a fact. Both now
  return `unknown`.
- `_parse_dim` treated an unrecognized unit as points, turning `"1.5inch"`
  into 1.5pt, a 72x error with no warning. Added `mp` (millipoints, legal XFA,
  previously 1000x wrong), `.5in` and `+1in` forms, and a magnitude bound.
- Design-time bbox emitted the parent subform's offset for a field declaring
  only width and height, which is a concrete-looking answer where "unknown" is
  the truth. Added the `page` key with a per-`pageArea` offset reset, so
  multi-page coordinates are interpretable.
- `extract_xfa_fields`'s error path omitted `is_xfa`, so `result["is_xfa"]`
  raised `KeyError` exactly when the call failed. `success`, `is_xfa` and
  `xfa_type` are now on every response.
- The XFA probe in `analyze_pdf_health` ran after the analysis completed,
  inside the same `try`, so a pypdf failure on a file MuPDF reads fine would
  discard a successful result. Now locally guarded.
- `UnboundLocalError` in `mixins/table_extraction.py` (package since removed).
- Three `F821` undefined `canvas` names in `permit_forms.py`, annotations
  referencing a function-local import. Harmless under
  `from __future__ import annotations` but a real `NameError` for anything
  calling `typing.get_type_hints()` on them.
- `uv run pytest` had been aborting at collection since the mixin refactor.
  `tests/test_server.py` imported nine names that had moved into
  `mixins_official`, and pytest treats a collection `ImportError` as fatal, so
  the documented test command ran zero tests.
- Three hardcoded version strings that had drifted apart: `__init__.py` claimed
  `0.1.0`, `server_info` reported `2.0.12`, and `main()`'s fallback said
  `2.1.0`, while the package was on `2.3.1`. All now single-sourced from
  package metadata via `mcp_pdf._version`.
- `[tool.hatchling.build.targets.sdist]` was a dead table (hatchling reads
  `tool.hatch`), so its config had never applied.

### Added

- `canonical_collisions` in `extract_xfa_fields`, reporting distinct XFA names
  that canonicalize to one key. That key is the cross-form join, so a collision
  writes one human answer into two different slots.
- `categories` counts, and documentation for the fourth category, `other`. It
  is the default bucket and holds the majority of fields on any non-zipForm
  producer, so callers branching on the three previously-documented categories
  were dropping most of a non-zipForm schema.
- Bounds on XFA template size, recursion depth and inline field count, all
  env-tunable (`MCP_PDF_MAX_XFA_TEMPLATE_BYTES`, `MCP_PDF_MAX_XFA_DEPTH`,
  `MCP_PDF_MAX_XFA_INLINE_FIELDS`). expat caps entity amplification relative to
  input size, so bounding input is what bounds expansion.
- `field_type_raw` on `extract_form_data` fields.

### Changed

- Both XFA tools offload their blocking pypdf and ElementTree work via
  `run_in_executor`, so one large form no longer stalls every other request.
- `xfa_parts` preserves file order instead of sorting alphabetically.
- Tests went from 31 to 99 for XFA, and the balance inverted. The previous
  suite had one failure-path test, which passed *because* of the bare-except
  defect and so certified it. Now covered: four damage modes, corrupt and
  oversize and deeply-nested templates, entity expansion, XXE, unknown
  profiles, uppercase plumbing entries, `exclGroup`, dimension parsing,
  collisions, truncation, and every response-shape contract key. Field types
  previously had zero assertions anywhere despite being the stated reason for
  the release that introduced them.
- Removed 12,681 lines. Wheel 200K to 112K, sdist 323K to 252K. Ruff findings
  in `src/` 139 to 19, undefined names 7 to 0.

---

## Earlier releases (semver)

- **2.3.0** XFA (dynamic Adobe LiveCycle) form support: `is_xfa_pdf`,
  `extract_xfa_fields`, detection wired into `extract_form_data`,
  `convert_to_images` and `analyze_pdf_health`.
- **2.2.1** Stripped operator-private paths from the sdist and added
  `[tool.hatch.build.targets.sdist]` exclusions to prevent recurrence.
- **2.2.0** `markdown_to_pdf` via pandoc, with PDF-engine auto-detection across
  xelatex, pdflatex, tectonic, weasyprint and wkhtmltopdf.
- **2.1.x** Chapter-aware extraction (`detect_structure`,
  `split_pdf_by_structure`, `batch_extract`), file-first output for large
  responses, optional `camelot`/`tabula` extras.
- **2.0.x** `extract_vector_graphics` (PDF to SVG), configurable size limit via
  `MCP_PDF_MAX_SIZE`.
- **1.x** Initial release under the `mcp-pdf` name, FastMCP mixin architecture,
  core extraction and form tooling.
