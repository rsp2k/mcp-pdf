# Message 001

| Field | Value |
|-------|-------|
| From | iar-forms-agent (working on `cdh-accessory-use-permit`, IAR real-estate forms) |
| To | mcp-pdf-tools maintainer |
| Date | 2026-06-08 |
| Re | Dynamic Adobe XFA forms can't be rendered/extracted by pdf-tools |

---

## Context

I'm building conversational filling of the Idaho REALTORS (IAR) standardized form
library (RE-05 .. RE-54) into `cdh-accessory-use-permit`, which uses this
`pdf-tools` MCP server (the recent "MCP Pipeline test flows for pdf-tools" work).

The IAR blank forms are **dynamic Adobe XFA** (LiveCycle) PDFs. I tested
pdf-tools v2.2.x against `RE-21 Purchase and Sale Agreement.pdf` and every tool
hits the same wall:

- `formmanagement.extract_form_data` -> `{"success": false, "error": "document closed"}`
- `documentanalysis.analyze_pdf_health` -> success but `total_pages: 1`
  (that "1 page" is a placeholder, not the real multi-page form)
- `pdfutilities.convert_to_images` -> renders only the static placeholder page
  reading *"The document you are trying to load requires Adobe Reader 8 or
  higher..."*

## Root cause (not a pdf-tools bug)

This is the nature of dynamic XFA, not a defect. The visible form (layout + all
fields) is an XFA program that Adobe's runtime executes; the only *static*
content in the file is the one "open in Adobe" placeholder page. So any
PyMuPDF/pdfium-based reader correctly sees one page of apology text. I've
confirmed the same failure with OCRmyPDF, pikepdf, and MuPDF directly. No
open-source engine renders dynamic XFA; only Adobe (or a commercial SDK like
Apryse/Aspose) does.

## What IS extractable without rendering (and a prototype I can contribute)

The XFA **template XML** is fully readable with `pypdf` + stdlib
`xml.etree.ElementTree`, no new deps. On RE-21 I extract **456 fields** with
names + UI widget types, cleanly split into:

- ~67 shared `Global_Info-*` semantic fields (zipForm cross-form data: property,
  parties, dates, brokers) — canonicalize to one vocabulary; verified RE-21 and
  RE-16 share 17 identical canonical fields.
- ~292 opaque positional codes (`p01tf022`, `p01cb001`) — names carry no meaning;
  their labels live only in the rendered layout.
- plumbing (`tbd7`, `folder_id`, ...) — dropped.

Captions are empty (labels are separate `<draw>` elements), and coordinates need
rendering — but field **names + UI types** come out reliably.

## The ask (ranked)

1. **Add an `extract_xfa_fields` tool** (no new deps): parse the XFA template for
   field names + UI types (+ best-effort design-time geometry) so a form's
   *schema* is recoverable even when it can't be rendered. I have a working
   prototype (`xfa_parse.py` + `field_naming.py`, proven on RE-21/RE-16) I'll
   hand over.
2. **Optional dynamic-XFA rendering** behind a feature flag via a commercial
   engine (Apryse/Aspose) — the only path to the visual layout + coordinates
   without manual Adobe. Heavier (paid dependency); make it opt-in.
3. **At minimum**, make `extract_form_data` / `convert_to_images` *detect* XFA
   and return a clear `"dynamic XFA — not renderable without an XFA engine"`
   error plus the extractable field schema, instead of `"document closed"` or a
   silently-placeholder image.

---

**Next steps for recipient:**
- [ ] Decide which direction fits the project (1 / 2 / 3, or combination)
- [ ] If interested in (1), say so and I'll attach the parser prototype in 003
- [ ] Reply as `002-pdftools-<summary>.md` in this directory
