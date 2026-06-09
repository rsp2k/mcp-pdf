# XFA test fixtures

## `synthetic_dynamic_xfa.pdf`

A minimal, **license-clean** dynamic XFA form for regression-testing XFA
detection and `extract_xfa_fields`. Hand-built; contains **no** copyrighted form
content. ~2 KB.

Structure:
- 1 static "Please wait... requires Adobe Reader 8 or higher" placeholder page
  (what a real dynamic XFA shows to non-Adobe readers — exercises detection).
- Catalog `/NeedsRendering true` — the canonical dynamic-XFA marker.
- XFA packet in **array form** `[template, datasets]` (exercises the
  name/stream-pair parsing path, not the single-stream path).
- An 8-field XFA `<template>` covering all three categories:

| field name | category | exercises |
|---|---|---|
| `Global_Info-Test-Property-Address` | shared | canonicalization -> `test_property_address`; `textEdit` -> `text`; **in-field `<caption>` harvest** ("Property Address") |
| `Global_Info-Test-Buyer-Name` | shared | shared + `checkButton` -> `checkbox` |
| `p01tf001` | positional | opaque code, page-1 text |
| `p01cb002` | positional | opaque code, page-1 checkbox |
| `p02te001` | positional | **different page prefix** — regex must not over-match |
| `tbd1` | plumbing (dropped) | `^tbd\d+$` denylist |
| `folder_id` | plumbing (dropped) | exact-name denylist |
| `Global_Info-Invisibind-Test` | plumbing (dropped) | **denylist precedence** — dropped despite the `Global_Info-` shared prefix |

Expected `classify()` result: **2 shared, 3 positional, 3 plumbing dropped, 1
caption harvested**. (Verified with the prototype `xfa_parse.py` + `field_naming.py`.)
