<div align="center">

# 📄 MCP PDF

<img src="https://img.shields.io/badge/MCP-PDF%20Tools-red?style=for-the-badge&logo=adobe-acrobat-reader" alt="MCP PDF">

**A FastMCP server for PDF processing**

*47 tools for text extraction, OCR, tables, forms, annotations, markdown↔PDF, and more*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.0+-green.svg?style=flat-square)](https://github.com/jlowin/fastmcp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/mcp-pdf?style=flat-square)](https://pypi.org/project/mcp-pdf/)

**Works great with [MCP Office Tools](https://git.supported.systems/MCP/mcp-office-tools)**

</div>

---

## What It Does

MCP PDF extracts content from PDFs using multiple libraries with automatic fallbacks. If one method fails, it tries another.

**Core capabilities:**
- **Text extraction** via PyMuPDF, pdfplumber, or pypdf (auto-fallback)
- **Table extraction** via Camelot, pdfplumber, or Tabula (auto-fallback)
- **OCR** for scanned documents via Tesseract
- **Form handling** - extract, fill, and create PDF forms
- **Document assembly** - merge, split, reorder pages
- **Annotations** - sticky notes, highlights, stamps
- **Vector graphics** - extract to SVG for schematics and technical drawings
- **Format conversion** - PDF ↔ Markdown (PDF→MD via PyMuPDF, MD→PDF via pandoc)

---

## Quick Start

```bash
# Run from PyPI (one-shot, no permanent install)
uvx mcp-pdf

# Add to Claude Code — note the `--` separator before uvx
claude mcp add pdf-tools -- uvx mcp-pdf

# Include the markdown_to_pdf tool (requires pandoc on host)
claude mcp add pdf-tools -- uvx --from "mcp-pdf[markdown]" mcp-pdf
```

> `uvx` caches tool installs aggressively. After upgrading to a new release, force a refresh with `uvx --refresh mcp-pdf` (or `uvx --refresh --from "mcp-pdf[markdown]" mcp-pdf` if you're using extras).

<details>
<summary><b>Development Installation</b></summary>

```bash
git clone https://github.com/rsp2k/mcp-pdf
cd mcp-pdf
uv sync

# System dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils ghostscript

# For markdown_to_pdf — pick one PDF-engine route:
sudo apt-get install pandoc tectonic                                          # recommended (small)
# or:  sudo apt-get install pandoc texlive-xetex texlive-latex-extra          # full TeX
# or:  sudo apt-get install pandoc && pip install weasyprint                  # skip TeX

# Verify
uv run python examples/verify_installation.py
```

</details>

---

## Tools

### Content Extraction

| Tool | What it does |
|------|-------------|
| `extract_text` | Pull text from PDF pages with automatic chunking for large files |
| `extract_tables` | Extract tables to JSON, CSV, or Markdown |
| `extract_images` | Extract embedded images |
| `extract_links` | Get all hyperlinks with page filtering |
| `ocr_pdf` | OCR scanned documents using Tesseract |
| `extract_vector_graphics` | Export vector graphics to SVG (schematics, charts, drawings) |

### Format Conversion

| Tool | What it does |
|------|-------------|
| `pdf_to_markdown` | Convert PDF to markdown preserving structure; extracts images and SVG vectors to disk |
| `markdown_to_pdf` | Convert `.md` files (or inline text) to PDF via pandoc with auto-detected engine |

**`markdown_to_pdf` requires:** `pip install mcp-pdf[markdown]` plus the pandoc binary and at least one PDF engine (`xelatex`, `pdflatex`, `tectonic`, `weasyprint`, or `wkhtmltopdf`) on PATH. The tool auto-detects what's available and uses the highest-quality one. Pass `pdf_engine=` to override or `extra_args=` for raw pandoc options.

### Document Analysis

| Tool | What it does |
|------|-------------|
| `extract_metadata` | Get title, author, creation date, page count, etc. |
| `get_document_structure` | Extract table of contents and bookmarks |
| `analyze_layout` | Detect columns, headers, footers |
| `is_scanned_pdf` | Check if PDF needs OCR |
| `compare_pdfs` | Diff two PDFs by text, structure, or metadata |
| `analyze_pdf_health` | Check for corruption, optimization opportunities |
| `analyze_pdf_security` | Report encryption, permissions, signatures |

### Forms

| Tool | What it does |
|------|-------------|
| `extract_form_data` | Get form field names and values |
| `fill_form_pdf` | Fill form fields from JSON |
| `create_form_pdf` | Create new forms with text fields, checkboxes, dropdowns |
| `add_form_fields` | Add fields to existing PDFs |

### Permit Forms (Coordinate-Based)

For scanned PDFs or forms without interactive fields. Draws text at (x, y) coordinates.

| Tool | What it does |
|------|-------------|
| `fill_permit_form` | Fill any PDF by drawing at coordinates (works with scanned forms) |
| `get_field_schema` | Get field definitions for validation or UI generation |
| `validate_permit_form_data` | Check data against field schema before filling |
| `preview_field_positions` | Generate PDF showing field boundaries (debugging) |
| `insert_attachment_pages` | Insert image/text pages with "See page X" references |

**Requires:** `pip install mcp-pdf[forms]` (adds reportlab dependency)

### Document Assembly

| Tool | What it does |
|------|-------------|
| `merge_pdfs` | Combine multiple PDFs with bookmark preservation |
| `split_pdf_by_pages` | Split by page ranges |
| `split_pdf_by_bookmarks` | Split at chapter/section boundaries |
| `reorder_pdf_pages` | Rearrange pages in custom order |

### Annotations

| Tool | What it does |
|------|-------------|
| `add_sticky_notes` | Add comment annotations |
| `add_highlights` | Highlight text regions |
| `add_stamps` | Add Approved/Draft/Confidential stamps |
| `extract_all_annotations` | Export annotations to JSON |

---

## How Fallbacks Work

The server tries multiple libraries for each operation:

**Text extraction:**
1. PyMuPDF (fastest)
2. pdfplumber (better for complex layouts)
3. pypdf (most compatible)

**Table extraction:**
1. Camelot (best accuracy, requires Ghostscript)
2. pdfplumber (no dependencies)
3. Tabula (requires Java)

If a PDF fails with one library, the next is tried automatically.

---

## Token Management

Large PDFs can overflow MCP response limits. The server handles this:

- **Automatic chunking** splits large documents into page groups
- **Table row limits** prevent huge tables from blowing up responses
- **Summary mode** returns structure without full content

```python
# Get first 10 pages
result = await extract_text("huge.pdf", pages="1-10")

# Limit table rows
tables = await extract_tables("data.pdf", max_rows_per_table=50)

# Structure only
tables = await extract_tables("data.pdf", summary_only=True)
```

---

## URL Processing

PDFs can be fetched directly from HTTPS URLs:

```python
result = await extract_text("https://example.com/report.pdf")
```

Files are cached locally for subsequent operations.

---

## System Dependencies

Some features require system packages:

| Feature | Dependency |
|---------|-----------|
| OCR | `tesseract-ocr` |
| Camelot tables | `ghostscript` |
| Tabula tables | `default-jre-headless` |
| PDF to images | `poppler-utils` |
| `markdown_to_pdf` | `pandoc` + one of: `tectonic`, `texlive-xetex` (+ `texlive-latex-extra`), `weasyprint`, `wkhtmltopdf` |

### Picking a PDF engine for `markdown_to_pdf`

Pandoc takes markdown → HTML or LaTeX → PDF. The LaTeX path produces the most polished output but needs a TeX install. Trade-offs:

| Engine | Disk size | Notes |
|--------|----------|-------|
| **`tectonic`** | ~30 MB | **Recommended for new installs.** Single static binary. Downloads LaTeX packages on demand — no upfront mass-install. |
| `xelatex` + `texlive-latex-extra` | ~500 MB | Best output once installed. Use if you already run TeX. The `-extra` package matters: pandoc's default template needs `lastpage`, `xcolor`, `framed`, `fancyhdr`, etc. — all of which live there, **not** in `texlive-xetex`. |
| `xelatex` alone (just `texlive-xetex`) | ~200 MB | **Often breaks.** Expect `! LaTeX Error: File 'X.sty' not found` on real docs. |
| `weasyprint` | ~40 MB | Pure-Python (`pip install weasyprint`) + cairo/pango system libs. HTML/CSS path — no LaTeX. Good for simple docs; weaker on math, footnotes, citations. |
| `wkhtmltopdf` | ~40 MB | Older HTML-to-PDF tool. Adequate but less actively maintained. |

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils ghostscript default-jre-headless

# For markdown_to_pdf — pick one engine route:

# Option A — tectonic (smallest, downloads packages on demand)
sudo apt-get install pandoc
# tectonic isn't in apt — install via cargo or download static binary:
#   https://tectonic-typesetting.github.io/en-US/install.html

# Option B — full TeX (best quality, large download)
sudo apt-get install pandoc texlive-xetex texlive-latex-extra texlive-fonts-extra

# Option C — weasyprint (skip TeX entirely)
sudo apt-get install pandoc
pip install weasyprint
```

**Arch Linux:**
```bash
sudo pacman -S tesseract tesseract-data-eng poppler ghostscript jre-openjdk-headless

# For markdown_to_pdf — pick one engine route:

# Option A — tectonic (recommended for new installs, in official repo)
sudo pacman -S pandoc tectonic

# Option B — full TeX (best output, ~500 MB)
sudo pacman -S pandoc texlive-xetex texlive-latexextra texlive-fontsextra

# Option C — weasyprint (skip TeX)
sudo pacman -S pandoc
pip install weasyprint   # or: uv pip install weasyprint

# Option D — wkhtmltopdf (from AUR)
yay -S wkhtmltopdf-static
```

**macOS (Homebrew):**
```bash
brew install tesseract poppler ghostscript

# For markdown_to_pdf — pick one engine route:

# Option A — tectonic (recommended)
brew install pandoc tectonic

# Option B — full TeX (mactex-no-gui includes the latex-extra equivalent)
brew install pandoc
brew install --cask mactex-no-gui

# Option C — weasyprint
brew install pandoc weasyprint
```

## Optional Extras

The base install stays lean. Heavy or niche dependencies are gated behind extras:

| Extra | Adds | When to install |
|-------|------|----------------|
| `mcp-pdf[forms]` | `reportlab` | Form creation tools (`create_form_pdf`, permit forms) |
| `mcp-pdf[tables]` | `camelot-py`, `tabula-py` | Higher-accuracy table extraction (also needs Java + Ghostscript) |
| `mcp-pdf[markdown]` | `pypandoc` | `markdown_to_pdf` tool (also needs pandoc binary) |
| `mcp-pdf[all]` | All of the above | Want everything |

---

## Configuration

Optional environment variables:

| Variable | Purpose |
|----------|---------|
| `MCP_PDF_ALLOWED_PATHS` | Colon-separated directories for file output |
| `PDF_TEMP_DIR` | Temp directory for processing (default: `/tmp/mcp-pdf-processing`) |
| `TESSDATA_PREFIX` | Tesseract language data location |

---

## Development

```bash
# Run tests
uv run pytest

# With coverage
uv run pytest --cov=mcp_pdf

# Format
uv run black src/ tests/

# Lint
uv run ruff check src/ tests/
```

---

## License

MIT

</div>
