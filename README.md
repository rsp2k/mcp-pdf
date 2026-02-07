<div align="center">

# 📄 MCP PDF

<img src="https://img.shields.io/badge/MCP-PDF%20Tools-red?style=for-the-badge&logo=adobe-acrobat-reader" alt="MCP PDF">

**A FastMCP server for PDF processing**

*41 tools for text extraction, OCR, tables, forms, annotations, and more*

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

---

## Quick Start

```bash
# Install from PyPI
uvx mcp-pdf

# Or add to Claude Code
claude mcp add pdf-tools uvx mcp-pdf
```

<details>
<summary><b>Development Installation</b></summary>

```bash
git clone https://github.com/rsp2k/mcp-pdf
cd mcp-pdf
uv sync

# System dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils ghostscript

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
| `pdf_to_markdown` | Convert PDF to markdown preserving structure |
| `ocr_pdf` | OCR scanned documents using Tesseract |
| `extract_vector_graphics` | Export vector graphics to SVG (schematics, charts, drawings) |

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

Ubuntu/Debian:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils ghostscript default-jre-headless
```

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
