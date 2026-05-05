# Quick Start Guide

## 1. Installation

### Option A: Run from PyPI with uvx (Recommended for end users)

No clone required — `uvx` fetches and runs in an isolated cached venv:

```bash
# Bare install
uvx mcp-pdf

# With markdown_to_pdf support (requires pandoc on host)
uvx --from "mcp-pdf[markdown]" mcp-pdf

# Force a refresh after a new release
uvx --refresh --from "mcp-pdf[markdown]" mcp-pdf
```

### Option B: pip install from PyPI

```bash
pip install mcp-pdf
# Or with optional extras:
pip install "mcp-pdf[markdown]"   # adds markdown_to_pdf
pip install "mcp-pdf[forms]"      # adds form creation tools
pip install "mcp-pdf[tables]"     # adds Camelot/Tabula table extraction
pip install "mcp-pdf[all]"        # everything
```

### Option C: Local development with uv

```bash
# Clone the repository
git clone https://github.com/rsp2k/mcp-pdf
cd mcp-pdf

# Install with uv
uv sync

# Verify installation
uv run python examples/verify_installation.py
```

### Option D: Using Docker

```bash
git clone https://github.com/rsp2k/mcp-pdf
cd mcp-pdf

docker compose build
docker compose run --rm mcp-pdf python examples/verify_installation.py
```

## 2. System Dependencies

`uvx` and `pip` only handle Python deps. Some tools call out to system binaries that you'll need to install separately:

| Binary | Required for |
|--------|-------------|
| `tesseract` | `ocr_pdf` |
| `ghostscript` | Camelot table extraction |
| `java` (JRE) | Tabula table extraction |
| `poppler` | PDF→image conversion |
| `pandoc` | `markdown_to_pdf` |
| `xelatex` / `pdflatex` / `tectonic` / `weasyprint` / `wkhtmltopdf` | `markdown_to_pdf` (need at least one) |

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    tesseract-ocr tesseract-ocr-eng \
    poppler-utils ghostscript \
    python3-tk default-jre-headless

# For markdown_to_pdf
sudo apt-get install -y pandoc texlive-xetex
```

### Arch Linux
```bash
sudo pacman -S \
    tesseract tesseract-data-eng \
    poppler ghostscript \
    jre-openjdk-headless tk

# For markdown_to_pdf
sudo pacman -S pandoc texlive-xetex
# Lighter alternative engines: tectonic (official repo),
# wkhtmltopdf (AUR), or `pip install weasyprint` (works in any venv)
```

### macOS (Homebrew)
```bash
brew install tesseract poppler ghostscript

# For markdown_to_pdf
brew install pandoc
brew install --cask mactex-no-gui   # full TeX with xelatex/pdflatex
# Or lighter:
brew install weasyprint
```

### Windows
- Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Install Poppler: http://blog.alivate.com.au/poppler-windows/
- Install Ghostscript: https://www.ghostscript.com/download/gsdnld.html
- Install Java: https://www.java.com/download/
- Install Pandoc (for `markdown_to_pdf`): https://pandoc.org/installing.html
- Install MiKTeX or wkhtmltopdf for the PDF engine

## 3. Adding to Claude Code / Claude Desktop

### Easiest — `claude mcp add` with uvx

```bash
# Bare
claude mcp add pdf-tools -- uvx mcp-pdf

# With markdown_to_pdf support
claude mcp add pdf-tools -- uvx --from "mcp-pdf[markdown]" mcp-pdf
```

The `--` separator is required so the Claude CLI doesn't try to parse `--from` as one of its own flags.

### Manual config (Claude Desktop)

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `~/.config/Claude/claude_desktop_config.json` (Linux):

```json
{
  "mcpServers": {
    "pdf-tools": {
      "command": "uvx",
      "args": ["--from", "mcp-pdf[markdown]", "mcp-pdf"]
    }
  }
}
```

## 4. Test the Tools

```bash
# Test with a sample PDF
uv run python examples/test_pdf_tools.py /path/to/your/document.pdf
```

## 5. Common Issues

### OCR not working
- Check Tesseract is installed: `tesseract --version`
- Install language packs: `sudo apt-get install tesseract-ocr-[lang]` (Debian) or `sudo pacman -S tesseract-data-[lang]` (Arch)

### Table extraction failing
- Check Java is installed: `java -version`
- For Camelot issues, ensure Ghostscript is installed: `gs --version`

### `markdown_to_pdf` errors
- "pandoc binary not found" → install pandoc (see System Dependencies)
- "No PDF engine found" → install at least one of `xelatex`, `pdflatex`, `tectonic`, `weasyprint`, `wkhtmltopdf`
- "Pandoc died with exitcode 43" + `mktexfmt` errors → your TeX install is missing format files; rebuild with `sudo fmtutil-sys --all` or use a different engine via `pdf_engine="weasyprint"`
- The tool reports `detected_engines` in its response — check that field to see what's actually available

### Large PDF issues
- Process specific pages: `pages="1-10"` or `pages="1,3,5"`
- Increase memory: `export JAVA_OPTS="-Xmx2g"`

## 6. Example Usage in Claude

Once configured, you can ask Claude:

- "Extract text from the PDF at /path/to/document.pdf"
- "Check if /path/to/scan.pdf is a scanned document"
- "Extract all tables from /path/to/report.pdf and format as markdown"
- "Convert /path/to/document.pdf to markdown format"
- "Extract images from the first 5 pages of /path/to/presentation.pdf"
- "Build a PDF from /path/to/notes.md with a table of contents"

## 7. Verify the Built-in Test

Convert this README itself to PDF as a smoke test once everything is wired up:

```python
markdown_to_pdf(
    markdown_path="QUICKSTART.md",
    output_path="/tmp/quickstart.pdf",
    toc=True,
)
```

The response includes `detected_engines` so you can see exactly what's installed on your host.

## Need Help?

- Check the full README.md for detailed documentation
- Run tests: `uv run pytest`
- Enable debug mode: Set `DEBUG=true` in your .env file
