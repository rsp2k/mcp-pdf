# Quick Start Guide

## 1. Installation

### Option A: Using UV (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/rpm/mcp-pdf-tools
cd mcp-pdf-tools

# Install with uv
uv sync

# Verify installation
uv run python examples/verify_installation.py
```

### Option B: Using Docker

```bash
# Clone the repository
git clone https://github.com/rpm/mcp-pdf-tools
cd mcp-pdf-tools

# Build and run with Docker
docker-compose build
docker-compose run --rm mcp-pdf-tools python examples/verify_installation.py
```

### Option C: From PyPI

```bash
pip install mcp-pdf-tools
```

## 2. System Dependencies

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    ghostscript \
    python3-tk \
    default-jre-headless
```

### macOS
```bash
brew install tesseract poppler ghostscript
```

### Windows
- Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Install Poppler: http://blog.alivate.com.au/poppler-windows/
- Install Ghostscript: https://www.ghostscript.com/download/gsdnld.html
- Install Java: https://www.java.com/download/

## 3. Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pdf-tools": {
      "command": "uv",
      "args": ["run", "mcp-pdf-tools"],
      "cwd": "/home/rpm/claude/mcp-pdf-tools"
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
- Install language packs: `sudo apt-get install tesseract-ocr-[lang]`

### Table extraction failing
- Check Java is installed: `java -version`
- For Camelot issues, ensure Ghostscript is installed

### Large PDF issues
- Process specific pages: `pages=[0, 1, 2]`
- Increase memory: `export JAVA_OPTS="-Xmx2g"`

## 6. Example Usage in Claude

Once configured, you can ask Claude:

- "Extract text from the PDF at /path/to/document.pdf"
- "Check if /path/to/scan.pdf is a scanned document"
- "Extract all tables from /path/to/report.pdf and format as markdown"
- "Convert /path/to/document.pdf to markdown format"
- "Extract images from the first 5 pages of /path/to/presentation.pdf"

## Need Help?

- Check the full README.md for detailed documentation
- Run tests: `uv run pytest`
- Enable debug mode: Set `DEBUG=true` in your .env file
