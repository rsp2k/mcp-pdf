# 🔧 Local Development Guide for MCP PDF

This guide shows how to test MCP PDF locally during development before publishing to PyPI.

## 📋 Prerequisites

- Python 3.10+
- uv package manager
- Claude Desktop app
- Git repository cloned locally

## 🚀 Quick Start for Local Testing

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/rsp2k/mcp-pdf.git
cd mcp-pdf

# Install dependencies
uv sync --dev

# Verify installation
uv run python -c "from mcp_pdf.server import create_server; print('✅ MCP PDF loads successfully')"
```

### 2. Add MCP Server to Claude Desktop

#### For Production Use (PyPI Installation)

Install the published version from PyPI:

```bash
# For personal use across all projects
claude mcp add -s local pdf-tools uvx mcp-pdf

# For project-specific use (isolated to current directory)
claude mcp add -s project pdf-tools uvx mcp-pdf
```

#### For Local Development (Source Installation)

When developing MCP PDF itself, use the local source:

```bash
# For development from local source
claude mcp add -s project pdf-tools-dev uv -- --directory /path/to/mcp-pdf-tools run mcp-pdf
```

Or if you're in the mcp-pdf directory:

```bash
# Development server from current directory
claude mcp add -s project pdf-tools-dev uv -- --directory . run mcp-pdf
```

### 3. Alternative: Manual Server Testing

You can also run the server manually for debugging:

```bash
# Run the MCP server directly
uv run mcp-pdf

# Or run with specific FastMCP options
uv run python -m mcp_pdf.server
```

### 4. Test Core Functionality

Once connected to Claude Code, test these key features:

#### Basic PDF Processing
```
"Extract text from this PDF file: /path/to/test.pdf"
"Get metadata from this PDF: /path/to/document.pdf"
"Check if this PDF is scanned: /path/to/scan.pdf"
```

#### Security Features
```
"Try to extract text from a very large PDF"
"Process a PDF with 2000 pages" (should be limited to 1000)
```

#### Advanced Features
```
"Extract tables from this PDF: /path/to/tables.pdf"
"Convert this PDF to markdown: /path/to/document.pdf"
"Add annotations to this PDF: /path/to/target.pdf"
```

## 🔒 Security Testing

Verify the security hardening works:

### File Size Limits
- Try processing a PDF larger than 100MB
- Should see: "PDF file too large: X bytes > 104857600"

### Page Count Limits  
- Try processing a PDF with >1000 pages
- Should see: "PDF too large for processing: X pages > 1000"

### Path Traversal Protection
- Test with malicious paths like `../../../etc/passwd`
- Should be blocked with security error

### JSON Input Validation
- Large JSON inputs (>10KB) should be rejected
- Malformed JSON should return clean error messages

## 🐛 Debugging

### Enable Debug Logging
```bash
export DEBUG=true
uv run mcp-pdf
```

### Check Security Functions
```bash
# Test security validation functions
uv run python test_security_features.py

# Run integration tests
uv run python test_integration.py
```

### Verify Package Structure
```bash
# Check package builds correctly
uv build

# Verify package metadata
uv run twine check dist/*
```

## 📊 Testing Checklist

Before publishing, verify:

- [ ] All 23 PDF tools work correctly
- [ ] Security limits are enforced (file size, page count)
- [ ] Error messages are clean and helpful  
- [ ] No sensitive information leaked in errors
- [ ] Path traversal protection works
- [ ] JSON input validation works
- [ ] Memory limits prevent crashes
- [ ] CLI command `mcp-pdf` works
- [ ] Package imports correctly: `from mcp_pdf.server import create_server`

## 🚀 Publishing Pipeline

Once local testing passes:

1. **Version Bump**: Update version in `pyproject.toml`
2. **Build**: `uv build`  
3. **Test Upload**: `uv run twine upload --repository testpypi dist/*`
4. **Test Install**: `pip install -i https://test.pypi.org/simple/ mcp-pdf`
5. **Production Upload**: `uv run twine upload dist/*`

## 🔧 Development Commands

```bash
# Format code
uv run black src/ tests/

# Lint code  
uv run ruff check src/ tests/

# Run tests
uv run pytest

# Security scan
uv run pip-audit

# Build package
uv build

# Install editable for development
pip install -e .  # (in a venv)
```

## 🆘 Troubleshooting

### "Module not found" errors
- Ensure you're in the right directory
- Run `uv sync` to install dependencies
- Check Python path with `uv run python -c "import sys; print(sys.path)"`

### MCP server won't start
- Check that all system dependencies are installed (tesseract, java, ghostscript)
- Verify with: `uv run python examples/verify_installation.py`

### Security tests fail
- Run `uv run python test_security_features.py -v` for detailed output
- Check that security constants are properly set

This setup allows for rapid development and testing without polluting your system Python or needing to publish to PyPI for every change.