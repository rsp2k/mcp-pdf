# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP PDF Tools is a FastMCP server that provides comprehensive PDF processing capabilities including text extraction, table extraction, OCR, image extraction, and format conversion. The server is built on the FastMCP framework and provides intelligent method selection with automatic fallbacks.

## Development Commands

### Environment Setup
```bash
# Install with development dependencies
uv sync --dev

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils ghostscript python3-tk default-jre-headless
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=mcp_pdf_tools

# Run specific test file
uv run pytest tests/test_server.py

# Run specific test
uv run pytest tests/test_server.py::TestTextExtraction::test_extract_text_success
```

### Code Quality
```bash
# Format code
uv run black src/ tests/ examples/

# Lint code
uv run ruff check src/ tests/ examples/

# Type checking
uv run mypy src/
```

### Running the Server
```bash
# Run MCP server directly
uv run mcp-pdf-tools

# Verify installation
uv run python examples/verify_installation.py

# Test with sample PDF
uv run python examples/test_pdf_tools.py /path/to/test.pdf
```

### Building and Distribution
```bash
# Build package
uv build

# Upload to PyPI (requires credentials)
uv publish
```

## Architecture

### Core Components

- **`src/mcp_pdf_tools/server.py`**: Main server implementation with all PDF processing tools
- **FastMCP Framework**: Uses FastMCP for MCP protocol implementation
- **Multi-library approach**: Integrates PyMuPDF, pdfplumber, pypdf, Camelot, Tabula, and Tesseract

### Tool Categories

1. **Text Extraction**: `extract_text` - Intelligent method selection (PyMuPDF, pdfplumber, pypdf)
2. **Table Extraction**: `extract_tables` - Auto-fallback through Camelot → pdfplumber → Tabula
3. **OCR Processing**: `ocr_pdf` - Tesseract with preprocessing options
4. **Document Analysis**: `is_scanned_pdf`, `get_document_structure`, `extract_metadata`
5. **Format Conversion**: `pdf_to_markdown` - Clean markdown with file-based images (no verbose base64)
6. **Image Processing**: `extract_images` - Size filtering and file-based output (avoids context overflow)

### MCP Client-Friendly Design

**Optimized for MCP Context Management:**
- **Image Processing**: `extract_images` and `pdf_to_markdown` save images to files instead of returning base64 data
- **Prevents Context Overflow**: Avoids verbose output that can fill client message windows
- **File-Based Results**: Returns file paths, dimensions, and metadata instead of raw binary data
- **Human-Readable Sizes**: Includes formatted file sizes (e.g., "1.2 MB") for better user experience

### Intelligent Fallbacks

The server implements smart fallback mechanisms:
- Text extraction automatically detects scanned PDFs and suggests OCR
- Table extraction tries multiple methods until tables are found
- All operations include comprehensive error handling with helpful hints

### Dependencies Management

Critical system dependencies:
- **Tesseract OCR**: Required for `ocr_pdf` functionality
- **Java**: Required for Tabula table extraction
- **Ghostscript**: Required for Camelot table extraction
- **Poppler**: Required for PDF to image conversion

### Configuration

Environment variables (optional):
- `TESSDATA_PREFIX`: Tesseract language data location
- `PDF_TEMP_DIR`: Temporary file processing directory
- `DEBUG`: Enable debug logging

## Development Notes

### Testing Strategy
- Comprehensive unit tests with mocked PDF libraries
- Test fixtures for consistent PDF document simulation
- Error handling tests for all major failure modes
- Server initialization and tool registration validation

### Tool Implementation Pattern
All tools follow this pattern:
1. Validate PDF path using `validate_pdf_path()`
2. Try primary method with intelligent selection
3. Implement fallbacks where applicable
4. Return structured results with metadata
5. Include timing information and method used
6. Provide helpful error messages with troubleshooting hints

### Docker Support
The project includes Docker support with all system dependencies pre-installed, useful for consistent cross-platform development and deployment.

### MCP Integration
Tools are registered using FastMCP decorators and follow MCP protocol standards for tool descriptions and parameter validation.