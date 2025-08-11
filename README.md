# MCP PDF Tools

A comprehensive FastMCP server for PDF processing operations. This server provides powerful tools for extracting text, tables, images, and metadata from PDFs, performing OCR on scanned documents, and converting PDFs to various formats.

## Features

- **Text Extraction**: Multiple methods (PyMuPDF, pdfplumber, pypdf) with automatic selection
- **Table Extraction**: Support for both bordered and borderless tables using Camelot, Tabula, and pdfplumber
- **OCR**: Process scanned PDFs with Tesseract OCR, including preprocessing for better results
- **Document Analysis**: Extract structure, metadata, and check if PDFs are scanned
- **Image Extraction**: Extract images with size filtering
- **Format Conversion**: Convert PDFs to clean Markdown format
- **URL Support**: Process PDFs directly from HTTPS URLs with intelligent caching
- **Smart Detection**: Automatically detect the best method for each operation

## URL Support

All tools support processing PDFs directly from HTTPS URLs:

```bash
# Extract text from URL
mcp_pdf_tools extract_text "https://example.com/document.pdf"

# Extract tables from URL  
mcp_pdf_tools extract_tables "https://example.com/report.pdf"

# Convert URL PDF to markdown
mcp_pdf_tools pdf_to_markdown "https://example.com/paper.pdf"
```

**Features:**
- **Intelligent caching**: Downloaded PDFs are cached for 1 hour to avoid repeated downloads
- **Content validation**: Verifies content is actually a PDF file (checks magic bytes and content-type)
- **Security**: HTTPS URLs recommended (HTTP URLs show security warnings)
- **Proper headers**: Sends appropriate User-Agent for better server compatibility
- **Error handling**: Clear error messages for network issues or invalid content

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/rpm/mcp-pdf-tools
cd mcp-pdf-tools

# Install with uv
uv sync

# Install Tesseract OCR (required for OCR functionality)
# On Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# On macOS:
brew install tesseract

# On Windows:
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Using pip

```bash
pip install mcp-pdf-tools

# Install system dependencies for OCR
# Same as above for Tesseract
```

## Configuration

### Claude Desktop Integration

Add to your Claude configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "pdf-tools": {
      "command": "uv",
      "args": ["run", "mcp-pdf-tools"],
      "cwd": "/path/to/mcp-pdf-tools"
    }
  }
}
```

Or if installed via pip:

```json
{
  "mcpServers": {
    "pdf-tools": {
      "command": "mcp-pdf-tools"
    }
  }
}
```

### Claude Code Integration

For development with Claude Code, add the MCP server from your local development directory:

```bash
claude mcp add pdf-tools "uvx --from /path/to/mcp-pdf-tools mcp-pdf-tools"
```

### Environment Variables

Create a `.env` file in your project directory:

```bash
# Optional: Tesseract configuration
TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Optional: Temporary file directory
PDF_TEMP_DIR=/tmp/pdf_processing

# Optional: Enable debug logging
DEBUG=true
```

## Usage Examples

### Text Extraction

```python
# Basic text extraction
result = await extract_text(
    pdf_path="/path/to/document.pdf"
)

# Extract specific pages with layout preservation
result = await extract_text(
    pdf_path="/path/to/document.pdf",
    pages=[0, 1, 2],  # First 3 pages
    preserve_layout=True,
    method="pdfplumber"  # Or "auto", "pymupdf", "pypdf"
)
```

### Table Extraction

```python
# Extract all tables
result = await extract_tables(
    pdf_path="/path/to/document.pdf"
)

# Extract tables from specific pages in markdown format
result = await extract_tables(
    pdf_path="/path/to/document.pdf",
    pages=[2, 3],
    output_format="markdown"  # Or "json", "csv"
)
```

### OCR for Scanned PDFs

```python
# Basic OCR
result = await ocr_pdf(
    pdf_path="/path/to/scanned.pdf"
)

# OCR with multiple languages and preprocessing
result = await ocr_pdf(
    pdf_path="/path/to/scanned.pdf",
    languages=["eng", "fra", "deu"],
    preprocess=True,
    dpi=300
)
```

### Document Analysis

```python
# Check if PDF is scanned
result = await is_scanned_pdf(
    pdf_path="/path/to/document.pdf"
)

# Get document structure and metadata
result = await get_document_structure(
    pdf_path="/path/to/document.pdf"
)

# Extract comprehensive metadata
result = await extract_metadata(
    pdf_path="/path/to/document.pdf"
)
```

### Format Conversion

```python
# Convert to Markdown
result = await pdf_to_markdown(
    pdf_path="/path/to/document.pdf",
    include_images=True,
    include_metadata=True
)
```

### Image Extraction

```python
# Extract images with size filtering
result = await extract_images(
    pdf_path="/path/to/document.pdf",
    min_width=200,
    min_height=200,
    output_format="png"  # Or "jpeg"
)
```

### Advanced Analysis

```python
# Analyze document health and quality
result = await analyze_pdf_health(
    pdf_path="/path/to/document.pdf"
)

# Classify content type and structure
result = await classify_content(
    pdf_path="/path/to/document.pdf"
)

# Generate content summary
result = await summarize_content(
    pdf_path="/path/to/document.pdf",
    summary_length="medium",  # "short", "medium", "long"
    pages="1,2,3"  # Specific pages
)

# Analyze page layout
result = await analyze_layout(
    pdf_path="/path/to/document.pdf",
    pages="1,2,3",
    include_coordinates=True
)
```

### Content Manipulation

```python
# Extract form data
result = await extract_form_data(
    pdf_path="/path/to/form.pdf"
)

# Split PDF into separate files
result = await split_pdf(
    pdf_path="/path/to/document.pdf",
    split_pages="5,10,15",  # Split after pages 5, 10, 15
    output_prefix="section"
)

# Merge multiple PDFs
result = await merge_pdfs(
    pdf_paths=["/path/to/doc1.pdf", "/path/to/doc2.pdf"],
    output_filename="merged_document.pdf"
)

# Rotate specific pages
result = await rotate_pages(
    pdf_path="/path/to/document.pdf",
    page_rotations={"1": 90, "3": 180}  # Page 1: 90°, Page 3: 180°
)
```

### Optimization and Repair

```python
# Optimize PDF file size
result = await optimize_pdf(
    pdf_path="/path/to/large.pdf",
    optimization_level="balanced",  # "light", "balanced", "aggressive"
    preserve_quality=True
)

# Repair corrupted PDF
result = await repair_pdf(
    pdf_path="/path/to/corrupted.pdf"
)

# Compare two PDFs
result = await compare_pdfs(
    pdf_path1="/path/to/original.pdf",
    pdf_path2="/path/to/modified.pdf",
    comparison_type="all"  # "text", "structure", "metadata", "all"
)
```

### Visual Analysis

```python
# Extract charts and diagrams
result = await extract_charts(
    pdf_path="/path/to/report.pdf",
    pages="2,3,4",
    min_size=150  # Minimum size for chart detection
)

# Detect watermarks
result = await detect_watermarks(
    pdf_path="/path/to/document.pdf"
)

# Security analysis
result = await analyze_pdf_security(
    pdf_path="/path/to/document.pdf"
)
```

## Available Tools

### Core Processing Tools
| Tool | Description |
|------|-------------|
| `extract_text` | Extract text with multiple methods and layout preservation |
| `extract_tables` | Extract tables in various formats (JSON, CSV, Markdown) |
| `ocr_pdf` | Perform OCR on scanned PDFs with preprocessing |
| `extract_images` | Extract images with filtering options |
| `pdf_to_markdown` | Convert PDF to clean Markdown format |

### Document Analysis Tools  
| Tool | Description |
|------|-------------|
| `is_scanned_pdf` | Check if a PDF is scanned or text-based |
| `get_document_structure` | Extract document structure, outline, and basic metadata |
| `extract_metadata` | Extract comprehensive metadata and file statistics |
| `analyze_pdf_health` | Comprehensive PDF health and quality analysis |
| `analyze_pdf_security` | Analyze PDF security features and potential issues |
| `classify_content` | Classify and analyze PDF content type and structure |
| `summarize_content` | Generate summary and key insights from PDF content |

### Layout and Visual Analysis Tools
| Tool | Description |
|------|-------------|
| `analyze_layout` | Analyze PDF page layout including text blocks, columns, and spacing |
| `extract_charts` | Extract and analyze charts, diagrams, and visual elements |
| `detect_watermarks` | Detect and analyze watermarks in PDF |

### Content Manipulation Tools
| Tool | Description |
|------|-------------|
| `extract_form_data` | Extract form fields and their values from PDF forms |
| `split_pdf` | Split PDF into multiple files at specified pages |
| `merge_pdfs` | Merge multiple PDFs into a single file |
| `rotate_pages` | Rotate specific pages by 90, 180, or 270 degrees |

### Utility and Optimization Tools
| Tool | Description |
|------|-------------|
| `compare_pdfs` | Compare two PDFs for differences in text, structure, and metadata |
| `convert_to_images` | Convert PDF pages to image files |
| `optimize_pdf` | Optimize PDF file size and performance |
| `repair_pdf` | Attempt to repair corrupted or damaged PDF files |

## Development

### Setup Development Environment

```bash
# Clone and enter directory
git clone https://github.com/rpm/mcp-pdf-tools
cd mcp-pdf-tools

# Install with development dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black src/ tests/
uv run ruff check src/ tests/
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=mcp_pdf_tools

# Run specific test
uv run pytest tests/test_server.py::test_extract_text
```

### Building for PyPI

```bash
# Build the package
uv build

# Upload to PyPI (requires credentials)
uv publish
```

## Troubleshooting

### OCR Not Working

1. **Tesseract not installed**: Make sure Tesseract is installed on your system
2. **Language data missing**: Install additional language packs:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr-fra tesseract-ocr-deu
   
   # macOS
   brew install tesseract-lang
   ```

### Table Extraction Issues

1. **Java not found**: Tabula requires Java. Install Java 8 or higher.
2. **Camelot dependencies**: Install system dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-tk ghostscript
   
   # macOS
   brew install ghostscript tcl-tk
   ```

### Memory Issues with Large PDFs

For very large PDFs, consider:
1. Processing specific page ranges instead of the entire document
2. Increasing available memory for Python
3. Using the streaming capabilities of pdfplumber for text extraction

## Architecture

The server uses intelligent fallback mechanisms:

1. **Text Extraction**: Automatically detects if a PDF is scanned and suggests OCR
2. **Table Extraction**: Tries multiple methods (Camelot → pdfplumber → Tabula) until tables are found
3. **Error Handling**: Graceful degradation with informative error messages

## Performance Tips

- For large PDFs, process in chunks using page ranges
- Use `method="pymupdf"` for fastest text extraction
- For complex tables, start with `method="camelot"`
- Enable preprocessing for better OCR results on poor quality scans

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

This MCP server leverages several excellent PDF processing libraries:
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for fast PDF operations
- [pdfplumber](https://github.com/jsvine/pdfplumber) for layout-aware extraction
- [Camelot](https://github.com/camelot-dev/camelot) for table extraction
- [Tabula-py](https://github.com/chezou/tabula-py) for Java-based table extraction
- [Tesseract](https://github.com/tesseract-ocr/tesseract) for OCR functionality
