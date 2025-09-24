# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP PDF is a FastMCP server that provides comprehensive PDF processing capabilities including text extraction, table extraction, OCR, image extraction, and format conversion. The server is built on the FastMCP framework and provides intelligent method selection with automatic fallbacks.

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

### Security Scanning
```bash
# Check for known vulnerabilities in dependencies
uv run safety check

# Audit Python packages for known vulnerabilities
uv run pip-audit

# Run comprehensive security scan
uv run safety check --json && uv run pip-audit --format=json
```

### Running the Server
```bash
# Run MCP server directly
uv run mcp-pdf

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

1. **Text Extraction**: `extract_text` - Intelligent method selection with automatic chunking for large files
2. **Table Extraction**: `extract_tables` - Auto-fallback through Camelot → pdfplumber → Tabula
3. **OCR Processing**: `ocr_pdf` - Tesseract with preprocessing options
4. **Document Analysis**: `is_scanned_pdf`, `get_document_structure`, `extract_metadata`
5. **Format Conversion**: `pdf_to_markdown` - Clean markdown with MCP resource URIs for images
6. **Image Processing**: `extract_images` - Extract images with custom output paths and clean summary output
7. **Link Extraction**: `extract_links` - Extract all hyperlinks with page filtering and type categorization
8. **PDF Forms**: `extract_form_data`, `create_form_pdf`, `fill_form_pdf`, `add_form_fields` - Complete form lifecycle management
9. **Document Assembly**: `merge_pdfs`, `split_pdf_by_pages`, `reorder_pdf_pages` - PDF manipulation and organization
10. **Annotations & Markup**: `add_sticky_notes`, `add_highlights`, `add_stamps`, `add_video_notes`, `extract_all_annotations` - Collaboration and multimedia review tools

### MCP Client-Friendly Design

**Optimized for MCP Context Management:**
- **Custom Output Paths**: `extract_images` allows users to specify where images are saved
- **Clean Summary Output**: Returns concise extraction summary instead of verbose image metadata
- **Resource URIs**: `pdf_to_markdown` uses `pdf-image://{image_id}` protocol for seamless client integration  
- **Prevents Context Overflow**: Avoids verbose output that fills client message windows
- **User Control**: Flexible output directory support with automatic directory creation

### Intelligent Fallbacks and Token Management

The server implements smart fallback mechanisms:
- Text extraction automatically detects scanned PDFs and suggests OCR
- Table extraction tries multiple methods until tables are found
- All operations include comprehensive error handling with helpful hints

**Smart Chunking for Large PDFs:**
- Automatic token estimation and overflow prevention
- Page-boundary chunking (default 10 pages per chunk)  
- Intelligent truncation at sentence boundaries when needed
- Clear guidance for accessing subsequent chunks
- Prevents MCP "response too large" errors commonly reported by users

### Dependencies Management

Critical system dependencies:
- **Tesseract OCR**: Required for `ocr_pdf` functionality
- **Java**: Required for Tabula table extraction
- **Ghostscript**: Required for Camelot table extraction
- **Poppler**: Required for PDF to image conversion

### Configuration

Environment variables (optional):
- `TESSDATA_PREFIX`: Tesseract language data location
- `PDF_TEMP_DIR`: Temporary file processing directory (defaults to `/tmp/mcp-pdf-processing`)
- `DEBUG`: Enable debug logging

### Security Features

The server implements comprehensive security hardening:

**Input Validation:**
- File size limits: 100MB for PDFs, 50MB for images
- Page count limits: Max 1000 pages per document
- Path traversal protection for all file operations
- JSON input size limits (10KB) to prevent DoS attacks
- Safe parsing of user inputs with `ast.literal_eval` size limits

**Access Control:**
- Secure output directory validation (restricted to `/tmp`, `/var/tmp`, cache directory)
- URL allowlisting for download operations (configurable via `ALLOWED_DOMAINS`)
- File permission enforcement (0o700 for cache directories, 0o600 for cached files)

**Error Handling:**
- Sanitized error messages to prevent information disclosure
- Removal of sensitive data patterns (file paths, emails, SSNs)
- Generic error responses for failed operations

**Resource Management:**
- Streaming downloads with size checking to prevent memory exhaustion
- Page count validation to prevent resource exhaustion attacks
- Secure temporary file handling with automatic cleanup

**Vulnerability Scanning:**
- Integrated `safety` and `pip-audit` tools for dependency scanning
- GitHub Actions workflow for continuous security monitoring
- Daily automated vulnerability assessments

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

### PDF Form Tools

The server provides comprehensive PDF form capabilities:

**Form Creation (`create_form_pdf`)**:
- Create new interactive PDF forms from scratch
- Support for text fields, checkboxes, dropdowns, and signature fields
- Automatic field positioning with customizable layouts
- Multiple page size options (A4, Letter, Legal)

**Form Filling (`fill_form_pdf`)**:
- Fill existing PDF forms with JSON data
- Intelligent field type handling (text, checkbox, dropdown)
- Optional form flattening (make fields non-editable)
- Comprehensive error reporting for failed field fills

**Form Enhancement (`add_form_fields`)**:
- Add interactive fields to existing PDFs
- Preserve original document content and formatting
- Support for multi-page field placement
- Flexible field positioning and styling

**Form Extraction (`extract_form_data`)**:
- Extract all form fields and their current values
- Identify field types and constraints
- Form validation and structure analysis

### PDF Document Assembly

The server provides comprehensive document organization capabilities:

**PDF Merging (`merge_pdfs`)**:
- Combine multiple PDFs into single document
- Preserve bookmarks with automatic page number adjustment
- Generate table of contents from source filenames
- Optional page numbering for merged documents
- Intelligent error handling for problematic files

**Page Range Splitting (`split_pdf_by_pages`)**:
- Split PDFs by custom page ranges (1-5, 6-10, 11-end)
- Flexible naming patterns with placeholders
- Preserve relevant bookmarks in each split
- Support for single pages and "end" keyword

**Bookmark-Based Splitting (`split_pdf_by_bookmarks`)**:
- Automatically split at bookmark boundaries
- Configurable bookmark levels (chapters vs sections)
- Clean filename generation from bookmark titles
- Preserve document structure in splits

**Page Reordering (`reorder_pdf_pages`)**:
- Rearrange pages in any custom sequence
- Support for page duplication and omission
- Automatic bookmark reference adjustment
- Detailed tracking of page transformations

### PDF Video Annotations

The server provides innovative multimedia annotation capabilities:

**Video Sticky Notes (`add_video_notes`)**:
- Embed video files directly into PDF as attachments
- Create visual sticky notes with play button icons
- Click-to-launch functionality using JavaScript actions
- Smart format validation with FFmpeg conversion suggestions
- Supports multiple video formats (.mp4, .mov, .avi, .mkv, .webm)
- Automatic file size optimization recommendations
- Color-coded video notes with customizable sizes
- Self-contained multimedia PDFs with no external dependencies

**Technical Implementation:**
- Videos embedded as PDF file attachments with unique identifiers
- Screen annotations with JavaScript `exportDataObject` commands
- Compatible with Adobe Acrobat/Reader JavaScript security model
- Automatic video extraction and system player launch
- Visual indicators include play icons and video titles

**Format Optimization:**
- Intelligent format validation and compatibility checking
- Automatic FFmpeg conversion suggestions for unsupported formats
- File size warnings and compression recommendations for large videos
- Optimal settings: MP4 with H.264/AAC codec for maximum compatibility
- Example conversions provided for easy command-line optimization

**Use Cases:**
- Technical documentation with embedded demo videos
- Training materials with interactive multimedia content
- Inspection reports with video evidence
- Collaborative reviews with video explanations
- Educational content with supplementary video materials

### Docker Support
The project includes Docker support with all system dependencies pre-installed, useful for consistent cross-platform development and deployment.

### MCP Integration
Tools are registered using FastMCP decorators and follow MCP protocol standards for tool descriptions and parameter validation.

## Future Enhancement Ideas

Based on comprehensive PDF usage patterns, here are potential high-impact features for future development:

### 🎯 Priority 1: Document Assembly & Merging
- `merge_pdfs` - Combine multiple PDFs with bookmarks preservation
- `split_pdf_by_pages` - Extract specific page ranges
- `split_pdf_by_bookmarks` - Auto-split by chapters/sections
- `insert_pdf_pages` - Insert pages at specific positions
- `reorder_pdf_pages` - Drag-and-drop style page reordering

### 🔒 Priority 2: Digital Signatures & Security
- `add_digital_signature` - Sign with digital certificates
- `verify_pdf_signatures` - Validate signature authenticity
- `add_password_protection` - Encrypt with user/owner passwords
- `remove_pdf_passwords` - Decrypt protected PDFs
- `set_pdf_permissions` - Control print/copy/edit rights
- `redact_sensitive_data` - Black out confidential information

### ✏️ Priority 3: Advanced Annotations & Markup
- `add_sticky_notes` - Comments and reviews
- `add_highlights` - Text highlighting with colors
- `add_stamps` - Approved/Draft/Confidential stamps
- `add_drawings` - Freehand annotations and shapes
- `extract_all_annotations` - Export comments to JSON/CSV

### 🔍 Priority 4: Document Comparison & Analysis
- `compare_pdf_versions` - Visual diff between document versions
- `detect_pdf_changes` - Highlight additions/deletions
- `analyze_reading_order` - Accessibility compliance checking
- `extract_pdf_statistics` - Word count, reading time, complexity
- `detect_pdf_quality_issues` - Scan for structural problems

### 📄 Priority 5: Advanced Content Extraction
- ✅ `extract_links` - All URLs and internal links (IMPLEMENTED)
- `extract_pdf_fonts` - Font usage analysis
- `extract_pdf_colors` - Color palette extraction
- `extract_pdf_layers` - CAD/design layer information
- `convert_pdf_to_formats` - Word, Excel, PowerPoint, HTML conversion

### ⚡ Priority 6: Batch Operations & Automation
- `batch_process_pdfs` - Apply operations to multiple files
- `create_pdf_portfolio` - Combine different file types
- `auto_ocr_detection` - Smart OCR for scanned pages only
- `optimize_pdf_size` - Intelligent compression algorithms
- `standardize_pdf_metadata` - Bulk metadata updates

### 🚀 Innovative Features
- `ai_summarize_pdf` - Generate executive summaries
- `translate_pdf_text` - Multi-language document translation
- `create_pdf_quiz` - Auto-generate questions from content
- `extract_pdf_timeline` - Parse dates and create chronologies
- `analyze_pdf_accessibility` - WCAG compliance checking

### Implementation Notes
- **Document Assembly** features are universally needed and should be prioritized
- **Digital Signatures** provide high enterprise value
- **Batch Operations** essential for automation workflows
- All features should maintain MCP protocol standards and clean output formatting
- Consider user experience and context window optimization for each tool