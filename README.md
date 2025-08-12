# MCP PDF Tools: A Complete PDF Processing Powerhouse

*From basic text extraction to AI-powered document intelligence - 23 comprehensive tools for every PDF processing need*

---

## 🚀 What We Built

MCP PDF Tools has evolved from a simple 8-tool PDF processor into a **comprehensive 23-tool document intelligence platform**. Whether you're extracting tables from financial reports, analyzing document security, or building automated workflows, we've got you covered.

**🎯 Perfect for:**
- **Business Intelligence**: Financial report analysis, data extraction, document comparison
- **Academic Research**: Paper analysis, citation extraction, content summarization  
- **Document Security**: Security assessment, watermark detection, integrity verification
- **Automated Workflows**: Form processing, document splitting/merging, batch optimization

## ✨ Key Innovations

### 🧠 **Document Intelligence**
Go beyond simple extraction with AI-powered analysis:
- **Smart Classification**: Automatically detect document types (academic, legal, financial, etc.)
- **Intelligent Summarization**: Extract key insights and generate summaries
- **Content Analysis**: Topic extraction, language detection, complexity assessment
- **Quality Assessment**: Comprehensive health checks and optimization recommendations

### 📐 **Advanced Layout Processing**
Understand document structure, not just content:
- **Layout Analysis**: Column detection, reading order, text block analysis
- **Visual Element Extraction**: Charts, diagrams, and image processing
- **Watermark Detection**: Identify and analyze document watermarks
- **Form Processing**: Extract interactive form fields and values

### 🔧 **Professional Document Operations**
Handle complex document workflows:
- **Intelligent Splitting/Merging**: Precise page-level control
- **Security Analysis**: Encryption, permissions, vulnerability assessment
- **Document Repair**: Recover corrupted or damaged PDFs
- **Smart Optimization**: Multi-level compression with quality preservation

### 🌐 **Modern Web Integration**
Process PDFs from anywhere:
- **HTTPS URL Support**: Direct processing from web URLs
- **Intelligent Caching**: 1-hour smart caching to avoid repeated downloads
- **Content Validation**: Automatic PDF format verification
- **User-Friendly**: 1-based page numbering (page 1 = first page, not page 0!)

## 📊 Complete Tool Suite (23 Tools)

### 🔧 **Core Processing Tools**
| Tool | Description |
|------|-------------|
| `extract_text` | Multi-method text extraction with layout preservation |
| `extract_tables` | Intelligent table extraction (JSON, CSV, Markdown) |
| `ocr_pdf` | Advanced OCR with preprocessing for scanned documents |
| `extract_images` | Image extraction with size filtering and format options |
| `pdf_to_markdown` | Clean markdown conversion with structure preservation |

### 🧠 **Document Analysis & Intelligence**
| Tool | Description |
|------|-------------|
| `classify_content` | AI-powered document type classification and analysis |
| `summarize_content` | Intelligent summarization with key insights extraction |
| `analyze_pdf_health` | Comprehensive quality assessment and optimization suggestions |
| `analyze_pdf_security` | Security feature analysis and vulnerability detection |
| `compare_pdfs` | Advanced document comparison (text, structure, metadata) |
| `is_scanned_pdf` | Smart detection of scanned vs. text-based documents |
| `get_document_structure` | Document outline and structural analysis |
| `extract_metadata` | Comprehensive metadata and statistics extraction |

### 📐 **Layout & Visual Analysis**
| Tool | Description |
|------|-------------|
| `analyze_layout` | Page layout analysis with column and spacing detection |
| `extract_charts` | Chart, diagram, and visual element extraction |
| `detect_watermarks` | Watermark detection and analysis |

### 🔨 **Content Manipulation**
| Tool | Description |
|------|-------------|
| `extract_form_data` | Interactive PDF form data extraction |
| `split_pdf` | Intelligent document splitting at specified pages |
| `merge_pdfs` | Multi-document merging with page range tracking |
| `rotate_pages` | Precise page rotation (90°/180°/270°) |

### ⚡ **Optimization & Utilities**
| Tool | Description |
|------|-------------|
| `convert_to_images` | PDF to image conversion with quality control |
| `optimize_pdf` | Multi-level file size optimization |
| `repair_pdf` | Automated corruption repair and recovery |

## 🎯 Real-World Usage Examples

### 📊 Business Intelligence Workflow
```python
# Comprehensive financial report analysis
health = await analyze_pdf_health("quarterly-report.pdf")
classification = await classify_content("quarterly-report.pdf") 
summary = await summarize_content("quarterly-report.pdf", summary_length="medium")
tables = await extract_tables("quarterly-report.pdf", pages="5,6,7")
charts = await extract_charts("quarterly-report.pdf")

print(f"Document type: {classification['document_type']}")
print(f"Health score: {health['overall_health_score']}")
print(f"Key insights: {summary['key_insights']}")
```

### 📚 Academic Research Processing
```python
# Process research papers with full analysis
layout = await analyze_layout("research-paper.pdf", pages="1,2,3")
summary = await summarize_content("research-paper.pdf", summary_length="long")
references = await extract_text("research-paper.pdf", pages="15,16,17")
document_health = await analyze_pdf_health("research-paper.pdf")

print(f"Reading complexity: {layout['layout_statistics']['reading_complexity']}")
print(f"Main topics: {summary['key_topics']}")
```

### 🔒 Document Security Assessment
```python
# Comprehensive security analysis
security = await analyze_pdf_security("sensitive-document.pdf")
watermarks = await detect_watermarks("sensitive-document.pdf")
health = await analyze_pdf_health("sensitive-document.pdf")

print(f"Encryption status: {security['encryption']['encryption_type']}")
print(f"Security warnings: {security['security_warnings']}")
print(f"Watermarks detected: {watermarks['has_watermarks']}")
```

### 📋 Automated Form Processing
```python
# Extract and process form data
forms = await extract_form_data("application-form.pdf")
health = await analyze_pdf_health("application-form.pdf")

required_fields = [f for f in forms['form_fields'] if f['is_required']]
filled_fields = [f for f in forms['form_fields'] if f['field_value']]

print(f"Form completion: {len(filled_fields)}/{len(required_fields)} required fields")
```

## 🌐 URL Processing - Work with PDFs Anywhere

All tools support direct HTTPS URL processing:

```python
# Process PDFs directly from the web
await extract_text("https://example.com/report.pdf")
await analyze_layout("https://company.com/whitepaper.pdf", pages="1,2,3")
await extract_tables("https://research.org/data.pdf", output_format="csv")
```

**Advanced URL Features:**
- **Intelligent Caching**: 1-hour cache prevents repeated downloads
- **Content Validation**: Verifies PDF format and integrity  
- **Security Headers**: Proper User-Agent and secure requests
- **Error Handling**: Clear messages for network/content issues

## 🛠 Installation & Setup

### Quick Start
```bash
# Clone and install
git clone https://github.com/rpm/mcp-pdf-tools
cd mcp-pdf-tools
uv sync

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils ghostscript

# Verify installation
uv run python examples/verify_installation.py
```

### Claude Desktop Integration
Add to your Claude configuration (`~/Library/Application Support/Claude/claude_desktop_config.json`):

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

### Claude Code Integration
```bash
claude mcp add pdf-tools "uvx --from /path/to/mcp-pdf-tools mcp-pdf-tools"
```

## 📖 Usage Examples

### Text Extraction with Layout Preservation
```python
# Basic text extraction
result = await extract_text("document.pdf")

# Extract specific pages with layout preservation
result = await extract_text(
    pdf_path="document.pdf",
    pages=[1, 2, 3],  # First 3 pages (1-based numbering)
    preserve_layout=True,
    method="pdfplumber"
)
```

### Advanced Table Extraction
```python
# Extract all tables
result = await extract_tables("document.pdf")

# Extract tables from specific pages in markdown format
result = await extract_tables(
    pdf_path="document.pdf",
    pages=[2, 3],  # Pages 2 and 3 (1-based numbering)
    output_format="markdown"
)
```

### Document Analysis & Intelligence
```python
# Comprehensive document analysis
health = await analyze_pdf_health("document.pdf")
classification = await classify_content("document.pdf")
summary = await summarize_content(
    pdf_path="document.pdf",
    summary_length="medium",
    pages="1,2,3"  # Specific pages (1-based numbering)
)
```

### Content Manipulation
```python
# Split PDF into separate files  
result = await split_pdf(
    pdf_path="document.pdf",
    split_pages="5,10,15",  # Split after pages 5, 10, 15 (1-based)
    output_prefix="section"
)

# Merge multiple PDFs
result = await merge_pdfs(
    pdf_paths=["/path/to/doc1.pdf", "/path/to/doc2.pdf"],
    output_filename="merged_document.pdf"
)

# Rotate specific pages
result = await rotate_pages(
    pdf_path="document.pdf",
    page_rotations={"1": 90, "3": 180}  # Page 1: 90°, Page 3: 180° (1-based)
)
```

### Visual Analysis
```python
# Extract charts and diagrams
result = await extract_charts(
    pdf_path="/path/to/report.pdf",
    pages="2,3,4",  # Pages 2, 3, 4 (1-based numbering)
    min_size=150
)

# Detect watermarks
result = await detect_watermarks("document.pdf")

# Security analysis
result = await analyze_pdf_security("document.pdf")
```

### Optimization & Repair
```python
# Optimize PDF file size
result = await optimize_pdf(
    pdf_path="large-document.pdf",
    optimization_level="balanced",  # "light", "balanced", "aggressive"
    preserve_quality=True
)

# Repair corrupted PDF
result = await repair_pdf("corrupted-document.pdf")
```

## ⚡ Performance & Architecture

### Multi-Library Intelligence
Rather than relying on a single approach, we use intelligent fallback systems:
- **Text Extraction**: PyMuPDF → pdfplumber → pypdf (automatic selection)
- **Table Extraction**: Camelot → pdfplumber → Tabula (tries until success)
- **Smart Detection**: Automatically detects scanned PDFs and suggests OCR

### Async-First Design
All operations are built with modern async/await patterns:
```python
# All tools are fully async
results = await asyncio.gather(
    extract_text("doc1.pdf"),
    analyze_layout("doc2.pdf"),  
    extract_tables("doc3.pdf")
)
```

### Resource Management
- **Memory Efficient**: Streaming processing for large documents
- **Smart Caching**: Intelligent URL caching and resource cleanup
- **Performance Monitoring**: All operations include timing metrics

## 🔧 Development

### Setup Development Environment
```bash
# Install with development dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black src/ tests/ examples/
uv run ruff check src/ tests/ examples/

# Type checking
uv run mypy src/
```

### Quality Standards
- ✅ **100% Lint-Free**: All code passes `ruff` checks
- ✅ **Type Safety**: Comprehensive type hints with `mypy`
- ✅ **Error Handling**: Consistent error patterns across all tools
- ✅ **Documentation**: Clear docstrings and usage examples
- ✅ **Testing**: Comprehensive test coverage

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Test with coverage
uv run pytest --cov=mcp_pdf_tools

# Test specific functionality
uv run pytest tests/test_server.py::test_extract_text

# Verify page numbering (1-based conversion)
uv run python test_pages_parameter.py
```

## 🚀 Advanced Features

### Environment Variables
```bash
# Optional configuration
TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata  # Tesseract data location
PDF_TEMP_DIR=/tmp/pdf_processing                     # Temporary file directory  
DEBUG=true                                           # Enable debug logging
```

### Docker Support
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-eng \
    poppler-utils ghostscript \
    default-jre-headless
# ... rest of Dockerfile
```

## 🔍 Troubleshooting

### OCR Issues
```bash
# Install language packs
sudo apt-get install tesseract-ocr-fra tesseract-ocr-deu

# macOS
brew install tesseract-lang
```

### Table Extraction Issues  
```bash
# Install Java (required for Tabula)
sudo apt-get install default-jre-headless

# Install Ghostscript (required for Camelot)
sudo apt-get install ghostscript
```

### Memory Issues with Large PDFs
- Process specific page ranges: `pages="1,2,3"`
- Use streaming capabilities: `method="pdfplumber"`
- Consider splitting large documents first

## 🏗 Architecture Deep-Dive

### Intelligent Method Selection
```python
# Automatic fallback system
async def extract_text_with_fallback(pdf_path: str):
    try:
        return await extract_with_pymupdf(pdf_path)  # Fast, good for most PDFs
    except Exception:
        try:
            return await extract_with_pdfplumber(pdf_path)  # Layout-aware
        except Exception:
            return await extract_with_pypdf(pdf_path)  # Maximum compatibility
```

### User Experience Design
```python
# Before: Confusing zero-based indexing
pages=[0, 1, 2]  # First 3 pages - not intuitive!

# After: Natural 1-based indexing
pages=[1, 2, 3]  # First 3 pages - makes perfect sense!

# Internal conversion happens automatically
def parse_pages_parameter(pages):
    # Convert 1-based user input to 0-based internal representation
    return [max(0, p - 1) for p in user_pages]
```

## 🤝 Contributing

We welcome contributions! Here's how to get involved:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Add tests** for new functionality
4. **Ensure code quality**: `uv run ruff check && uv run pytest`
5. **Submit a pull request**

### Development Workflow
```bash
# Setup development environment
git clone https://github.com/your-username/mcp-pdf-tools
cd mcp-pdf-tools
uv sync --dev

# Make changes and test
uv run pytest
uv run ruff check src/

# Submit changes
git add .
git commit -m "Add amazing new feature"
git push origin feature/amazing-feature
```

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project leverages several excellent libraries:
- **[PyMuPDF](https://github.com/pymupdf/PyMuPDF)**: Fast PDF operations and rendering
- **[pdfplumber](https://github.com/jsvine/pdfplumber)**: Layout-aware text extraction
- **[Camelot](https://github.com/camelot-dev/camelot)**: Advanced table extraction
- **[Tabula-py](https://github.com/chezou/tabula-py)**: Java-based table extraction
- **[Tesseract](https://github.com/tesseract-ocr/tesseract)**: Industry-standard OCR
- **[FastMCP](https://github.com/phdowling/fastmcp)**: Modern MCP server framework

## 🔗 Links & Resources

- **[GitHub Repository](https://github.com/rpm/mcp-pdf-tools)**
- **[MCP Protocol Documentation](https://modelcontextprotocol.io/)**
- **[FastMCP Framework](https://github.com/phdowling/fastmcp)**
- **[Issue Tracker](https://github.com/rpm/mcp-pdf-tools/issues)**

---

## 🌟 Why MCP PDF Tools?

**🚀 Comprehensive**: 23 specialized tools covering every PDF processing need  
**🧠 Intelligent**: AI-powered analysis and smart method selection  
**🌐 Modern**: HTTPS URL support with intelligent caching  
**👥 User-Friendly**: Intuitive 1-based page numbering and clear APIs  
**🔧 Production-Ready**: Robust error handling and performance optimization  
**📈 Scalable**: Async architecture with efficient resource management  

Whether you're building document analysis pipelines, creating intelligent workflows, or need reliable PDF processing for your applications, MCP PDF Tools provides the comprehensive foundation you need.

**Ready to get started?** Clone the repo and run `uv run python examples/verify_installation.py` to see all 23 tools in action!

---

*Built with ❤️ using modern Python, FastMCP, and the power of intelligent document processing. Questions? Open an issue or contribute - we'd love to hear about your use cases!*