# Claude Desktop MCP Configuration

This document explains how the MCP PDF Tools server has been configured for Claude Desktop.

## Configuration Location

The MCP configuration has been added to:
```
/home/rpm/.config/Claude/claude_desktop_config.json
```

## PDF Tools Server Configuration

The following configuration has been added to your Claude Desktop:

```json
{
  "mcpServers": {
    "pdf-tools": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/rpm/claude/mcp-pdf-tools",
        "run",
        "mcp-pdf-tools"
      ],
      "env": {
        "PDF_TEMP_DIR": "/tmp/mcp-pdf-processing"
      }
    }
  }
}
```

## What This Enables

With this configuration, all your Claude sessions will have access to:

- **extract_text**: Extract text from PDFs with multiple method support
- **extract_tables**: Extract tables from PDFs with intelligent fallbacks
- **extract_images**: Extract and filter images from PDFs
- **extract_metadata**: Get comprehensive PDF metadata and file information
- **get_document_structure**: Analyze PDF structure, outline, and fonts
- **is_scanned_pdf**: Detect if PDFs are scanned/image-based
- **ocr_pdf**: Perform OCR on scanned PDFs with preprocessing
- **pdf_to_markdown**: Convert PDFs to clean markdown format

## Environment Variables

- `PDF_TEMP_DIR`: Set to `/tmp/mcp-pdf-processing` for temporary file processing

## Backup

A backup of your original configuration has been saved to:
```
/home/rpm/.config/Claude/claude_desktop_config.json.backup
```

## Testing

The server has been tested and is working correctly. You can verify it's available in new Claude sessions by checking for the `mcp__pdf-tools__*` functions.

## Troubleshooting

If you encounter issues:

1. **Server not starting**: Check that all dependencies are installed:
   ```bash
   cd /home/rpm/claude/mcp-pdf-tools
   uv sync --dev
   ```

2. **System dependencies missing**: Install required packages:
   ```bash
   sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils ghostscript python3-tk default-jre-headless
   ```

3. **Permission issues**: Ensure temp directory exists:
   ```bash
   mkdir -p /tmp/mcp-pdf-processing
   chmod 755 /tmp/mcp-pdf-processing
   ```

4. **Test server manually**:
   ```bash
   cd /home/rpm/claude/mcp-pdf-tools
   uv run mcp-pdf-tools --help
   ```