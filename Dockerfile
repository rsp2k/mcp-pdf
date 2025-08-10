# MCP PDF Tools Docker Image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # PDF libraries
    poppler-utils \
    # OCR dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    # Image processing
    libmagic1 \
    # Java for Tabula
    default-jre-headless \
    # Build dependencies
    gcc \
    g++ \
    python3-dev \
    # Ghostscript for Camelot
    ghostscript \
    python3-tk \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY examples/ ./examples/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create directory for PDF processing
RUN mkdir -p /tmp/pdf_processing

# Set environment variables
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
ENV PDF_TEMP_DIR=/tmp/pdf_processing
ENV PYTHONUNBUFFERED=1

# Expose the MCP server (stdio)
ENTRYPOINT ["python", "-m", "mcp_pdf_tools.server"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python examples/verify_installation.py || exit 1
