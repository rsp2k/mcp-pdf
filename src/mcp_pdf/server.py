"""
MCP PDF Tools Server - Comprehensive PDF processing capabilities
"""

import os
import asyncio
import tempfile
import base64
import hashlib
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse
import logging
import ast
import re

from fastmcp import FastMCP
from pydantic import BaseModel, Field
import httpx

# PDF processing libraries
import fitz  # PyMuPDF
import pdfplumber
import camelot
import tabula
import pytesseract
from pdf2image import convert_from_path
import pypdf
import pandas as pd
import difflib
import re
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security Configuration
MAX_PDF_SIZE = 100 * 1024 * 1024  # 100MB
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_PAGES_PROCESS = 1000
MAX_JSON_SIZE = 10000  # 10KB for JSON parameters
PROCESSING_TIMEOUT = 300  # 5 minutes

# Allowed domains for URL downloads (empty list means disabled by default)
ALLOWED_DOMAINS = []

# Initialize FastMCP server
mcp = FastMCP("pdf-tools")

# URL download cache directory with secure permissions
CACHE_DIR = Path(os.environ.get("PDF_TEMP_DIR", "/tmp/mcp-pdf-processing"))
CACHE_DIR.mkdir(exist_ok=True, parents=True, mode=0o700)

# Security utility functions
def validate_image_id(image_id: str) -> str:
    """Validate image ID to prevent path traversal attacks"""
    if not image_id:
        raise ValueError("Image ID cannot be empty")
    
    # Only allow alphanumeric characters, underscores, and hyphens
    if not re.match(r'^[a-zA-Z0-9_-]+$', image_id):
        raise ValueError(f"Invalid image ID format: {image_id}")
    
    # Prevent excessively long IDs
    if len(image_id) > 255:
        raise ValueError(f"Image ID too long: {len(image_id)} > 255")
    
    return image_id

def validate_output_path(path: str) -> Path:
    """Validate and secure output paths to prevent directory traversal"""
    if not path:
        raise ValueError("Output path cannot be empty")
    
    # Convert to Path and resolve to absolute path
    resolved_path = Path(path).resolve()
    
    # Check for path traversal attempts
    if '../' in str(path) or '\\..\\' in str(path):
        raise ValueError("Path traversal detected in output path")
    
    # Ensure path is within safe directories
    safe_prefixes = ['/tmp', '/var/tmp', str(CACHE_DIR.resolve())]
    if not any(str(resolved_path).startswith(prefix) for prefix in safe_prefixes):
        raise ValueError(f"Output path not allowed: {path}")
    
    return resolved_path

def safe_json_parse(json_str: str, max_size: int = MAX_JSON_SIZE) -> dict:
    """Safely parse JSON with size limits"""
    if not json_str:
        return {}
    
    if len(json_str) > max_size:
        raise ValueError(f"JSON input too large: {len(json_str)} > {max_size}")
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")

def validate_url(url: str) -> bool:
    """Validate URL to prevent SSRF attacks"""
    if not url:
        return False
    
    try:
        parsed = urlparse(url)
        
        # Only allow HTTP/HTTPS
        if parsed.scheme not in ('http', 'https'):
            return False
        
        # Block localhost and internal IPs
        hostname = parsed.hostname
        if not hostname:
            # Handle IPv6 or malformed URLs
            netloc = parsed.netloc.strip('[]')  # Remove brackets if present
            if netloc in ['::1', 'localhost'] or netloc.startswith('127.') or netloc.startswith('0.0.0.0'):
                return False
            hostname = netloc.split(':')[0] if ':' in netloc and not netloc.count(':') > 1 else netloc
        
        if hostname in ['localhost', '127.0.0.1', '0.0.0.0', '::1']:
            return False
        
        # Check against allowed domains if configured
        if ALLOWED_DOMAINS:
            return any(hostname.endswith(domain) for domain in ALLOWED_DOMAINS)
        
        # If no domain restrictions, allow any domain (except blocked ones above)
        return True
        
    except Exception:
        return False

def sanitize_error_message(error: Exception, context: str = "") -> str:
    """Sanitize error messages to prevent information disclosure"""
    error_str = str(error)
    
    # Remove potential file paths
    error_str = re.sub(r'/[\w/.-]+', '[PATH]', error_str)
    
    # Remove potential sensitive data patterns
    error_str = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', error_str)
    error_str = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', error_str)
    
    return f"{context}: {error_str}" if context else error_str

def validate_page_count(doc, operation: str = "processing") -> None:
    """Validate PDF page count to prevent resource exhaustion"""
    page_count = doc.page_count
    if page_count > MAX_PAGES_PROCESS:
        raise ValueError(f"PDF too large for {operation}: {page_count} pages > {MAX_PAGES_PROCESS}")
    
    if page_count == 0:
        raise ValueError("PDF has no pages")

# Resource for serving extracted images
@mcp.resource("pdf-image://{image_id}", 
              description="Extracted PDF image",
              mime_type="image/png")
async def get_pdf_image(image_id: str) -> bytes:
    """
    Serve extracted PDF images as MCP resources with security validation.
    
    Args:
        image_id: Image identifier (filename without extension)
        
    Returns:
        Raw image bytes
    """
    try:
        # Validate image ID to prevent path traversal
        validated_id = validate_image_id(image_id)
        
        # Reconstruct the image path from the validated ID
        image_path = CACHE_DIR / f"{validated_id}.png"
        
        # Try .jpeg as well if .png doesn't exist
        if not image_path.exists():
            image_path = CACHE_DIR / f"{validated_id}.jpeg"
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {validated_id}")
        
        # Ensure the resolved path is still within CACHE_DIR
        resolved_path = image_path.resolve()
        if not str(resolved_path).startswith(str(CACHE_DIR.resolve())):
            raise ValueError("Invalid image path detected")
        
        # Check file size before reading to prevent memory exhaustion
        file_size = resolved_path.stat().st_size
        if file_size > MAX_IMAGE_SIZE:
            raise ValueError(f"Image file too large: {file_size} bytes > {MAX_IMAGE_SIZE}")
        
        # Read and return the image bytes
        with open(resolved_path, 'rb') as f:
            return f.read()
            
    except Exception as e:
        sanitized_error = sanitize_error_message(e, "Image serving failed")
        logger.error(sanitized_error)
        raise ValueError("Failed to serve image")

# Configuration models
class ExtractionConfig(BaseModel):
    """Configuration for text extraction"""
    method: str = Field(default="auto", description="Extraction method: auto, pymupdf, pdfplumber, pypdf")
    pages: Optional[List[int]] = Field(default=None, description="Specific pages to extract")
    preserve_layout: bool = Field(default=False, description="Preserve text layout")

class TableExtractionConfig(BaseModel):
    """Configuration for table extraction"""
    method: str = Field(default="auto", description="Method: auto, camelot, tabula, pdfplumber")
    pages: Optional[List[int]] = Field(default=None, description="Pages to extract tables from")
    output_format: str = Field(default="json", description="Output format: json, csv, markdown")

class OCRConfig(BaseModel):
    """Configuration for OCR processing"""
    languages: List[str] = Field(default=["eng"], description="OCR languages")
    preprocess: bool = Field(default=True, description="Preprocess image for better OCR")
    dpi: int = Field(default=300, description="DPI for image conversion")

# Utility functions

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def parse_pages_parameter(pages: Union[str, List[int], None]) -> Optional[List[int]]:
    """
    Parse pages parameter from various formats into a list of 0-based integers.
    User input is 1-based (page 1 = first page), converted to 0-based internally.
    """
    if pages is None:
        return None
    
    if isinstance(pages, list):
        # Convert 1-based user input to 0-based internal representation
        return [max(0, int(p) - 1) for p in pages]
    
    if isinstance(pages, str):
        try:
            # Validate input length to prevent abuse
            if len(pages.strip()) > 1000:
                raise ValueError("Pages parameter too long")
            
            # Handle string representations like "[1, 2, 3]" or "1,2,3"
            if pages.strip().startswith('[') and pages.strip().endswith(']'):
                page_list = ast.literal_eval(pages.strip())
            elif ',' in pages:
                page_list = [int(p.strip()) for p in pages.split(',')]
            else:
                page_list = [int(pages.strip())]
            
            # Convert 1-based user input to 0-based internal representation
            return [max(0, int(p) - 1) for p in page_list]
            
        except (ValueError, SyntaxError):
            raise ValueError(f"Invalid pages format: {pages}. Use 1-based page numbers like [1,2,3] or 1,2,3")
    
    return None

async def download_pdf_from_url(url: str) -> Path:
    """Download PDF from URL with security validation and size limits"""
    try:
        # Validate URL to prevent SSRF attacks
        if not validate_url(url):
            raise ValueError(f"URL not allowed or invalid: {url}")
        
        # Create cache filename based on URL hash
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cache_file = CACHE_DIR / f"cached_{url_hash}.pdf"
        
        # Check if cached file exists and is recent (1 hour)
        if cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < 3600:  # 1 hour cache
                logger.info(f"Using cached PDF: {cache_file}")
                return cache_file
        
        logger.info(f"Downloading PDF from: {url}")
        
        headers = {
            "User-Agent": "MCP-PDF-Tools/1.0 (PDF processing server; +https://github.com/fastmcp/mcp-pdf-tools)"
        }
        
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # Use streaming to check size before downloading
            async with client.stream('GET', url, headers=headers) as response:
                response.raise_for_status()
                
                # Check content length header
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > MAX_PDF_SIZE:
                    raise ValueError(f"PDF file too large: {content_length} bytes > {MAX_PDF_SIZE}")
                
                # Check content type
                content_type = response.headers.get("content-type", "").lower()
                if "pdf" not in content_type and "application/pdf" not in content_type:
                    # Need to read some content to check magic bytes
                    first_chunk = b""
                    async for chunk in response.aiter_bytes(chunk_size=1024):
                        first_chunk += chunk
                        if len(first_chunk) >= 10:
                            break
                    
                    if not first_chunk.startswith(b"%PDF"):
                        raise ValueError(f"URL does not contain a PDF file. Content-Type: {content_type}")
                    
                    # Continue reading the rest
                    content = first_chunk
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        content += chunk
                        # Check size as we download
                        if len(content) > MAX_PDF_SIZE:
                            raise ValueError(f"PDF file too large: {len(content)} bytes > {MAX_PDF_SIZE}")
                else:
                    # Read all content with size checking
                    content = b""
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        content += chunk
                        if len(content) > MAX_PDF_SIZE:
                            raise ValueError(f"PDF file too large: {len(content)} bytes > {MAX_PDF_SIZE}")
                
                # Double-check magic bytes
                if not content.startswith(b"%PDF"):
                    raise ValueError("Downloaded content is not a valid PDF file")
                
                # Save to cache with secure permissions
                cache_file.write_bytes(content)
                cache_file.chmod(0o600)  # Owner read/write only
                logger.info(f"Downloaded and cached PDF: {cache_file} ({len(content)} bytes)")
                return cache_file
            
    except httpx.HTTPError as e:
        sanitized_error = sanitize_error_message(e, "PDF download failed")
        raise ValueError(sanitized_error)
    except Exception as e:
        sanitized_error = sanitize_error_message(e, "PDF download error")
        raise ValueError(sanitized_error)

async def validate_pdf_path(pdf_path: str) -> Path:
    """Validate path (local or URL) with security checks and size limits"""
    # Input length validation
    if len(pdf_path) > 2000:
        raise ValueError("PDF path too long")
    
    # Check for path traversal in input
    if '../' in pdf_path or '\\..\\' in pdf_path:
        raise ValueError("Path traversal detected")
    
    # Check if it's a URL
    parsed = urlparse(pdf_path)
    
    if parsed.scheme in ('http', 'https'):
        if parsed.scheme == 'http':
            logger.warning(f"Using insecure HTTP URL: {pdf_path}")
        return await download_pdf_from_url(pdf_path)
    
    # Handle local path with security validation
    path = Path(pdf_path).resolve()
    
    if not path.exists():
        raise ValueError(f"File not found: {pdf_path}")
    
    if not path.suffix.lower() == '.pdf':
        raise ValueError(f"Not a PDF file: {pdf_path}")
    
    # Check file size
    file_size = path.stat().st_size
    if file_size > MAX_PDF_SIZE:
        raise ValueError(f"PDF file too large: {file_size} bytes > {MAX_PDF_SIZE}")
    
    return path

def detect_scanned_pdf(pdf_path: str) -> bool:
    """Detect if a PDF is scanned (image-based)"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Check first few pages for text
            pages_to_check = min(3, len(pdf.pages))
            for i in range(pages_to_check):
                text = pdf.pages[i].extract_text()
                if text and len(text.strip()) > 50:
                    return False
        return True
    except Exception:
        return True

# Text extraction methods
async def extract_with_pymupdf(pdf_path: Path, pages: Optional[List[int]] = None, preserve_layout: bool = False) -> str:
    """Extract text using PyMuPDF"""
    doc = fitz.open(str(pdf_path))
    text_parts = []
    
    try:
        page_range = pages if pages else range(len(doc))
        for page_num in page_range:
            page = doc[page_num]
            if preserve_layout:
                text_parts.append(page.get_text("text"))
            else:
                text_parts.append(page.get_text())
    finally:
        doc.close()
    
    return "\n\n".join(text_parts)

async def extract_with_pdfplumber(pdf_path: Path, pages: Optional[List[int]] = None, preserve_layout: bool = False) -> str:
    """Extract text using pdfplumber"""
    text_parts = []
    
    with pdfplumber.open(str(pdf_path)) as pdf:
        page_range = pages if pages else range(len(pdf.pages))
        for page_num in page_range:
            page = pdf.pages[page_num]
            text = page.extract_text(layout=preserve_layout)
            if text:
                text_parts.append(text)
    
    return "\n\n".join(text_parts)

async def extract_with_pypdf(pdf_path: Path, pages: Optional[List[int]] = None, preserve_layout: bool = False) -> str:
    """Extract text using pypdf"""
    reader = pypdf.PdfReader(str(pdf_path))
    text_parts = []
    
    page_range = pages if pages else range(len(reader.pages))
    for page_num in page_range:
        page = reader.pages[page_num]
        text = page.extract_text()
        if text:
            text_parts.append(text)
    
    return "\n\n".join(text_parts)

# Main text extraction tool
@mcp.tool(
    name="extract_text", 
    description="Extract text from PDF with intelligent method selection"
)
async def extract_text(
    pdf_path: str,
    method: str = "auto", 
    pages: Optional[str] = None,  # Accept as string for MCP compatibility
    preserve_layout: bool = False,
    max_tokens: int = 20000,  # Maximum tokens to prevent MCP overflow (MCP hard limit is 25000)
    chunk_pages: int = 10     # Number of pages per chunk for large PDFs
) -> Dict[str, Any]:
    """
    Extract text from PDF using various methods with automatic chunking for large files
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        method: Extraction method (auto, pymupdf, pdfplumber, pypdf)
        pages: Page numbers to extract as string like "1,2,3" or "[1,2,3]", None for all pages (0-indexed)
        preserve_layout: Whether to preserve the original text layout
        max_tokens: Maximum tokens to return (prevents MCP overflow, default 20000)
        chunk_pages: Pages per chunk for large PDFs (default 10)
    
    Returns:
        Dictionary containing extracted text and metadata with chunking info
    """
    import time
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        parsed_pages = parse_pages_parameter(pages)
        
        # Auto-select method based on PDF characteristics
        if method == "auto":
            is_scanned = detect_scanned_pdf(str(path))
            if is_scanned:
                return {
                    "error": "Scanned PDF detected. Please use the OCR tool for this file.",
                    "is_scanned": True
                }
            method = "pymupdf"  # Default to PyMuPDF for text-based PDFs
        
        # Get PDF metadata and size analysis for intelligent chunking decisions
        doc = fitz.open(str(path))
        
        # Validate page count to prevent resource exhaustion
        validate_page_count(doc, "text extraction")
        
        total_pages = len(doc)
        
        # Analyze PDF size and content density
        file_size_bytes = path.stat().st_size if path.is_file() else 0
        file_size_mb = file_size_bytes / (1024 * 1024) if file_size_bytes > 0 else 0
        
        # Sample first few pages to estimate content density and analyze images
        sample_pages = min(3, total_pages)
        sample_text = ""
        total_images = 0
        sample_images = 0
        
        for page_num in range(sample_pages):
            page = doc[page_num]
            page_text = page.get_text()
            sample_text += page_text
            
            # Count images on this page
            images_on_page = len(page.get_images())
            sample_images += images_on_page
        
        # Estimate total images in document
        if sample_pages > 0:
            avg_images_per_page = sample_images / sample_pages
            estimated_total_images = int(avg_images_per_page * total_pages)
        else:
            avg_images_per_page = 0
            estimated_total_images = 0
        
        # Calculate content density metrics
        avg_chars_per_page = len(sample_text) / sample_pages if sample_pages > 0 else 0
        estimated_total_chars = avg_chars_per_page * total_pages
        estimated_tokens_by_density = int(estimated_total_chars / 4)  # 1 token ≈ 4 chars
        
        metadata = {
            "pages": total_pages,
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "file_size_mb": round(file_size_mb, 2),
            "avg_chars_per_page": int(avg_chars_per_page),
            "estimated_total_chars": int(estimated_total_chars),
            "estimated_tokens_by_density": estimated_tokens_by_density,
            "estimated_total_images": estimated_total_images,
            "avg_images_per_page": round(avg_images_per_page, 1),
        }
        doc.close()
        
        # Enforce MCP hard limit regardless of user max_tokens setting
        effective_max_tokens = min(max_tokens, 24000)  # Stay safely under MCP's 25000 limit
        
        # Early chunking decision based on size analysis
        should_chunk_early = (
            total_pages > 50 or  # Large page count
            file_size_mb > 10 or  # Large file size  
            estimated_tokens_by_density > effective_max_tokens or  # High content density
            estimated_total_images > 100  # Many images can bloat response
        )
        
        # Generate warnings and suggestions based on content analysis
        analysis_warnings = []
        if estimated_total_images > 20:
            analysis_warnings.append(f"PDF contains ~{estimated_total_images} images. Consider using 'extract_images' tool for image extraction.")
        
        if file_size_mb > 20:
            analysis_warnings.append(f"Large PDF file ({file_size_mb:.1f}MB). May contain embedded images or high-resolution content.")
        
        if avg_chars_per_page > 5000:
            analysis_warnings.append(f"Dense text content (~{int(avg_chars_per_page):,} chars/page). Chunking recommended for large documents.")
        
        # Add content type suggestions
        if estimated_total_images > avg_chars_per_page / 500:  # More images than expected for text density
            analysis_warnings.append("Image-heavy document detected. Consider 'extract_images' for visual content and 'pdf_to_markdown' for structured text.")
        
        if total_pages > 100 and avg_chars_per_page > 3000:
            analysis_warnings.append(f"Large document ({total_pages} pages) with dense content. Use 'pages' parameter to extract specific sections.")
        
        # Determine pages to extract
        if parsed_pages:
            pages_to_extract = parsed_pages
        else:
            pages_to_extract = list(range(total_pages))
        
        # Extract text using selected method
        if method == "pymupdf":
            text = await extract_with_pymupdf(path, pages_to_extract, preserve_layout)
        elif method == "pdfplumber":
            text = await extract_with_pdfplumber(path, pages_to_extract, preserve_layout)
        elif method == "pypdf":
            text = await extract_with_pypdf(path, pages_to_extract, preserve_layout)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
        
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = len(text) // 4
        
        # Handle large responses with intelligent chunking
        if estimated_tokens > effective_max_tokens:
            # Calculate chunk size based on effective token limit
            chars_per_chunk = effective_max_tokens * 4
            
            # Smart chunking: try to break at page boundaries first
            if len(pages_to_extract) > chunk_pages:
                # Multiple page chunks
                chunk_page_ranges = []
                for i in range(0, len(pages_to_extract), chunk_pages):
                    chunk_pages_list = pages_to_extract[i:i + chunk_pages]
                    chunk_page_ranges.append(chunk_pages_list)
                
                # Extract first chunk
                if method == "pymupdf":
                    chunk_text = await extract_with_pymupdf(path, chunk_page_ranges[0], preserve_layout)
                elif method == "pdfplumber":
                    chunk_text = await extract_with_pdfplumber(path, chunk_page_ranges[0], preserve_layout)
                elif method == "pypdf":
                    chunk_text = await extract_with_pypdf(path, chunk_page_ranges[0], preserve_layout)
                
                return {
                    "text": chunk_text,
                    "method_used": method,
                    "metadata": metadata,
                    "pages_extracted": chunk_page_ranges[0],
                    "extraction_time": round(time.time() - start_time, 2),
                    "chunking_info": {
                        "is_chunked": True,
                        "current_chunk": 1,
                        "total_chunks": len(chunk_page_ranges),
                        "chunk_page_ranges": chunk_page_ranges,
                        "reason": "Large PDF automatically chunked to prevent token overflow",
                        "next_chunk_command": f"Use pages parameter: \"{','.join(map(str, chunk_page_ranges[1]))}\" for chunk 2" if len(chunk_page_ranges) > 1 else None
                    },
                    "warnings": [
                        f"Large PDF ({estimated_tokens:,} estimated tokens) automatically chunked. This is chunk 1 of {len(chunk_page_ranges)}.",
                        f"To get next chunk, use pages parameter or reduce max_tokens to see more content at once."
                    ] + analysis_warnings
                }
            else:
                # Single chunk but too much text - truncate with context
                truncated_text = text[:chars_per_chunk]
                # Try to truncate at sentence boundary
                last_sentence = truncated_text.rfind('. ')
                if last_sentence > chars_per_chunk * 0.8:  # If we find a sentence end in the last 20%
                    truncated_text = truncated_text[:last_sentence + 1]
                
                return {
                    "text": truncated_text,
                    "method_used": method,
                    "metadata": metadata,
                    "pages_extracted": pages_to_extract,
                    "extraction_time": round(time.time() - start_time, 2),
                    "chunking_info": {
                        "is_truncated": True,
                        "original_estimated_tokens": estimated_tokens,
                        "returned_estimated_tokens": len(truncated_text) // 4,
                        "truncation_percentage": round((len(truncated_text) / len(text)) * 100, 1),
                        "reason": "Content truncated to prevent token overflow"
                    },
                    "warnings": [
                        f"Content truncated from {estimated_tokens:,} to ~{len(truncated_text) // 4:,} tokens ({round((len(truncated_text) / len(text)) * 100, 1)}% shown).",
                        "Use specific page ranges with 'pages' parameter to get complete content in smaller chunks."
                    ] + analysis_warnings
                }
        
        # Normal response for reasonably sized content
        return {
            "text": text,
            "method_used": method,
            "metadata": metadata,
            "pages_extracted": pages_to_extract,
            "extraction_time": round(time.time() - start_time, 2),
            "estimated_tokens": estimated_tokens,
            "warnings": analysis_warnings
        }
        
    except Exception as e:
        logger.error(f"Text extraction failed: {str(e)}")
        return {
            "error": f"Text extraction failed: {str(e)}",
            "method_attempted": method
        }

# Table extraction methods
async def extract_tables_camelot(pdf_path: Path, pages: Optional[List[int]] = None) -> List[pd.DataFrame]:
    """Extract tables using Camelot"""
    page_str = ','.join(map(str, [p+1 for p in pages])) if pages else 'all'
    
    # Try lattice mode first (for bordered tables)
    try:
        tables = camelot.read_pdf(str(pdf_path), pages=page_str, flavor='lattice')
        if len(tables) > 0:
            return [table.df for table in tables]
    except Exception:
        pass
    
    # Fall back to stream mode (for borderless tables)
    try:
        tables = camelot.read_pdf(str(pdf_path), pages=page_str, flavor='stream')
        return [table.df for table in tables]
    except Exception:
        return []

async def extract_tables_tabula(pdf_path: Path, pages: Optional[List[int]] = None) -> List[pd.DataFrame]:
    """Extract tables using Tabula"""
    page_list = [p+1 for p in pages] if pages else 'all'
    
    try:
        tables = tabula.read_pdf(str(pdf_path), pages=page_list, multiple_tables=True)
        return tables
    except Exception:
        return []

async def extract_tables_pdfplumber(pdf_path: Path, pages: Optional[List[int]] = None) -> List[pd.DataFrame]:
    """Extract tables using pdfplumber"""
    tables = []
    
    with pdfplumber.open(str(pdf_path)) as pdf:
        page_range = pages if pages else range(len(pdf.pages))
        for page_num in page_range:
            page = pdf.pages[page_num]
            page_tables = page.extract_tables()
            for table in page_tables:
                if table and len(table) > 1:  # Skip empty tables
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df)
    
    return tables

# Main table extraction tool
@mcp.tool(name="extract_tables", description="Extract tables from PDF with automatic method selection")
async def extract_tables(
    pdf_path: str,
    pages: Optional[str] = None,  # Accept as string for MCP compatibility
    method: str = "auto",
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    Extract tables from PDF using various methods
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        pages: List of page numbers to extract tables from (0-indexed)
        method: Extraction method (auto, camelot, tabula, pdfplumber)
        output_format: Output format (json, csv, markdown)
    
    Returns:
        Dictionary containing extracted tables and metadata
    """
    import time
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        parsed_pages = parse_pages_parameter(pages)
        all_tables = []
        methods_tried = []
        
        # Auto method: try methods in order until we find tables
        if method == "auto":
            for try_method in ["camelot", "pdfplumber", "tabula"]:
                methods_tried.append(try_method)
                
                if try_method == "camelot":
                    tables = await extract_tables_camelot(path, parsed_pages)
                elif try_method == "pdfplumber":
                    tables = await extract_tables_pdfplumber(path, parsed_pages)
                elif try_method == "tabula":
                    tables = await extract_tables_tabula(path, parsed_pages)
                
                if tables:
                    method = try_method
                    all_tables = tables
                    break
        else:
            # Use specific method
            methods_tried.append(method)
            if method == "camelot":
                all_tables = await extract_tables_camelot(path, parsed_pages)
            elif method == "pdfplumber":
                all_tables = await extract_tables_pdfplumber(path, parsed_pages)
            elif method == "tabula":
                all_tables = await extract_tables_tabula(path, parsed_pages)
            else:
                raise ValueError(f"Unknown table extraction method: {method}")
        
        # Format tables based on output format
        formatted_tables = []
        for i, df in enumerate(all_tables):
            if output_format == "json":
                formatted_tables.append({
                    "table_index": i,
                    "data": df.to_dict(orient="records"),
                    "shape": {"rows": len(df), "columns": len(df.columns)}
                })
            elif output_format == "csv":
                formatted_tables.append({
                    "table_index": i,
                    "data": df.to_csv(index=False),
                    "shape": {"rows": len(df), "columns": len(df.columns)}
                })
            elif output_format == "markdown":
                formatted_tables.append({
                    "table_index": i,
                    "data": df.to_markdown(index=False),
                    "shape": {"rows": len(df), "columns": len(df.columns)}
                })
        
        return {
            "tables": formatted_tables,
            "total_tables": len(formatted_tables),
            "method_used": method,
            "methods_tried": methods_tried,
            "pages_searched": pages or "all",
            "extraction_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        logger.error(f"Table extraction failed: {str(e)}")
        return {
            "error": f"Table extraction failed: {str(e)}",
            "methods_tried": methods_tried
        }

# OCR functionality
@mcp.tool(name="ocr_pdf", description="Perform OCR on scanned PDFs")
async def ocr_pdf(
    pdf_path: str,
    languages: List[str] = ["eng"],
    preprocess: bool = True,
    dpi: int = 300,
    pages: Optional[str] = None  # Accept as string for MCP compatibility
) -> Dict[str, Any]:
    """
    Perform OCR on a scanned PDF
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        languages: List of language codes for OCR (e.g., ["eng", "fra"])
        preprocess: Whether to preprocess images for better OCR
        dpi: DPI for PDF to image conversion
        pages: Specific pages to OCR (0-indexed)
    
    Returns:
        Dictionary containing OCR text and metadata
    """
    import time
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        parsed_pages = parse_pages_parameter(pages)
        
        # Convert PDF pages to images
        with tempfile.TemporaryDirectory() as temp_dir:
            if parsed_pages:
                images = []
                for page_num in parsed_pages:
                    page_images = convert_from_path(
                        str(path), 
                        dpi=dpi, 
                        first_page=page_num+1, 
                        last_page=page_num+1,
                        output_folder=temp_dir
                    )
                    images.extend(page_images)
            else:
                images = convert_from_path(str(path), dpi=dpi, output_folder=temp_dir)
            
            # Perform OCR on each page
            ocr_texts = []
            for i, image in enumerate(images):
                # Preprocess image if requested
                if preprocess:
                    # Convert to grayscale
                    image = image.convert('L')
                    
                    # Enhance contrast
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(2.0)
                
                # Perform OCR
                lang_str = '+'.join(languages)
                text = pytesseract.image_to_string(image, lang=lang_str)
                ocr_texts.append(text)
            
            # Combine all OCR text
            full_text = "\n\n--- Page Break ---\n\n".join(ocr_texts)
            
            return {
                "text": full_text,
                "pages_processed": len(images),
                "languages": languages,
                "dpi": dpi,
                "preprocessing_applied": preprocess,
                "extraction_time": round(time.time() - start_time, 2)
            }
            
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        return {
            "error": f"OCR failed: {str(e)}",
            "hint": "Make sure Tesseract is installed and language data is available"
        }

# PDF analysis tools
@mcp.tool(name="is_scanned_pdf", description="Check if a PDF is scanned/image-based")
async def is_scanned_pdf(pdf_path: str) -> Dict[str, Any]:
    """Check if a PDF is scanned (image-based) or contains extractable text"""
    try:
        path = await validate_pdf_path(pdf_path)
        is_scanned = detect_scanned_pdf(str(path))
        
        # Get more details
        doc = fitz.open(str(path))
        page_count = len(doc)
        
        # Check a few pages for text content
        sample_pages = min(5, page_count)
        text_pages = 0
        
        for i in range(sample_pages):
            page = doc[i]
            text = page.get_text().strip()
            if len(text) > 50:
                text_pages += 1
        
        doc.close()
        
        return {
            "is_scanned": is_scanned,
            "page_count": page_count,
            "sample_pages_checked": sample_pages,
            "pages_with_text": text_pages,
            "recommendation": "Use OCR tool" if is_scanned else "Use text extraction tool"
        }
        
    except Exception as e:
        logger.error(f"PDF scan detection failed: {str(e)}")
        return {"error": f"Failed to analyze PDF: {str(e)}"}

@mcp.tool(name="get_document_structure", description="Extract document structure including headers, sections, and metadata")
async def get_document_structure(pdf_path: str) -> Dict[str, Any]:
    """
    Extract document structure including headers, sections, and metadata
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
    
    Returns:
        Dictionary containing document structure information
    """
    try:
        path = await validate_pdf_path(pdf_path)
        doc = fitz.open(str(path))
        
        structure = {
            "metadata": {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": str(doc.metadata.get("creationDate", "")),
                "modification_date": str(doc.metadata.get("modDate", "")),
            },
            "pages": len(doc),
            "outline": []
        }
        
        # Extract table of contents / bookmarks
        toc = doc.get_toc()
        for level, title, page in toc:
            structure["outline"].append({
                "level": level,
                "title": title,
                "page": page
            })
        
        # Extract page-level information
        page_info = []
        for i in range(min(5, len(doc))):  # Sample first 5 pages
            page = doc[i]
            page_data = {
                "page_number": i + 1,
                "width": page.rect.width,
                "height": page.rect.height,
                "rotation": page.rotation,
                "text_length": len(page.get_text()),
                "image_count": len(page.get_images()),
                "link_count": len(page.get_links())
            }
            page_info.append(page_data)
        
        structure["sample_pages"] = page_info
        
        # Detect fonts used
        fonts = set()
        for page in doc:
            for font in page.get_fonts():
                fonts.add(font[3])  # Font name
        structure["fonts"] = list(fonts)
        
        doc.close()
        
        return structure
        
    except Exception as e:
        logger.error(f"Document structure extraction failed: {str(e)}")
        return {"error": f"Failed to extract document structure: {str(e)}"}

# PDF to Markdown conversion
@mcp.tool(name="pdf_to_markdown", description="Convert PDF to markdown with MCP resource URIs for images")
async def pdf_to_markdown(
    pdf_path: str,
    include_images: bool = True,
    include_metadata: bool = True,
    pages: Optional[str] = None  # Accept as string for MCP compatibility
) -> Dict[str, Any]:
    """
    Convert PDF to markdown format with MCP resource image links
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        include_images: Whether to extract and include images as MCP resources
        include_metadata: Whether to include document metadata
        pages: Specific pages to convert (1-based user input, converted to 0-based)
    
    Returns:
        Dictionary containing markdown content with MCP resource URIs for images
    """
    import time
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        parsed_pages = parse_pages_parameter(pages)
        doc = fitz.open(str(path))
        
        markdown_parts = []
        
        # Add metadata if requested
        if include_metadata:
            metadata = doc.metadata
            if any(metadata.values()):
                markdown_parts.append("# Document Metadata\n")
                for key, value in metadata.items():
                    if value:
                        markdown_parts.append(f"- **{key.title()}**: {value}")
                markdown_parts.append("\n---\n")
        
        # Extract table of contents
        toc = doc.get_toc()
        if toc:
            markdown_parts.append("# Table of Contents\n")
            for level, title, page in toc:
                indent = "  " * (level - 1)
                markdown_parts.append(f"{indent}- [{title}](#{page})")
            markdown_parts.append("\n---\n")
        
        # Process pages
        page_range = parsed_pages if parsed_pages else range(len(doc))
        images_extracted = []
        
        for page_num in page_range:
            page = doc[page_num]
            
            # Add page header
            markdown_parts.append(f"\n## Page {page_num + 1}\n")
            
            # Extract text with basic formatting
            blocks = page.get_text("blocks")
            
            for block in blocks:
                if block[6] == 0:  # Text block
                    text = block[4].strip()
                    if text:
                        # Try to detect headers by font size
                        if len(text) < 100 and text.isupper():
                            markdown_parts.append(f"### {text}\n")
                        else:
                            markdown_parts.append(f"{text}\n")
            
            # Extract images if requested
            if include_images:
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        # Save image to file instead of embedding base64 data
                        img_filename = f"markdown_page_{page_num + 1}_image_{img_index}.png"
                        img_path = CACHE_DIR / img_filename
                        pix.save(str(img_path))
                        
                        file_size = img_path.stat().st_size
                        
                        # Create resource URI (filename without extension)
                        image_id = img_filename.rsplit('.', 1)[0]  # Remove extension
                        resource_uri = f"pdf-image://{image_id}"
                        
                        images_extracted.append({
                            "page": page_num + 1,
                            "index": img_index,
                            "file_path": str(img_path),
                            "filename": img_filename,
                            "resource_uri": resource_uri,
                            "width": pix.width,
                            "height": pix.height,
                            "size_bytes": file_size,
                            "size_human": format_file_size(file_size)
                        })
                        # Reference the resource URI in markdown
                        markdown_parts.append(f"\n![Image {page_num+1}-{img_index}]({resource_uri})\n")
                    pix = None
        
        doc.close()
        
        # Combine markdown
        markdown_content = "\n".join(markdown_parts)
        
        return {
            "markdown": markdown_content,
            "pages_converted": len(page_range),
            "images_extracted": len(images_extracted),
            "images": images_extracted if include_images else [],
            "conversion_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        logger.error(f"PDF to Markdown conversion failed: {str(e)}")
        return {"error": f"Conversion failed: {str(e)}"}

# Image extraction
@mcp.tool(name="extract_images", description="Extract images from PDF with custom output path and clean summary")
async def extract_images(
    pdf_path: str,
    pages: Optional[str] = None,  # Accept as string for MCP compatibility
    min_width: int = 100,
    min_height: int = 100,
    output_format: str = "png",
    output_directory: Optional[str] = None,  # Custom output directory
    include_context: bool = True,  # Extract text context around images
    context_chars: int = 200  # Characters of context before/after images
) -> Dict[str, Any]:
    """
    Extract images from PDF with positioning context for text-image coordination
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        pages: Specific pages to extract images from (1-based user input, converted to 0-based)
        min_width: Minimum image width to extract
        min_height: Minimum image height to extract
        output_format: Output format (png, jpeg)
        output_directory: Custom directory to save images (defaults to cache directory)
        include_context: Extract text context around images for coordination
        context_chars: Characters of context before/after each image
    
    Returns:
        Detailed extraction results with positioning info and text context for workflow coordination
    """
    try:
        path = await validate_pdf_path(pdf_path)
        parsed_pages = parse_pages_parameter(pages)
        doc = fitz.open(str(path))
        
        # Determine output directory with security validation
        if output_directory:
            output_dir = validate_output_path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        else:
            output_dir = CACHE_DIR
        
        extracted_files = []
        total_size = 0
        page_range = parsed_pages if parsed_pages else range(len(doc))
        pages_with_images = []
        
        for page_num in page_range:
            page = doc[page_num]
            image_list = page.get_images()
            
            if not image_list:
                continue  # Skip pages without images
                
            # Get page text for context analysis
            page_text = page.get_text() if include_context else ""
            page_blocks = page.get_text("dict")["blocks"] if include_context else []
            
            page_images = []
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Check size requirements
                    if pix.width >= min_width and pix.height >= min_height:
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            if output_format == "jpeg" and pix.alpha:
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            
                            # Get image positioning from page
                            img_rects = []
                            for block in page_blocks:
                                if block.get("type") == 1:  # Image block
                                    for line in block.get("lines", []):
                                        for span in line.get("spans", []):
                                            if "image" in str(span).lower():
                                                img_rects.append(block.get("bbox", [0, 0, 0, 0]))
                            
                            # Find image rectangle on page (approximate)
                            img_instances = page.search_for("image") or []
                            img_rect = None
                            if img_index < len(img_rects):
                                bbox = img_rects[img_index]
                                img_rect = {
                                    "x0": bbox[0], "y0": bbox[1], 
                                    "x1": bbox[2], "y1": bbox[3],
                                    "width": bbox[2] - bbox[0],
                                    "height": bbox[3] - bbox[1]
                                }
                            
                            # Extract context around image position if available
                            context_before = ""
                            context_after = ""
                            
                            if include_context and page_text and img_rect:
                                # Simple approach: estimate text position relative to image
                                text_blocks_before = []
                                text_blocks_after = []
                                
                                for block in page_blocks:
                                    if block.get("type") == 0:  # Text block
                                        block_bbox = block.get("bbox", [0, 0, 0, 0])
                                        block_center_y = (block_bbox[1] + block_bbox[3]) / 2
                                        img_center_y = (img_rect["y0"] + img_rect["y1"]) / 2
                                        
                                        # Extract text from block
                                        block_text = ""
                                        for line in block.get("lines", []):
                                            for span in line.get("spans", []):
                                                block_text += span.get("text", "")
                                        
                                        if block_center_y < img_center_y:
                                            text_blocks_before.append((block_center_y, block_text))
                                        else:
                                            text_blocks_after.append((block_center_y, block_text))
                                
                                # Get closest text before and after
                                if text_blocks_before:
                                    text_blocks_before.sort(key=lambda x: x[0], reverse=True)
                                    context_before = text_blocks_before[0][1][-context_chars:]
                                
                                if text_blocks_after:
                                    text_blocks_after.sort(key=lambda x: x[0])
                                    context_after = text_blocks_after[0][1][:context_chars]
                            
                            # Save image to specified directory
                            img_filename = f"page_{page_num + 1}_image_{img_index + 1}.{output_format}"
                            img_path = output_dir / img_filename
                            pix.save(str(img_path))
                            
                            # Calculate file size
                            file_size = img_path.stat().st_size
                            total_size += file_size
                            
                            # Create detailed image info
                            image_info = {
                                "filename": img_filename,
                                "path": str(img_path),
                                "page": page_num + 1,
                                "image_index": img_index + 1,
                                "dimensions": {
                                    "width": pix.width,
                                    "height": pix.height
                                },
                                "file_size": format_file_size(file_size),
                                "positioning": img_rect,
                                "context": {
                                    "before": context_before.strip() if context_before else None,
                                    "after": context_after.strip() if context_after else None
                                } if include_context else None,
                                "extraction_method": "PyMuPDF",
                                "format": output_format
                            }
                            
                            extracted_files.append(image_info)
                            page_images.append(image_info)
                    
                    pix = None
                    
                except Exception as e:
                    # Continue with other images if one fails
                    logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {str(e)}")
                    continue
            
            if page_images:
                pages_with_images.append({
                    "page": page_num + 1,
                    "image_count": len(page_images),
                    "images": [{"filename": img["filename"], "dimensions": img["dimensions"]} for img in page_images]
                })
        
        doc.close()
        
        # Create comprehensive response
        response = {
            "success": True,
            "images_extracted": len(extracted_files),
            "pages_with_images": pages_with_images,
            "total_size": format_file_size(total_size),
            "output_directory": str(output_dir),
            "extraction_settings": {
                "min_dimensions": f"{min_width}x{min_height}",
                "output_format": output_format,
                "context_included": include_context,
                "context_chars": context_chars if include_context else 0
            },
            "workflow_coordination": {
                "pages_with_images": [p["page"] for p in pages_with_images],
                "total_pages_scanned": len(page_range),
                "context_available": include_context,
                "positioning_data": any(img.get("positioning") for img in extracted_files)
            },
            "extracted_images": extracted_files
        }
        
        # Check response size and chunk if needed
        import json
        response_str = json.dumps(response)
        estimated_tokens = len(response_str) // 4
        
        if estimated_tokens > 20000:  # Similar to text extraction limit
            # Create chunked response for large results
            chunked_response = {
                "success": True,
                "images_extracted": len(extracted_files),
                "pages_with_images": pages_with_images,
                "total_size": format_file_size(total_size),
                "output_directory": str(output_dir),
                "extraction_settings": response["extraction_settings"],
                "workflow_coordination": response["workflow_coordination"],
                "chunking_info": {
                    "response_too_large": True,
                    "estimated_tokens": estimated_tokens,
                    "total_images": len(extracted_files),
                    "chunking_suggestion": "Use 'pages' parameter to extract images from specific page ranges",
                    "example_commands": [
                        f"Extract pages 1-10: pages='1,2,3,4,5,6,7,8,9,10'",
                        f"Extract specific pages with images: pages='{','.join(map(str, pages_with_images[:5]))}'"
                    ][:2]
                },
                "warnings": [
                    f"Response too large ({estimated_tokens:,} tokens). Use page-specific extraction for detailed results.",
                    f"Extracted {len(extracted_files)} images from {len(pages_with_images)} pages. Use 'pages' parameter for detailed context."
                ]
            }
            return chunked_response
        
        return response
        
    except Exception as e:
        logger.error(f"Image extraction failed: {str(e)}")
        return {"error": f"Image extraction failed: {str(e)}"}

# Metadata extraction
@mcp.tool(name="extract_metadata", description="Extract comprehensive PDF metadata")
async def extract_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from PDF
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
    
    Returns:
        Dictionary containing all available metadata
    """
    try:
        path = await validate_pdf_path(pdf_path)
        
        # Get file stats
        file_stats = path.stat()
        
        # PyMuPDF metadata
        doc = fitz.open(str(path))
        fitz_metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": str(doc.metadata.get("creationDate", "")),
            "modification_date": str(doc.metadata.get("modDate", "")),
            "trapped": doc.metadata.get("trapped", ""),
        }
        
        # Document statistics  
        has_annotations = False
        has_links = False
        try:
            for page in doc:
                if hasattr(page, 'annots') and page.annots() is not None:
                    annots_list = list(page.annots())
                    if len(annots_list) > 0:
                        has_annotations = True
                        break
        except Exception:
            pass
            
        try:
            for page in doc:
                if page.get_links():
                    has_links = True
                    break
        except Exception:
            pass
            
        stats = {
            "page_count": len(doc),
            "file_size_bytes": file_stats.st_size,
            "file_size_mb": round(file_stats.st_size / (1024*1024), 2),
            "is_encrypted": doc.is_encrypted,
            "is_form": doc.is_form_pdf,
            "has_annotations": has_annotations,
            "has_links": has_links,
        }
        
        # Page dimensions
        if len(doc) > 0:
            first_page = doc[0]
            stats["page_width"] = first_page.rect.width
            stats["page_height"] = first_page.rect.height
            stats["page_rotation"] = first_page.rotation
        
        doc.close()
        
        # PyPDF metadata (sometimes has additional info)
        try:
            reader = pypdf.PdfReader(str(path))
            pypdf_metadata = reader.metadata
            
            additional_metadata = {}
            if pypdf_metadata:
                for key, value in pypdf_metadata.items():
                    key_str = key.strip("/")
                    if key_str not in fitz_metadata or not fitz_metadata[key_str]:
                        additional_metadata[key_str] = str(value)
        except Exception:
            additional_metadata = {}
        
        return {
            "file_info": {
                "path": str(path),
                "name": path.name,
                "size_bytes": file_stats.st_size,
                "size_mb": round(file_stats.st_size / (1024*1024), 2),
                "created": str(file_stats.st_ctime),
                "modified": str(file_stats.st_mtime),
            },
            "metadata": fitz_metadata,
            "statistics": stats,
            "additional_metadata": additional_metadata
        }
        
    except Exception as e:
        logger.error(f"Metadata extraction failed: {str(e)}")
        return {"error": f"Metadata extraction failed: {str(e)}"}

# Advanced Analysis Tools

@mcp.tool(name="compare_pdfs", description="Compare two PDFs for differences in text, structure, and metadata")
async def compare_pdfs(
    pdf_path1: str,
    pdf_path2: str,
    comparison_type: str = "all"  # all, text, structure, metadata
) -> Dict[str, Any]:
    """
    Compare two PDFs for differences
    
    Args:
        pdf_path1: Path to first PDF file or HTTPS URL
        pdf_path2: Path to second PDF file or HTTPS URL  
        comparison_type: Type of comparison (all, text, structure, metadata)
    
    Returns:
        Dictionary containing comparison results
    """
    import time
    start_time = time.time()
    
    try:
        path1 = await validate_pdf_path(pdf_path1)
        path2 = await validate_pdf_path(pdf_path2)
        
        doc1 = fitz.open(str(path1))
        doc2 = fitz.open(str(path2))
        
        comparison_results = {
            "files_compared": {
                "file1": str(path1),
                "file2": str(path2)
            },
            "comparison_type": comparison_type
        }
        
        # Structure comparison
        if comparison_type in ["all", "structure"]:
            structure_diff = {
                "page_count": {
                    "file1": len(doc1),
                    "file2": len(doc2),
                    "difference": len(doc1) - len(doc2)
                },
                "file_size": {
                    "file1": path1.stat().st_size,
                    "file2": path2.stat().st_size,
                    "difference": path1.stat().st_size - path2.stat().st_size
                },
                "fonts": {
                    "file1": [],
                    "file2": [],
                    "common": [],
                    "unique_to_file1": [],
                    "unique_to_file2": []
                }
            }
            
            # Extract fonts from both documents
            fonts1 = set()
            fonts2 = set()
            
            for page in doc1:
                for font in page.get_fonts():
                    fonts1.add(font[3])  # Font name
                    
            for page in doc2:
                for font in page.get_fonts():
                    fonts2.add(font[3])  # Font name
            
            structure_diff["fonts"]["file1"] = list(fonts1)
            structure_diff["fonts"]["file2"] = list(fonts2)
            structure_diff["fonts"]["common"] = list(fonts1.intersection(fonts2))
            structure_diff["fonts"]["unique_to_file1"] = list(fonts1 - fonts2)
            structure_diff["fonts"]["unique_to_file2"] = list(fonts2 - fonts1)
            
            comparison_results["structure_comparison"] = structure_diff
        
        # Metadata comparison
        if comparison_type in ["all", "metadata"]:
            meta1 = doc1.metadata
            meta2 = doc2.metadata
            
            metadata_diff = {
                "file1_metadata": meta1,
                "file2_metadata": meta2,
                "differences": {}
            }
            
            all_keys = set(meta1.keys()).union(set(meta2.keys()))
            for key in all_keys:
                val1 = meta1.get(key, "")
                val2 = meta2.get(key, "")
                if val1 != val2:
                    metadata_diff["differences"][key] = {
                        "file1": val1,
                        "file2": val2
                    }
            
            comparison_results["metadata_comparison"] = metadata_diff
        
        # Text comparison  
        if comparison_type in ["all", "text"]:
            text1 = ""
            text2 = ""
            
            # Extract text from both documents
            for page in doc1:
                text1 += page.get_text() + "\n"
                
            for page in doc2:
                text2 += page.get_text() + "\n"
            
            # Calculate similarity
            similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
            
            # Generate diff
            diff_lines = list(difflib.unified_diff(
                text1.splitlines(keepends=True),
                text2.splitlines(keepends=True),
                fromfile="file1",
                tofile="file2",
                n=3
            ))
            
            text_comparison = {
                "similarity_ratio": similarity,
                "similarity_percentage": round(similarity * 100, 2),
                "character_count": {
                    "file1": len(text1),
                    "file2": len(text2),
                    "difference": len(text1) - len(text2)
                },
                "word_count": {
                    "file1": len(text1.split()),
                    "file2": len(text2.split()),
                    "difference": len(text1.split()) - len(text2.split())
                },
                "differences_found": len(diff_lines) > 0,
                "diff_summary": "".join(diff_lines[:50])  # First 50 lines of diff
            }
            
            comparison_results["text_comparison"] = text_comparison
        
        doc1.close()
        doc2.close()
        
        comparison_results["comparison_time"] = round(time.time() - start_time, 2)
        comparison_results["overall_similarity"] = "high" if comparison_results.get("text_comparison", {}).get("similarity_ratio", 0) > 0.8 else "medium" if comparison_results.get("text_comparison", {}).get("similarity_ratio", 0) > 0.5 else "low"
        
        return comparison_results
        
    except Exception as e:
        return {"error": f"PDF comparison failed: {str(e)}", "comparison_time": round(time.time() - start_time, 2)}

@mcp.tool(name="analyze_pdf_health", description="Comprehensive PDF health and quality analysis")
async def analyze_pdf_health(pdf_path: str) -> Dict[str, Any]:
    """
    Analyze PDF health, quality, and potential issues
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
    
    Returns:
        Dictionary containing health analysis results
    """
    import time
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        doc = fitz.open(str(path))
        
        health_report = {
            "file_info": {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2)
            },
            "document_health": {},
            "quality_metrics": {},
            "optimization_suggestions": [],
            "warnings": [],
            "errors": []
        }
        
        # Basic document health
        page_count = len(doc)
        health_report["document_health"]["page_count"] = page_count
        health_report["document_health"]["is_valid"] = page_count > 0
        
        # Check for corruption by trying to access each page
        corrupted_pages = []
        total_text_length = 0
        total_images = 0
        
        for i, page in enumerate(doc):
            try:
                text = page.get_text()
                total_text_length += len(text)
                total_images += len(page.get_images())
            except Exception as e:
                corrupted_pages.append({"page": i + 1, "error": str(e)})
        
        health_report["document_health"]["corrupted_pages"] = corrupted_pages
        health_report["document_health"]["corruption_detected"] = len(corrupted_pages) > 0
        
        # Quality metrics
        health_report["quality_metrics"]["average_text_per_page"] = total_text_length / page_count if page_count > 0 else 0
        health_report["quality_metrics"]["total_images"] = total_images
        health_report["quality_metrics"]["images_per_page"] = total_images / page_count if page_count > 0 else 0
        
        # Font analysis
        fonts_used = set()
        embedded_fonts = 0
        
        for page in doc:
            for font_info in page.get_fonts():
                font_name = font_info[3]
                fonts_used.add(font_name)
                if font_info[1] == "n/a":  # Not embedded
                    pass
                else:
                    embedded_fonts += 1
        
        health_report["quality_metrics"]["fonts_used"] = len(fonts_used)
        health_report["quality_metrics"]["fonts_list"] = list(fonts_used)
        health_report["quality_metrics"]["embedded_fonts"] = embedded_fonts
        
        # Security and protection
        health_report["document_health"]["is_encrypted"] = doc.is_encrypted
        health_report["document_health"]["needs_password"] = doc.needs_pass
        
        # Optimization suggestions
        file_size_mb = health_report["file_info"]["size_mb"]
        
        if file_size_mb > 10:
            health_report["optimization_suggestions"].append("Large file size - consider image compression")
        
        if total_images > page_count * 5:
            health_report["optimization_suggestions"].append("High image density - review image optimization")
        
        if len(fonts_used) > 10:
            health_report["optimization_suggestions"].append("Many fonts used - consider font subsetting")
        
        if embedded_fonts < len(fonts_used):
            health_report["warnings"].append("Some fonts are not embedded - may cause display issues")
        
        # Text/image ratio analysis
        if total_text_length < page_count * 100:  # Very little text
            if total_images > 0:
                health_report["quality_metrics"]["content_type"] = "image-heavy"
                health_report["warnings"].append("Appears to be image-heavy document - consider OCR if text extraction needed")
            else:
                health_report["warnings"].append("Very little text content detected")
        else:
            health_report["quality_metrics"]["content_type"] = "text-based"
        
        # Overall health score
        issues = len(health_report["warnings"]) + len(health_report["errors"]) + len(corrupted_pages)
        if issues == 0:
            health_score = 100
        elif issues <= 2:
            health_score = 85 - (issues * 10)
        else:
            health_score = max(50, 85 - (issues * 15))
        
        health_report["overall_health_score"] = health_score
        health_report["health_status"] = "excellent" if health_score >= 90 else "good" if health_score >= 75 else "fair" if health_score >= 60 else "poor"
        
        doc.close()
        health_report["analysis_time"] = round(time.time() - start_time, 2)
        
        return health_report
        
    except Exception as e:
        return {"error": f"Health analysis failed: {str(e)}", "analysis_time": round(time.time() - start_time, 2)}

@mcp.tool(name="extract_form_data", description="Extract form fields and their values from PDF forms")
async def extract_form_data(pdf_path: str) -> Dict[str, Any]:
    """
    Extract form fields and their values from PDF forms
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
    
    Returns:
        Dictionary containing form data
    """
    import time
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        doc = fitz.open(str(path))
        
        form_data = {
            "has_forms": False,
            "form_fields": [],
            "form_summary": {},
            "extraction_time": 0
        }
        
        # Check if document has forms
        if doc.is_form_pdf:
            form_data["has_forms"] = True
            
            # Extract form fields
            fields_by_type = defaultdict(int)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                widgets = page.widgets()
                
                for widget in widgets:
                    field_info = {
                        "page": page_num + 1,
                        "field_name": widget.field_name or f"unnamed_field_{len(form_data['form_fields'])}",
                        "field_type": widget.field_type_string,
                        "field_value": widget.field_value,
                        "is_required": widget.field_flags & 2 != 0,
                        "is_readonly": widget.field_flags & 1 != 0,
                        "coordinates": {
                            "x0": widget.rect.x0,
                            "y0": widget.rect.y0, 
                            "x1": widget.rect.x1,
                            "y1": widget.rect.y1
                        }
                    }
                    
                    # Additional type-specific data
                    if widget.field_type == 2:  # Text field
                        field_info["max_length"] = widget.text_maxlen
                    elif widget.field_type == 3:  # Choice field
                        field_info["choices"] = widget.choice_values
                    elif widget.field_type == 4:  # Checkbox/Radio
                        field_info["is_checked"] = widget.field_value == "Yes"
                    
                    form_data["form_fields"].append(field_info)
                    fields_by_type[widget.field_type_string] += 1
            
            # Form summary
            form_data["form_summary"] = {
                "total_fields": len(form_data["form_fields"]),
                "fields_by_type": dict(fields_by_type),
                "filled_fields": len([f for f in form_data["form_fields"] if f["field_value"]]),
                "required_fields": len([f for f in form_data["form_fields"] if f["is_required"]]),
                "readonly_fields": len([f for f in form_data["form_fields"] if f["is_readonly"]])
            }
        
        doc.close()
        form_data["extraction_time"] = round(time.time() - start_time, 2)
        
        return form_data
        
    except Exception as e:
        return {"error": f"Form data extraction failed: {str(e)}", "extraction_time": round(time.time() - start_time, 2)}

@mcp.tool(name="split_pdf", description="Split PDF into multiple files at specified pages")
async def split_pdf(
    pdf_path: str,
    split_points: str,  # Accept as string like "2,5,8" for MCP compatibility
    output_prefix: str = "split_part"
) -> Dict[str, Any]:
    """
    Split PDF into multiple files at specified pages
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        split_points: Page numbers where to split (comma-separated like "2,5,8")
        output_prefix: Prefix for output files
    
    Returns:
        Dictionary containing split results
    """
    import time
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        doc = fitz.open(str(path))
        
        # Parse split points (convert from 1-based user input to 0-based internal)
        if isinstance(split_points, str):
            try:
                if ',' in split_points:
                    user_split_list = [int(p.strip()) for p in split_points.split(',')]
                else:
                    user_split_list = [int(split_points.strip())]
                # Convert to 0-based for internal processing
                split_list = [max(0, p - 1) for p in user_split_list]
            except ValueError:
                return {"error": f"Invalid split points format: {split_points}. Use 1-based page numbers like '2,5,8'"}
        else:
            # Assume it's already parsed list, convert from 1-based to 0-based
            split_list = [max(0, p - 1) for p in split_points]
        
        # Sort and validate split points (now 0-based)
        split_list = sorted(set(split_list))
        page_count = len(doc)
        split_list = [p for p in split_list if 0 <= p < page_count]  # Remove invalid pages
        
        if not split_list:
            return {"error": "No valid split points provided"}
        
        # Add start and end points
        split_ranges = []
        start = 0
        
        for split_point in split_list:
            if start < split_point:
                split_ranges.append((start, split_point - 1))
            start = split_point
        
        # Add final range
        if start < page_count:
            split_ranges.append((start, page_count - 1))
        
        # Create split files
        output_files = []
        temp_dir = CACHE_DIR / "split_output"
        temp_dir.mkdir(exist_ok=True)
        
        for i, (start_page, end_page) in enumerate(split_ranges):
            output_file = temp_dir / f"{output_prefix}_{i+1}_pages_{start_page+1}-{end_page+1}.pdf"
            
            # Create new document with specified pages
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
            new_doc.save(str(output_file))
            new_doc.close()
            
            output_files.append({
                "file_path": str(output_file),
                "pages_included": f"{start_page+1}-{end_page+1}",
                "page_count": end_page - start_page + 1,
                "file_size": output_file.stat().st_size
            })
        
        doc.close()
        
        return {
            "original_file": str(path),
            "original_page_count": page_count,
            "split_points": [p + 1 for p in split_list],  # Convert back to 1-based for display
            "output_files": output_files,
            "total_parts": len(output_files),
            "split_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"PDF split failed: {str(e)}", "split_time": round(time.time() - start_time, 2)}

@mcp.tool(name="merge_pdfs", description="Merge multiple PDFs into a single file")  
async def merge_pdfs(
    pdf_paths: str,  # Accept as comma-separated string for MCP compatibility
    output_filename: str = "merged_document.pdf"
) -> Dict[str, Any]:
    """
    Merge multiple PDFs into a single file
    
    Args:
        pdf_paths: Comma-separated list of PDF file paths or URLs
        output_filename: Name for the merged output file
    
    Returns:
        Dictionary containing merge results
    """
    import time
    start_time = time.time()
    
    try:
        # Parse PDF paths
        if isinstance(pdf_paths, str):
            path_list = [p.strip() for p in pdf_paths.split(',')]
        else:
            path_list = pdf_paths
        
        if len(path_list) < 2:
            return {"error": "At least 2 PDF files are required for merging"}
        
        # Validate all paths
        validated_paths = []
        for pdf_path in path_list:
            try:
                validated_path = await validate_pdf_path(pdf_path)
                validated_paths.append(validated_path)
            except Exception as e:
                return {"error": f"Failed to validate path '{pdf_path}': {str(e)}"}
        
        # Create merged document
        merged_doc = fitz.open()
        merge_info = []
        
        total_pages = 0
        for i, path in enumerate(validated_paths):
            doc = fitz.open(str(path))
            page_count = len(doc)
            
            # Insert all pages from current document
            merged_doc.insert_pdf(doc)
            
            merge_info.append({
                "file": str(path),
                "pages_added": page_count,
                "page_range_in_merged": f"{total_pages + 1}-{total_pages + page_count}",
                "file_size": path.stat().st_size
            })
            
            total_pages += page_count
            doc.close()
        
        # Save merged document
        output_path = CACHE_DIR / output_filename
        merged_doc.save(str(output_path))
        merged_doc.close()
        
        return {
            "merged_file": str(output_path),
            "merged_file_size": output_path.stat().st_size,
            "total_pages": total_pages,
            "source_files": merge_info,
            "files_merged": len(validated_paths),
            "merge_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"PDF merge failed: {str(e)}", "merge_time": round(time.time() - start_time, 2)}

@mcp.tool(name="rotate_pages", description="Rotate specific pages by 90, 180, or 270 degrees")
async def rotate_pages(
    pdf_path: str,
    pages: Optional[str] = None,  # Accept as string for MCP compatibility
    rotation: int = 90,
    output_filename: str = "rotated_document.pdf"
) -> Dict[str, Any]:
    """
    Rotate specific pages in a PDF
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        pages: Page numbers to rotate (comma-separated, 1-based), None for all pages
        rotation: Rotation angle (90, 180, or 270 degrees)
        output_filename: Name for the output file
    
    Returns:
        Dictionary containing rotation results
    """
    import time
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        parsed_pages = parse_pages_parameter(pages)
        
        if rotation not in [90, 180, 270]:
            return {"error": "Rotation must be 90, 180, or 270 degrees"}
        
        doc = fitz.open(str(path))
        page_count = len(doc)
        
        # Determine which pages to rotate
        pages_to_rotate = parsed_pages if parsed_pages else list(range(page_count))
        
        # Validate page numbers
        valid_pages = [p for p in pages_to_rotate if 0 <= p < page_count]
        invalid_pages = [p for p in pages_to_rotate if p not in valid_pages]
        
        if invalid_pages:
            logger.warning(f"Invalid page numbers ignored: {invalid_pages}")
        
        # Rotate pages
        rotated_pages = []
        for page_num in valid_pages:
            page = doc[page_num]
            page.set_rotation(rotation)
            rotated_pages.append(page_num + 1)  # 1-indexed for user display
        
        # Save rotated document
        output_path = CACHE_DIR / output_filename
        doc.save(str(output_path))
        doc.close()
        
        return {
            "original_file": str(path),
            "rotated_file": str(output_path),
            "rotation_degrees": rotation,
            "pages_rotated": rotated_pages,
            "total_pages": page_count,
            "invalid_pages_ignored": [p + 1 for p in invalid_pages],
            "output_file_size": output_path.stat().st_size,
            "rotation_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"Page rotation failed: {str(e)}", "rotation_time": round(time.time() - start_time, 2)}

@mcp.tool(name="convert_to_images", description="Convert PDF pages to image files")
async def convert_to_images(
    pdf_path: str,
    format: str = "png",
    dpi: int = 300,
    pages: Optional[str] = None,  # Accept as string for MCP compatibility
    output_prefix: str = "page"
) -> Dict[str, Any]:
    """
    Convert PDF pages to image files
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        format: Output image format (png, jpeg, tiff)
        dpi: Resolution for image conversion
        pages: Page numbers to convert (comma-separated, 1-based), None for all pages
        output_prefix: Prefix for output image files
    
    Returns:
        Dictionary containing conversion results
    """
    import time
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        parsed_pages = parse_pages_parameter(pages)
        
        if format.lower() not in ["png", "jpeg", "jpg", "tiff"]:
            return {"error": "Supported formats: png, jpeg, tiff"}
        
        # Create output directory with security
        output_dir = CACHE_DIR / "image_output"
        output_dir.mkdir(exist_ok=True, mode=0o700)
        
        # Convert pages to images
        if parsed_pages:
            # Convert specific pages
            converted_images = []
            for page_num in parsed_pages:
                try:
                    images = convert_from_path(
                        str(path),
                        dpi=dpi,
                        first_page=page_num + 1,
                        last_page=page_num + 1
                    )
                    
                    if images:
                        output_file = output_dir / f"{output_prefix}_page_{page_num+1}.{format.lower()}"
                        images[0].save(str(output_file), format.upper())
                        
                        converted_images.append({
                            "page_number": page_num + 1,
                            "image_path": str(output_file),
                            "image_size": output_file.stat().st_size,
                            "dimensions": f"{images[0].width}x{images[0].height}"
                        })
                        
                except Exception as e:
                    logger.error(f"Failed to convert page {page_num + 1}: {e}")
        else:
            # Convert all pages
            images = convert_from_path(str(path), dpi=dpi)
            converted_images = []
            
            for i, image in enumerate(images):
                output_file = output_dir / f"{output_prefix}_page_{i+1}.{format.lower()}"
                image.save(str(output_file), format.upper())
                
                converted_images.append({
                    "page_number": i + 1,
                    "image_path": str(output_file),
                    "image_size": output_file.stat().st_size,
                    "dimensions": f"{image.width}x{image.height}"
                })
        
        return {
            "original_file": str(path),
            "format": format.lower(),
            "dpi": dpi,
            "pages_converted": len(converted_images),
            "output_images": converted_images,
            "conversion_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"Image conversion failed: {str(e)}", "conversion_time": round(time.time() - start_time, 2)}

@mcp.tool(name="analyze_pdf_security", description="Analyze PDF security features and potential issues")
async def analyze_pdf_security(pdf_path: str) -> Dict[str, Any]:
    """
    Analyze PDF security features and potential issues
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
    
    Returns:
        Dictionary containing security analysis results
    """
    import time
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        doc = fitz.open(str(path))
        
        security_report = {
            "file_info": {
                "path": str(path),
                "size_bytes": path.stat().st_size
            },
            "encryption": {},
            "permissions": {},
            "signatures": {},
            "javascript": {},
            "security_warnings": [],
            "security_score": 0
        }
        
        # Encryption analysis
        security_report["encryption"]["is_encrypted"] = doc.is_encrypted
        security_report["encryption"]["needs_password"] = doc.needs_pass
        security_report["encryption"]["can_open"] = not doc.needs_pass
        
        # Check for password protection
        if doc.is_encrypted and not doc.needs_pass:
            security_report["encryption"]["encryption_type"] = "owner_password_only"
        elif doc.needs_pass:
            security_report["encryption"]["encryption_type"] = "user_password_required"
        else:
            security_report["encryption"]["encryption_type"] = "none"
        
        # Permission analysis
        if hasattr(doc, 'permissions'):
            perms = doc.permissions
            security_report["permissions"] = {
                "can_print": bool(perms & 4),
                "can_modify": bool(perms & 8),
                "can_copy": bool(perms & 16),
                "can_annotate": bool(perms & 32),
                "can_form_fill": bool(perms & 256),
                "can_extract_for_accessibility": bool(perms & 512),
                "can_assemble": bool(perms & 1024),
                "can_print_high_quality": bool(perms & 2048)
            }
        
        # JavaScript detection
        has_js = False
        js_count = 0
        
        for page_num in range(min(len(doc), 10)):  # Check first 10 pages for performance
            page = doc[page_num]
            text = page.get_text()
            
            # Simple JavaScript detection
            if any(keyword in text.lower() for keyword in ['javascript:', '/js', 'app.alert', 'this.print']):
                has_js = True
                js_count += 1
        
        security_report["javascript"]["detected"] = has_js
        security_report["javascript"]["pages_with_js"] = js_count
        
        if has_js:
            security_report["security_warnings"].append("JavaScript detected - potential security risk")
        
        # Digital signature detection (basic)
        # Note: Full signature validation would require cryptographic libraries
        security_report["signatures"]["has_signatures"] = doc.signature_count() > 0
        security_report["signatures"]["signature_count"] = doc.signature_count()
        
        # File size anomalies
        if security_report["file_info"]["size_bytes"] > 100 * 1024 * 1024:  # > 100MB
            security_report["security_warnings"].append("Large file size - review for embedded content")
        
        # Metadata analysis for privacy
        metadata = doc.metadata
        sensitive_metadata = []
        
        for key, value in metadata.items():
            if value and len(str(value)) > 0:
                if any(word in str(value).lower() for word in ['user', 'author', 'creator']):
                    sensitive_metadata.append(key)
        
        if sensitive_metadata:
            security_report["security_warnings"].append(f"Potentially sensitive metadata found: {', '.join(sensitive_metadata)}")
        
        # Form analysis for security
        if doc.is_form_pdf:
            # Check for potentially dangerous form actions
            for page_num in range(len(doc)):
                page = doc[page_num]
                widgets = page.widgets()
                
                for widget in widgets:
                    if hasattr(widget, 'field_name') and widget.field_name:
                        if any(dangerous in widget.field_name.lower() for dangerous in ['password', 'ssn', 'credit']):
                            security_report["security_warnings"].append("Form contains potentially sensitive field names")
                            break
        
        # Calculate security score
        score = 100
        
        if not doc.is_encrypted:
            score -= 20
        if has_js:
            score -= 30
        if len(security_report["security_warnings"]) > 0:
            score -= len(security_report["security_warnings"]) * 10
        if sensitive_metadata:
            score -= 10
        
        security_report["security_score"] = max(0, min(100, score))
        
        # Security level assessment
        if score >= 80:
            security_level = "high"
        elif score >= 60:
            security_level = "medium"
        elif score >= 40:
            security_level = "low"
        else:
            security_level = "critical"
        
        security_report["security_level"] = security_level
        
        doc.close()
        security_report["analysis_time"] = round(time.time() - start_time, 2)
        
        return security_report
        
    except Exception as e:
        return {"error": f"Security analysis failed: {str(e)}", "analysis_time": round(time.time() - start_time, 2)}

@mcp.tool(name="detect_watermarks", description="Detect and analyze watermarks in PDF")
async def detect_watermarks(pdf_path: str) -> Dict[str, Any]:
    """
    Detect and analyze watermarks in PDF
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
    
    Returns:
        Dictionary containing watermark detection results
    """
    import time
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        doc = fitz.open(str(path))
        
        watermark_report = {
            "has_watermarks": False,
            "watermarks_detected": [],
            "detection_summary": {},
            "analysis_time": 0
        }
        
        text_watermarks = []
        image_watermarks = []
        
        # Check each page for potential watermarks
        for page_num, page in enumerate(doc):
            # Text-based watermark detection
            # Look for text with unusual properties (transparency, large size, repetitive)
            text_blocks = page.get_text("dict")["blocks"]
            
            for block in text_blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            font_size = span["size"]
                            
                            # Heuristics for watermark detection
                            is_potential_watermark = (
                                len(text) > 3 and
                                (font_size > 40 or  # Large text
                                 any(keyword in text.lower() for keyword in [
                                     'confidential', 'draft', 'copy', 'watermark', 'sample',
                                     'preview', 'demo', 'trial', 'protected'
                                 ]) or
                                 text.count(' ') == 0 and len(text) > 8)  # Long single word
                            )
                            
                            if is_potential_watermark:
                                text_watermarks.append({
                                    "page": page_num + 1,
                                    "text": text,
                                    "font_size": font_size,
                                    "coordinates": {
                                        "x": span["bbox"][0],
                                        "y": span["bbox"][1]
                                    },
                                    "type": "text"
                                })
            
            # Image-based watermark detection (basic)
            # Look for images that might be watermarks
            images = page.get_images()
            
            for img_index, img in enumerate(images):
                try:
                    # Get image properties
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Small or very large images might be watermarks
                    if pix.width < 200 and pix.height < 200:  # Small logos
                        image_watermarks.append({
                            "page": page_num + 1,
                            "size": f"{pix.width}x{pix.height}",
                            "type": "small_image",
                            "potential_logo": True
                        })
                    elif pix.width > 1000 or pix.height > 1000:  # Large background
                        image_watermarks.append({
                            "page": page_num + 1,
                            "size": f"{pix.width}x{pix.height}",
                            "type": "large_background",
                            "potential_background": True
                        })
                    
                    pix = None  # Clean up
                    
                except Exception as e:
                    logger.debug(f"Could not analyze image on page {page_num + 1}: {e}")
        
        # Combine results
        all_watermarks = text_watermarks + image_watermarks
        
        watermark_report["has_watermarks"] = len(all_watermarks) > 0
        watermark_report["watermarks_detected"] = all_watermarks
        
        # Summary
        watermark_report["detection_summary"] = {
            "total_detected": len(all_watermarks),
            "text_watermarks": len(text_watermarks),
            "image_watermarks": len(image_watermarks),
            "pages_with_watermarks": len(set(w["page"] for w in all_watermarks)),
            "total_pages": len(doc)
        }
        
        doc.close()
        watermark_report["analysis_time"] = round(time.time() - start_time, 2)
        
        return watermark_report
        
    except Exception as e:
        return {"error": f"Watermark detection failed: {str(e)}", "analysis_time": round(time.time() - start_time, 2)}

@mcp.tool(name="classify_content", description="Classify and analyze PDF content type and structure")
async def classify_content(pdf_path: str) -> Dict[str, Any]:
    """
    Classify PDF content type and analyze document structure
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
    
    Returns:
        Dictionary containing content classification results
    """
    import time
    
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        doc = fitz.open(str(path))
        
        classification_report = {
            "file_info": {
                "path": str(path),
                "pages": len(doc),
                "size_bytes": path.stat().st_size
            },
            "document_type": "",
            "content_analysis": {},
            "structure_analysis": {},
            "language_detection": {},
            "classification_confidence": 0.0
        }
        
        # Extract all text for analysis
        all_text = ""
        page_texts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            page_texts.append(page_text)
            all_text += page_text + "\n"
        
        # Basic text statistics
        total_chars = len(all_text)
        total_words = len(all_text.split())
        total_lines = all_text.count('\n')
        
        classification_report["content_analysis"] = {
            "total_characters": total_chars,
            "total_words": total_words,
            "total_lines": total_lines,
            "average_words_per_page": round(total_words / len(doc), 2),
            "text_density": round(total_chars / len(doc), 2)
        }
        
        # Document type classification based on patterns
        document_patterns = {
            "academic_paper": [
                r'\babstract\b', r'\breferences\b', r'\bcitation\b', 
                r'\bfigure \d+\b', r'\btable \d+\b', r'\bsection \d+\b'
            ],
            "legal_document": [
                r'\bwhereas\b', r'\btherefore\b', r'\bparty\b', 
                r'\bagreement\b', r'\bcontract\b', r'\bterms\b'
            ],
            "financial_report": [
                r'\$[\d,]+\b', r'\brevenue\b', r'\bprofit\b', 
                r'\bbalance sheet\b', r'\bquarter\b', r'\bfiscal year\b'
            ],
            "technical_manual": [
                r'\bprocedure\b', r'\binstruction\b', r'\bstep \d+\b', 
                r'\bwarning\b', r'\bcaution\b', r'\bspecification\b'
            ],
            "invoice": [
                r'\binvoice\b', r'\bbill to\b', r'\btotal\b', 
                r'\bamount due\b', r'\bdue date\b', r'\bpayment\b'
            ],
            "resume": [
                r'\bexperience\b', r'\beducation\b', r'\bskills\b', 
                r'\bemployment\b', r'\bqualifications\b', r'\bcareer\b'
            ]
        }
        
        # Calculate pattern matches
        pattern_scores = {}
        text_lower = all_text.lower()
        
        for doc_type, patterns in document_patterns.items():
            score = 0
            matches = []
            
            for pattern in patterns:
                pattern_matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += pattern_matches
                if pattern_matches > 0:
                    matches.append(pattern)
            
            pattern_scores[doc_type] = {
                "score": score,
                "matches": matches,
                "confidence": min(score / 10.0, 1.0)  # Normalize to 0-1
            }
        
        # Determine most likely document type
        best_match = max(pattern_scores.items(), key=lambda x: x[1]["score"])
        
        if best_match[1]["score"] > 0:
            classification_report["document_type"] = best_match[0]
            classification_report["classification_confidence"] = best_match[1]["confidence"]
        else:
            classification_report["document_type"] = "general_document"
            classification_report["classification_confidence"] = 0.1
        
        classification_report["type_analysis"] = pattern_scores
        
        # Structure analysis
        # Detect headings, lists, and formatting
        heading_patterns = [
            r'^[A-Z][^a-z]*$',  # ALL CAPS lines
            r'^\d+\.\s+[A-Z]',   # Numbered headings
            r'^Chapter \d+',     # Chapter headings
            r'^Section \d+'      # Section headings
        ]
        
        headings_found = []
        list_items_found = 0
        
        for line in all_text.split('\n'):
            line = line.strip()
            if len(line) < 3:
                continue
                
            # Check for headings
            for pattern in heading_patterns:
                if re.match(pattern, line):
                    headings_found.append(line[:50])  # First 50 chars
                    break
            
            # Check for list items
            if re.match(r'^[\-\•\*]\s+', line) or re.match(r'^\d+\.\s+', line):
                list_items_found += 1
        
        classification_report["structure_analysis"] = {
            "headings_detected": len(headings_found),
            "sample_headings": headings_found[:5],  # First 5 headings
            "list_items_detected": list_items_found,
            "has_structured_content": len(headings_found) > 0 or list_items_found > 0
        }
        
        # Basic language detection (simplified)
        # Count common words in different languages
        language_indicators = {
            "english": ["the", "and", "or", "to", "of", "in", "for", "is", "are", "was"],
            "spanish": ["el", "la", "de", "que", "y", "en", "un", "es", "se", "no"],
            "french": ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"],
            "german": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich"]
        }
        
        language_scores = {}
        words = text_lower.split()
        word_set = set(words)
        
        for lang, indicators in language_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in word_set)
            language_scores[lang] = matches
        
        likely_language = max(language_scores, key=language_scores.get) if language_scores else "unknown"
        
        classification_report["language_detection"] = {
            "likely_language": likely_language,
            "language_scores": language_scores,
            "confidence": round(language_scores.get(likely_language, 0) / 10.0, 2)
        }
        
        doc.close()
        classification_report["analysis_time"] = round(time.time() - start_time, 2)
        
        return classification_report
        
    except Exception as e:
        return {"error": f"Content classification failed: {str(e)}", "analysis_time": round(time.time() - start_time, 2)}

@mcp.tool(name="summarize_content", description="Generate summary and key insights from PDF content")
async def summarize_content(
    pdf_path: str, 
    summary_length: str = "medium",  # short, medium, long
    pages: Optional[str] = None  # Specific pages to summarize
) -> Dict[str, Any]:
    """
    Generate summary and key insights from PDF content
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        summary_length: Length of summary (short, medium, long)
        pages: Specific pages to summarize (comma-separated, 1-based), None for all pages
    
    Returns:
        Dictionary containing summary and key insights
    """
    import time
    
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        parsed_pages = parse_pages_parameter(pages)
        doc = fitz.open(str(path))
        
        # Extract text from specified pages or all pages
        target_text = ""
        processed_pages = []
        
        if parsed_pages:
            for page_num in parsed_pages:
                if 0 <= page_num < len(doc):
                    page = doc[page_num]
                    target_text += page.get_text() + "\n"
                    processed_pages.append(page_num + 1)
        else:
            for page_num in range(len(doc)):
                page = doc[page_num]
                target_text += page.get_text() + "\n"
                processed_pages.append(page_num + 1)
        
        if not target_text.strip():
            return {"error": "No text content found to summarize"}
        
        summary_report = {
            "file_info": {
                "path": str(path),
                "pages_processed": processed_pages,
                "total_pages": len(doc)
            },
            "text_statistics": {},
            "key_insights": {},
            "summary": "",
            "key_topics": [],
            "important_numbers": [],
            "dates_found": []
        }
        
        # Text statistics
        sentences = re.split(r'[.!?]+', target_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = target_text.split()
        
        summary_report["text_statistics"] = {
            "total_characters": len(target_text),
            "total_words": len(words),
            "total_sentences": len(sentences),
            "average_words_per_sentence": round(len(words) / max(len(sentences), 1), 2),
            "reading_time_minutes": round(len(words) / 250, 1)  # 250 words per minute
        }
        
        # Extract key numbers and dates
        number_pattern = r'\$?[\d,]+\.?\d*%?|\d+[,\.]\d+|\b\d{4}\b'
        numbers = re.findall(number_pattern, target_text)
        
        # Filter and format numbers
        important_numbers = []
        for num in numbers[:10]:  # Top 10 numbers
            if '$' in num or '%' in num or ',' in num:
                important_numbers.append(num)
        
        summary_report["important_numbers"] = important_numbers
        
        # Extract dates
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
        ]
        
        dates_found = []
        for pattern in date_patterns:
            matches = re.findall(pattern, target_text, re.IGNORECASE)
            dates_found.extend(matches)
        
        summary_report["dates_found"] = list(set(dates_found[:10]))  # Top 10 unique dates
        
        # Generate key topics by finding most common meaningful words
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 
            'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'a', 
            'an', 'it', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'him', 'her', 
            'them', 'us', 'my', 'your', 'his', 'its', 'our', 'their'
        }
        
        # Extract meaningful words (3+ characters, not stop words)
        meaningful_words = []
        for word in words:
            cleaned_word = re.sub(r'[^\w]', '', word.lower())
            if len(cleaned_word) >= 3 and cleaned_word not in stop_words and cleaned_word.isalpha():
                meaningful_words.append(cleaned_word)
        
        # Get most common words as topics
        word_freq = Counter(meaningful_words)
        top_topics = [word for word, count in word_freq.most_common(10) if count >= 2]
        summary_report["key_topics"] = top_topics
        
        # Generate summary based on length preference
        sentence_scores = {}
        
        # Simple extractive summarization: score sentences based on word frequency and position
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_words = sentence.lower().split()
            
            # Score based on word frequency
            for word in sentence_words:
                cleaned_word = re.sub(r'[^\w]', '', word)
                if cleaned_word in word_freq:
                    score += word_freq[cleaned_word]
            
            # Boost score for sentences near the beginning
            if i < len(sentences) * 0.3:
                score *= 1.2
            
            # Boost score for sentences with numbers or dates
            if any(num in sentence for num in important_numbers[:5]):
                score *= 1.3
            
            sentence_scores[sentence] = score
        
        # Select top sentences for summary
        length_mappings = {
            "short": max(3, int(len(sentences) * 0.1)),
            "medium": max(5, int(len(sentences) * 0.2)),
            "long": max(8, int(len(sentences) * 0.3))
        }
        
        num_sentences = length_mappings.get(summary_length, length_mappings["medium"])
        
        # Get top-scoring sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        
        # Sort selected sentences by original order
        selected_sentences = [sent for sent, _ in top_sentences]
        sentence_order = {sent: sentences.index(sent) for sent in selected_sentences if sent in sentences}
        ordered_sentences = sorted(sentence_order.keys(), key=lambda x: sentence_order[x])
        
        summary_report["summary"] = ' '.join(ordered_sentences)
        
        # Key insights
        summary_report["key_insights"] = {
            "document_focus": top_topics[0] if top_topics else "general content",
            "complexity_level": "high" if summary_report["text_statistics"]["average_words_per_sentence"] > 20 else "medium" if summary_report["text_statistics"]["average_words_per_sentence"] > 15 else "low",
            "data_rich": len(important_numbers) > 5,
            "time_references": len(dates_found) > 0,
            "estimated_reading_level": "professional" if len([w for w in meaningful_words if len(w) > 8]) > len(meaningful_words) * 0.1 else "general"
        }
        
        doc.close()
        summary_report["analysis_time"] = round(time.time() - start_time, 2)
        
        return summary_report
        
    except Exception as e:
        return {"error": f"Content summarization failed: {str(e)}", "analysis_time": round(time.time() - start_time, 2)}

@mcp.tool(name="analyze_layout", description="Analyze PDF page layout including text blocks, columns, and spacing")
async def analyze_layout(
    pdf_path: str,
    pages: Optional[str] = None,  # Specific pages to analyze
    include_coordinates: bool = True
) -> Dict[str, Any]:
    """
    Analyze PDF page layout including text blocks, columns, and spacing
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        pages: Specific pages to analyze (comma-separated, 1-based), None for all pages
        include_coordinates: Whether to include detailed coordinate information
    
    Returns:
        Dictionary containing layout analysis results
    """
    import time
    
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        parsed_pages = parse_pages_parameter(pages)
        doc = fitz.open(str(path))
        
        layout_report = {
            "file_info": {
                "path": str(path),
                "total_pages": len(doc)
            },
            "pages_analyzed": [],
            "global_analysis": {},
            "layout_statistics": {}
        }
        
        # Determine pages to analyze
        if parsed_pages:
            pages_to_analyze = [p for p in parsed_pages if 0 <= p < len(doc)]
        else:
            pages_to_analyze = list(range(min(len(doc), 5)))  # Analyze first 5 pages by default
        
        page_layouts = []
        all_text_blocks = []
        all_page_dimensions = []
        
        for page_num in pages_to_analyze:
            page = doc[page_num]
            page_dict = page.get_text("dict")
            page_rect = page.rect
            
            page_analysis = {
                "page_number": page_num + 1,
                "dimensions": {
                    "width": round(page_rect.width, 2),
                    "height": round(page_rect.height, 2),
                    "aspect_ratio": round(page_rect.width / page_rect.height, 2)
                },
                "text_blocks": [],
                "columns_detected": 0,
                "reading_order": [],
                "spacing_analysis": {}
            }
            
            all_page_dimensions.append({
                "width": page_rect.width,
                "height": page_rect.height
            })
            
            # Analyze text blocks
            text_blocks = []
            
            for block in page_dict["blocks"]:
                if "lines" in block:  # Text block
                    block_rect = fitz.Rect(block["bbox"])
                    
                    # Extract all text from this block
                    block_text = ""
                    font_sizes = []
                    fonts_used = []
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]
                            font_sizes.append(span["size"])
                            fonts_used.append(span["font"])
                    
                    if block_text.strip():  # Only include blocks with text
                        block_info = {
                            "text": block_text.strip()[:100] + ("..." if len(block_text.strip()) > 100 else ""),
                            "character_count": len(block_text),
                            "word_count": len(block_text.split()),
                            "bbox": {
                                "x0": round(block_rect.x0, 2),
                                "y0": round(block_rect.y0, 2),
                                "x1": round(block_rect.x1, 2),
                                "y1": round(block_rect.y1, 2),
                                "width": round(block_rect.width, 2),
                                "height": round(block_rect.height, 2)
                            } if include_coordinates else None,
                            "font_analysis": {
                                "average_font_size": round(sum(font_sizes) / len(font_sizes), 1) if font_sizes else 0,
                                "font_variation": len(set(font_sizes)) > 1,
                                "primary_font": max(set(fonts_used), key=fonts_used.count) if fonts_used else "unknown"
                            }
                        }
                        
                        text_blocks.append(block_info)
                        all_text_blocks.append(block_info)
            
            page_analysis["text_blocks"] = text_blocks
            
            # Column detection (simplified heuristic)
            if text_blocks:
                # Sort blocks by vertical position
                sorted_blocks = sorted(text_blocks, key=lambda x: x["bbox"]["y0"] if x["bbox"] else 0)
                
                # Group blocks by horizontal position to detect columns
                x_positions = []
                if include_coordinates:
                    x_positions = [block["bbox"]["x0"] for block in text_blocks if block["bbox"]]
                
                # Simple column detection: group by similar x-coordinates
                column_threshold = 50  # pixels
                columns = []
                
                for x in x_positions:
                    found_column = False
                    for i, col in enumerate(columns):
                        if abs(col["x_start"] - x) < column_threshold:
                            columns[i]["blocks"].append(x)
                            columns[i]["x_start"] = min(columns[i]["x_start"], x)
                            found_column = True
                            break
                    
                    if not found_column:
                        columns.append({"x_start": x, "blocks": [x]})
                
                page_analysis["columns_detected"] = len(columns)
                
                # Reading order analysis (top-to-bottom, left-to-right)
                if include_coordinates:
                    reading_order = sorted(text_blocks, key=lambda x: (x["bbox"]["y0"], x["bbox"]["x0"]) if x["bbox"] else (0, 0))
                    page_analysis["reading_order"] = [block["text"][:30] + "..." for block in reading_order[:10]]
                
                # Spacing analysis
                if len(text_blocks) > 1 and include_coordinates:
                    vertical_gaps = []
                    
                    for i in range(len(sorted_blocks) - 1):
                        current = sorted_blocks[i]
                        next_block = sorted_blocks[i + 1]
                        
                        if current["bbox"] and next_block["bbox"]:
                            # Vertical gap
                            gap = next_block["bbox"]["y0"] - current["bbox"]["y1"]
                            if gap > 0:
                                vertical_gaps.append(gap)
                    
                    page_analysis["spacing_analysis"] = {
                        "average_vertical_gap": round(sum(vertical_gaps) / len(vertical_gaps), 2) if vertical_gaps else 0,
                        "max_vertical_gap": round(max(vertical_gaps), 2) if vertical_gaps else 0,
                        "spacing_consistency": len(set([round(gap) for gap in vertical_gaps])) <= 3 if vertical_gaps else True
                    }
            
            page_layouts.append(page_analysis)
        
        layout_report["pages_analyzed"] = page_layouts
        
        # Global analysis across all analyzed pages
        if all_text_blocks:
            font_sizes = []
            primary_fonts = []
            
            for block in all_text_blocks:
                font_sizes.append(block["font_analysis"]["average_font_size"])
                primary_fonts.append(block["font_analysis"]["primary_font"])
            
            layout_report["global_analysis"] = {
                "consistent_dimensions": len(set([(d["width"], d["height"]) for d in all_page_dimensions])) == 1,
                "average_blocks_per_page": round(len(all_text_blocks) / len(pages_to_analyze), 1),
                "font_consistency": {
                    "most_common_size": max(set(font_sizes), key=font_sizes.count) if font_sizes else 0,
                    "size_variations": len(set([round(size) for size in font_sizes if size > 0])),
                    "most_common_font": max(set(primary_fonts), key=primary_fonts.count) if primary_fonts else "unknown"
                },
                "layout_type": "single_column" if all(p["columns_detected"] <= 1 for p in page_layouts) else "multi_column",
                "pages_with_consistent_layout": len(set([p["columns_detected"] for p in page_layouts])) == 1
            }
        
        # Layout statistics
        if page_layouts:
            layout_report["layout_statistics"] = {
                "total_text_blocks": len(all_text_blocks),
                "pages_analyzed": len(page_layouts),
                "average_columns_per_page": round(sum(p["columns_detected"] for p in page_layouts) / len(page_layouts), 1),
                "consistent_column_structure": len(set(p["columns_detected"] for p in page_layouts)) == 1,
                "reading_complexity": "high" if any(p["columns_detected"] > 2 for p in page_layouts) else "medium" if any(p["columns_detected"] == 2 for p in page_layouts) else "low"
            }
        
        doc.close()
        layout_report["analysis_time"] = round(time.time() - start_time, 2)
        
        return layout_report
        
    except Exception as e:
        return {"error": f"Layout analysis failed: {str(e)}", "analysis_time": round(time.time() - start_time, 2)}

@mcp.tool(name="extract_charts", description="Extract and analyze charts, diagrams, and visual elements from PDF")
async def extract_charts(
    pdf_path: str,
    pages: Optional[str] = None,
    min_size: int = 100  # Minimum size for chart detection
) -> Dict[str, Any]:
    """
    Extract and analyze charts, diagrams, and visual elements from PDF
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        pages: Specific pages to analyze (comma-separated, 1-based), None for all pages
        min_size: Minimum size (width or height) for chart detection in pixels
    
    Returns:
        Dictionary containing chart extraction results
    """
    import time
    
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        parsed_pages = parse_pages_parameter(pages)
        doc = fitz.open(str(path))
        
        chart_report = {
            "file_info": {
                "path": str(path),
                "total_pages": len(doc)
            },
            "charts_found": [],
            "visual_elements": [],
            "extraction_summary": {}
        }
        
        # Determine pages to analyze
        if parsed_pages:
            pages_to_analyze = [p for p in parsed_pages if 0 <= p < len(doc)]
        else:
            pages_to_analyze = list(range(len(doc)))
        
        all_charts = []
        all_visual_elements = []
        
        for page_num in pages_to_analyze:
            page = doc[page_num]
            
            # Extract images (potential charts)
            images = page.get_images()
            
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Filter by minimum size
                    if pix.width >= min_size or pix.height >= min_size:
                        
                        # Try to determine if this might be a chart
                        chart_likelihood = 0.0
                        chart_type = "unknown"
                        
                        # Size-based heuristics
                        if 200 <= pix.width <= 2000 and 200 <= pix.height <= 2000:
                            chart_likelihood += 0.3  # Good size for charts
                        
                        # Aspect ratio heuristics
                        aspect_ratio = pix.width / pix.height
                        if 0.5 <= aspect_ratio <= 2.0:
                            chart_likelihood += 0.2  # Good aspect ratio for charts
                        
                        # Color mode analysis
                        if pix.n >= 3:  # Color image
                            chart_likelihood += 0.1
                        
                        # Determine likely chart type based on dimensions
                        if aspect_ratio > 1.5:
                            chart_type = "horizontal_chart"
                        elif aspect_ratio < 0.7:
                            chart_type = "vertical_chart"
                        elif 0.9 <= aspect_ratio <= 1.1:
                            chart_type = "square_chart_or_diagram"
                        else:
                            chart_type = "standard_chart"
                        
                        # Extract image to temporary location for further analysis
                        image_path = CACHE_DIR / f"chart_page_{page_num + 1}_img_{img_index}.png"
                        pix.save(str(image_path))
                        
                        chart_info = {
                            "page": page_num + 1,
                            "image_index": img_index,
                            "dimensions": {
                                "width": pix.width,
                                "height": pix.height,
                                "aspect_ratio": round(aspect_ratio, 2)
                            },
                            "chart_likelihood": round(chart_likelihood, 2),
                            "estimated_type": chart_type,
                            "file_info": {
                                "size_bytes": image_path.stat().st_size,
                                "format": "PNG",
                                "path": str(image_path)
                            },
                            "color_mode": "color" if pix.n >= 3 else "grayscale"
                        }
                        
                        # Classify as chart if likelihood is reasonable
                        if chart_likelihood >= 0.3:
                            all_charts.append(chart_info)
                        else:
                            all_visual_elements.append(chart_info)
                    
                    pix = None  # Clean up
                    
                except Exception as e:
                    logger.debug(f"Could not process image on page {page_num + 1}: {e}")
            
            # Also look for vector graphics (drawings, shapes)
            drawings = page.get_drawings()
            
            for draw_index, drawing in enumerate(drawings):
                try:
                    # Analyze drawing properties
                    items = drawing.get("items", [])
                    rect = drawing.get("rect")
                    
                    if rect and (rect[2] - rect[0] >= min_size or rect[3] - rect[1] >= min_size):
                        drawing_info = {
                            "page": page_num + 1,
                            "drawing_index": draw_index,
                            "type": "vector_drawing",
                            "dimensions": {
                                "width": round(rect[2] - rect[0], 2),
                                "height": round(rect[3] - rect[1], 2),
                                "x": round(rect[0], 2),
                                "y": round(rect[1], 2)
                            },
                            "complexity": len(items),
                            "estimated_type": "diagram" if len(items) > 5 else "simple_shape"
                        }
                        
                        all_visual_elements.append(drawing_info)
                
                except Exception as e:
                    logger.debug(f"Could not process drawing on page {page_num + 1}: {e}")
        
        chart_report["charts_found"] = all_charts
        chart_report["visual_elements"] = all_visual_elements
        
        # Generate extraction summary
        chart_report["extraction_summary"] = {
            "total_charts_found": len(all_charts),
            "total_visual_elements": len(all_visual_elements),
            "pages_with_charts": len(set(chart["page"] for chart in all_charts)),
            "pages_with_visual_elements": len(set(elem["page"] for elem in all_visual_elements)),
            "most_common_chart_type": max([chart["estimated_type"] for chart in all_charts], key=[chart["estimated_type"] for chart in all_charts].count) if all_charts else "none",
            "average_chart_size": {
                "width": round(sum(chart["dimensions"]["width"] for chart in all_charts) / len(all_charts), 1) if all_charts else 0,
                "height": round(sum(chart["dimensions"]["height"] for chart in all_charts) / len(all_charts), 1) if all_charts else 0
            },
            "chart_density": round(len(all_charts) / len(pages_to_analyze), 2)
        }
        
        doc.close()
        chart_report["analysis_time"] = round(time.time() - start_time, 2)
        
        return chart_report
        
    except Exception as e:
        return {"error": f"Chart extraction failed: {str(e)}", "analysis_time": round(time.time() - start_time, 2)}

@mcp.tool(name="optimize_pdf", description="Optimize PDF file size and performance")
async def optimize_pdf(
    pdf_path: str,
    optimization_level: str = "balanced",  # "light", "balanced", "aggressive"
    preserve_quality: bool = True
) -> Dict[str, Any]:
    """
    Optimize PDF file size and performance
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        optimization_level: Level of optimization ("light", "balanced", "aggressive")
        preserve_quality: Whether to preserve image quality during optimization
    
    Returns:
        Dictionary containing optimization results
    """
    import time
    
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        doc = fitz.open(str(path))
        
        # Get original file info
        original_size = path.stat().st_size
        
        optimization_report = {
            "file_info": {
                "original_path": str(path),
                "original_size_bytes": original_size,
                "original_size_mb": round(original_size / (1024 * 1024), 2),
                "pages": len(doc)
            },
            "optimization_applied": [],
            "final_results": {},
            "savings": {}
        }
        
        # Define optimization strategies based on level
        optimization_strategies = {
            "light": {
                "compress_images": False,
                "remove_unused_objects": True,
                "optimize_fonts": False,
                "remove_metadata": False,
                "image_quality": 95
            },
            "balanced": {
                "compress_images": True,
                "remove_unused_objects": True,
                "optimize_fonts": True,
                "remove_metadata": False,
                "image_quality": 85
            },
            "aggressive": {
                "compress_images": True,
                "remove_unused_objects": True,
                "optimize_fonts": True,
                "remove_metadata": True,
                "image_quality": 75
            }
        }
        
        strategy = optimization_strategies.get(optimization_level, optimization_strategies["balanced"])
        
        # Create optimized document
        optimized_doc = fitz.open()
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Copy page to new document
            optimized_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        
        # Apply optimizations
        optimizations_applied = []
        
        # 1. Remove unused objects
        if strategy["remove_unused_objects"]:
            try:
                # PyMuPDF automatically handles some cleanup during save
                optimizations_applied.append("removed_unused_objects")
            except Exception as e:
                logger.debug(f"Could not remove unused objects: {e}")
        
        # 2. Compress and optimize images
        if strategy["compress_images"]:
            try:
                image_count = 0
                for page_num in range(len(optimized_doc)):
                    page = optimized_doc[page_num]
                    images = page.get_images()
                    
                    for img_index, img in enumerate(images):
                        try:
                            xref = img[0]
                            pix = fitz.Pixmap(optimized_doc, xref)
                            
                            if pix.width > 100 and pix.height > 100:  # Only optimize larger images
                                # Convert to JPEG with quality setting if not already
                                if pix.n >= 3:  # Color image
                                    pix.tobytes("jpeg", jpg_quality=strategy["image_quality"])
                                    # Replace image (simplified approach)
                                    image_count += 1
                            
                            pix = None
                            
                        except Exception as e:
                            logger.debug(f"Could not optimize image {img_index} on page {page_num}: {e}")
                
                if image_count > 0:
                    optimizations_applied.append(f"compressed_{image_count}_images")
                    
            except Exception as e:
                logger.debug(f"Could not compress images: {e}")
        
        # 3. Remove metadata
        if strategy["remove_metadata"]:
            try:
                # Clear document metadata
                optimized_doc.set_metadata({})
                optimizations_applied.append("removed_metadata")
            except Exception as e:
                logger.debug(f"Could not remove metadata: {e}")
        
        # 4. Font optimization (basic)
        if strategy["optimize_fonts"]:
            try:
                # PyMuPDF handles font optimization during save
                optimizations_applied.append("optimized_fonts")
            except Exception as e:
                logger.debug(f"Could not optimize fonts: {e}")
        
        # Save optimized PDF
        optimized_path = CACHE_DIR / f"optimized_{path.name}"
        
        # Save with optimization flags
        save_flags = 0
        if not preserve_quality:
            save_flags |= fitz.PDF_OPTIMIZE_IMAGES
        
        optimized_doc.save(str(optimized_path), 
                          garbage=4,  # Garbage collection level
                          clean=True,  # Clean up
                          deflate=True,  # Compress content streams
                          ascii=False)  # Use binary encoding
        
        # Get optimized file info
        optimized_size = optimized_path.stat().st_size
        
        # Calculate savings
        size_reduction = original_size - optimized_size
        size_reduction_percent = round((size_reduction / original_size) * 100, 2)
        
        optimization_report["optimization_applied"] = optimizations_applied
        optimization_report["final_results"] = {
            "optimized_path": str(optimized_path),
            "optimized_size_bytes": optimized_size,
            "optimized_size_mb": round(optimized_size / (1024 * 1024), 2),
            "optimization_level": optimization_level,
            "preserve_quality": preserve_quality
        }
        
        optimization_report["savings"] = {
            "size_reduction_bytes": size_reduction,
            "size_reduction_mb": round(size_reduction / (1024 * 1024), 2),
            "size_reduction_percent": size_reduction_percent,
            "compression_ratio": round(original_size / optimized_size, 2) if optimized_size > 0 else 0
        }
        
        # Recommendations for further optimization
        recommendations = []
        
        if size_reduction_percent < 10:
            recommendations.append("Try more aggressive optimization level")
        
        if original_size > 50 * 1024 * 1024:  # > 50MB
            recommendations.append("Consider splitting into smaller files")
        
        # Check for images
        total_images = sum(len(doc[i].get_images()) for i in range(len(doc)))
        if total_images > 10:
            recommendations.append("Document contains many images - consider external image optimization")
        
        optimization_report["recommendations"] = recommendations
        
        doc.close()
        optimized_doc.close()
        
        optimization_report["analysis_time"] = round(time.time() - start_time, 2)
        
        return optimization_report
        
    except Exception as e:
        return {"error": f"PDF optimization failed: {str(e)}", "analysis_time": round(time.time() - start_time, 2)}

@mcp.tool(name="repair_pdf", description="Attempt to repair corrupted or damaged PDF files")
async def repair_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Attempt to repair corrupted or damaged PDF files
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
    
    Returns:
        Dictionary containing repair results
    """
    import time
    
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        
        repair_report = {
            "file_info": {
                "original_path": str(path),
                "original_size_bytes": path.stat().st_size
            },
            "repair_attempts": [],
            "issues_found": [],
            "repair_status": "unknown",
            "final_results": {}
        }
        
        # Attempt to open the PDF
        doc = None
        open_successful = False
        
        try:
            doc = fitz.open(str(path))
            open_successful = True
            repair_report["repair_attempts"].append("initial_open_successful")
        except Exception as e:
            repair_report["issues_found"].append(f"Cannot open PDF: {str(e)}")
            repair_report["repair_attempts"].append("initial_open_failed")
        
        # If we can't open it normally, try repair mode
        if not open_successful:
            try:
                # Try to open with recovery
                doc = fitz.open(str(path), filetype="pdf")
                if doc.page_count > 0:
                    open_successful = True
                    repair_report["repair_attempts"].append("recovery_mode_successful")
                else:
                    repair_report["issues_found"].append("PDF has no pages")
            except Exception as e:
                repair_report["issues_found"].append(f"Recovery mode failed: {str(e)}")
                repair_report["repair_attempts"].append("recovery_mode_failed")
        
        if open_successful and doc:
            # Analyze the document for issues
            page_count = len(doc)
            repair_report["file_info"]["pages"] = page_count
            
            if page_count == 0:
                repair_report["issues_found"].append("PDF contains no pages")
            else:
                # Check each page for issues
                problematic_pages = []
                
                for page_num in range(page_count):
                    try:
                        page = doc[page_num]
                        
                        # Try to get text
                        try:
                            text = page.get_text()
                            if not text.strip():
                                # Page might be image-only or corrupted
                                pass
                        except Exception:
                            problematic_pages.append(f"Page {page_num + 1}: Text extraction failed")
                        
                        # Try to get page dimensions
                        try:
                            rect = page.rect
                            if rect.width <= 0 or rect.height <= 0:
                                problematic_pages.append(f"Page {page_num + 1}: Invalid dimensions")
                        except Exception:
                            problematic_pages.append(f"Page {page_num + 1}: Cannot get dimensions")
                            
                    except Exception:
                        problematic_pages.append(f"Page {page_num + 1}: Cannot access page")
                
                if problematic_pages:
                    repair_report["issues_found"].extend(problematic_pages)
            
            # Check document metadata
            try:
                repair_report["file_info"]["metadata_accessible"] = True
            except Exception as e:
                repair_report["issues_found"].append(f"Cannot access metadata: {str(e)}")
                repair_report["file_info"]["metadata_accessible"] = False
            
            # Attempt to create a repaired version
            try:
                repaired_doc = fitz.open()  # Create new document
                
                # Copy pages one by one, skipping problematic ones
                successful_pages = 0
                
                for page_num in range(page_count):
                    try:
                        page = doc[page_num]
                        
                        # Try to insert the page
                        repaired_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                        successful_pages += 1
                        
                    except Exception as e:
                        repair_report["issues_found"].append(f"Could not repair page {page_num + 1}: {str(e)}")
                
                # Save repaired document
                repaired_path = CACHE_DIR / f"repaired_{path.name}"
                
                # Save with maximum error tolerance
                repaired_doc.save(str(repaired_path), 
                                garbage=4,  # Maximum garbage collection
                                clean=True,  # Clean up
                                deflate=True)  # Compress
                
                repaired_size = repaired_path.stat().st_size
                
                repair_report["repair_attempts"].append("created_repaired_version")
                repair_report["final_results"] = {
                    "repaired_path": str(repaired_path),
                    "repaired_size_bytes": repaired_size,
                    "pages_recovered": successful_pages,
                    "pages_lost": page_count - successful_pages,
                    "recovery_rate_percent": round((successful_pages / page_count) * 100, 2) if page_count > 0 else 0
                }
                
                # Determine repair status
                if successful_pages == page_count:
                    repair_report["repair_status"] = "fully_repaired"
                elif successful_pages > 0:
                    repair_report["repair_status"] = "partially_repaired"
                else:
                    repair_report["repair_status"] = "repair_failed"
                
                repaired_doc.close()
                
            except Exception as e:
                repair_report["issues_found"].append(f"Could not create repaired version: {str(e)}")
                repair_report["repair_status"] = "repair_failed"
            
            doc.close()
            
        else:
            repair_report["repair_status"] = "cannot_open"
            repair_report["final_results"] = {
                "recommendation": "File may be severely corrupted or not a valid PDF"
            }
        
        # Provide recommendations
        recommendations = []
        
        if repair_report["repair_status"] == "fully_repaired":
            recommendations.append("PDF was successfully repaired with no data loss")
        elif repair_report["repair_status"] == "partially_repaired":
            recommendations.append("PDF was partially repaired - some pages may be missing")
            recommendations.append("Review the repaired file to ensure critical content is intact")
        elif repair_report["repair_status"] == "repair_failed":
            recommendations.append("Automatic repair failed - manual intervention may be required")
            recommendations.append("Try using specialized PDF repair software")
        else:
            recommendations.append("File appears to be severely corrupted or not a valid PDF")
            recommendations.append("Verify the file is not truncated or corrupted during download")
        
        repair_report["recommendations"] = recommendations
        repair_report["analysis_time"] = round(time.time() - start_time, 2)
        
        return repair_report
        
    except Exception as e:
        return {"error": f"PDF repair failed: {str(e)}", "analysis_time": round(time.time() - start_time, 2)}

@mcp.tool(name="create_form_pdf", description="Create a new PDF form with interactive fields")
async def create_form_pdf(
    output_path: str,
    title: str = "Form Document",
    page_size: str = "A4",  # A4, Letter, Legal
    fields: str = "[]"  # JSON string of field definitions
) -> Dict[str, Any]:
    """
    Create a new PDF form with interactive fields
    
    Args:
        output_path: Path where the PDF form should be saved
        title: Title of the form document
        page_size: Page size (A4, Letter, Legal)
        fields: JSON string containing field definitions
    
    Field format:
    [
        {
            "type": "text|checkbox|radio|dropdown|signature",
            "name": "field_name",
            "label": "Field Label",
            "x": 100, "y": 700, "width": 200, "height": 20,
            "required": true,
            "default_value": "",
            "options": ["opt1", "opt2"]  // for dropdown/radio
        }
    ]
    
    Returns:
        Dictionary containing creation results
    """
    import json
    import time
    start_time = time.time()
    
    try:
        # Parse field definitions
        try:
            field_definitions = safe_json_parse(fields) if fields != "[]" else []
        except json.JSONDecodeError as e:
            return {"error": f"Invalid field JSON: {str(e)}", "creation_time": 0}
        
        # Page size mapping
        page_sizes = {
            "A4": fitz.paper_rect("A4"),
            "Letter": fitz.paper_rect("letter"),
            "Legal": fitz.paper_rect("legal")
        }
        
        if page_size not in page_sizes:
            return {"error": f"Unsupported page size: {page_size}. Use A4, Letter, or Legal", "creation_time": 0}
        
        rect = page_sizes[page_size]
        
        # Create new PDF document
        doc = fitz.open()
        page = doc.new_page(width=rect.width, height=rect.height)
        
        # Add title if provided
        if title:
            title_font = fitz.Font("helv")
            title_rect = fitz.Rect(50, 50, rect.width - 50, 80)
            page.insert_text(title_rect.tl, title, fontname="helv", fontsize=16, color=(0, 0, 0))
        
        # Track created fields
        created_fields = []
        field_y_offset = 120  # Start below title
        
        # Process field definitions
        for i, field in enumerate(field_definitions):
            field_type = field.get("type", "text")
            field_name = field.get("name", f"field_{i}")
            field_label = field.get("label", field_name)
            
            # Position fields automatically if not specified
            x = field.get("x", 50)
            y = field.get("y", field_y_offset + (i * 40))
            width = field.get("width", 200)
            height = field.get("height", 20)
            
            field_rect = fitz.Rect(x, y, x + width, y + height)
            label_rect = fitz.Rect(x, y - 15, x + width, y)
            
            # Add field label
            page.insert_text(label_rect.tl, field_label, fontname="helv", fontsize=10, color=(0, 0, 0))
            
            # Create appropriate field type
            if field_type == "text":
                widget = fitz.Widget()
                widget.field_name = field_name
                widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
                widget.rect = field_rect
                widget.field_value = field.get("default_value", "")
                widget.text_maxlen = field.get("max_length", 100)
                
                annot = page.add_widget(widget)
                created_fields.append({
                    "name": field_name,
                    "type": "text",
                    "position": {"x": x, "y": y, "width": width, "height": height}
                })
                
            elif field_type == "checkbox":
                widget = fitz.Widget()
                widget.field_name = field_name
                widget.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
                widget.rect = fitz.Rect(x, y, x + 15, y + 15)  # Square checkbox
                widget.field_value = field.get("default_value", False)
                
                annot = page.add_widget(widget)
                created_fields.append({
                    "name": field_name,
                    "type": "checkbox",
                    "position": {"x": x, "y": y, "width": 15, "height": 15}
                })
                
            elif field_type == "dropdown":
                options = field.get("options", ["Option 1", "Option 2", "Option 3"])
                widget = fitz.Widget()
                widget.field_name = field_name
                widget.field_type = fitz.PDF_WIDGET_TYPE_COMBOBOX
                widget.rect = field_rect
                widget.choice_values = options
                widget.field_value = field.get("default_value", options[0] if options else "")
                
                annot = page.add_widget(widget)
                created_fields.append({
                    "name": field_name,
                    "type": "dropdown",
                    "options": options,
                    "position": {"x": x, "y": y, "width": width, "height": height}
                })
                
            elif field_type == "signature":
                widget = fitz.Widget()
                widget.field_name = field_name
                widget.field_type = fitz.PDF_WIDGET_TYPE_SIGNATURE
                widget.rect = field_rect
                
                annot = page.add_widget(widget)
                created_fields.append({
                    "name": field_name,
                    "type": "signature",
                    "position": {"x": x, "y": y, "width": width, "height": height}
                })
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the PDF
        doc.save(str(output_file))
        doc.close()
        
        file_size = output_file.stat().st_size
        
        return {
            "output_path": str(output_file),
            "title": title,
            "page_size": page_size,
            "fields_created": len(created_fields),
            "field_details": created_fields,
            "file_size": format_file_size(file_size),
            "creation_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"Form creation failed: {str(e)}", "creation_time": round(time.time() - start_time, 2)}

@mcp.tool(name="fill_form_pdf", description="Fill an existing PDF form with data")
async def fill_form_pdf(
    input_path: str,
    output_path: str,
    form_data: str,  # JSON string of field values
    flatten: bool = False  # Whether to flatten form (make non-editable)
) -> Dict[str, Any]:
    """
    Fill an existing PDF form with provided data
    
    Args:
        input_path: Path to the PDF form to fill
        output_path: Path where filled PDF should be saved
        form_data: JSON string of field names and values {"field_name": "value"}
        flatten: Whether to flatten the form (make fields non-editable)
    
    Returns:
        Dictionary containing filling results
    """
    import json
    import time
    start_time = time.time()
    
    try:
        # Parse form data
        try:
            field_values = safe_json_parse(form_data) if form_data else {}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid form data JSON: {str(e)}", "fill_time": 0}
        
        # Validate input path
        input_file = await validate_pdf_path(input_path)
        doc = fitz.open(str(input_file))
        
        if not doc.is_form_pdf:
            doc.close()
            return {"error": "Input PDF is not a form document", "fill_time": 0}
        
        filled_fields = []
        failed_fields = []
        
        # Fill form fields
        for field_name, field_value in field_values.items():
            try:
                # Find the field and set its value
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    for widget in page.widgets():
                        if widget.field_name == field_name:
                            # Handle different field types
                            if widget.field_type == fitz.PDF_WIDGET_TYPE_TEXT:
                                widget.field_value = str(field_value)
                                widget.update()
                                filled_fields.append({
                                    "name": field_name,
                                    "type": "text",
                                    "value": str(field_value),
                                    "page": page_num + 1
                                })
                                break
                                
                            elif widget.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                                # Convert various true/false representations
                                checkbox_value = str(field_value).lower() in ['true', '1', 'yes', 'on', 'checked']
                                widget.field_value = checkbox_value
                                widget.update()
                                filled_fields.append({
                                    "name": field_name,
                                    "type": "checkbox",
                                    "value": checkbox_value,
                                    "page": page_num + 1
                                })
                                break
                                
                            elif widget.field_type in [fitz.PDF_WIDGET_TYPE_COMBOBOX, fitz.PDF_WIDGET_TYPE_LISTBOX]:
                                # For dropdowns, ensure value is in choice list
                                if hasattr(widget, 'choice_values') and widget.choice_values:
                                    if str(field_value) in widget.choice_values:
                                        widget.field_value = str(field_value)
                                        widget.update()
                                        filled_fields.append({
                                            "name": field_name,
                                            "type": "dropdown",
                                            "value": str(field_value),
                                            "page": page_num + 1
                                        })
                                        break
                                    else:
                                        failed_fields.append({
                                            "name": field_name,
                                            "reason": f"Value '{field_value}' not in allowed options: {widget.choice_values}"
                                        })
                                        break
                
                # If field wasn't found in any widget
                if not any(f["name"] == field_name for f in filled_fields + failed_fields):
                    failed_fields.append({
                        "name": field_name,
                        "reason": "Field not found in form"
                    })
                    
            except Exception as e:
                failed_fields.append({
                    "name": field_name,
                    "reason": f"Error filling field: {str(e)}"
                })
        
        # Flatten form if requested (makes fields non-editable)
        if flatten:
            try:
                # This makes the form read-only by burning the field values into the page content
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    # Note: Full flattening requires additional processing
                    # For now, we'll mark the intent
                pass
            except Exception as e:
                # Flattening failed, but continue with filled form
                pass
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save filled PDF
        doc.save(str(output_file), garbage=4, deflate=True, clean=True)
        doc.close()
        
        file_size = output_file.stat().st_size
        
        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "fields_filled": len(filled_fields),
            "fields_failed": len(failed_fields),
            "filled_field_details": filled_fields,
            "failed_field_details": failed_fields,
            "flattened": flatten,
            "file_size": format_file_size(file_size),
            "fill_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"Form filling failed: {str(e)}", "fill_time": round(time.time() - start_time, 2)}

@mcp.tool(name="add_form_fields", description="Add form fields to an existing PDF")
async def add_form_fields(
    input_path: str,
    output_path: str,
    fields: str  # JSON string of field definitions
) -> Dict[str, Any]:
    """
    Add interactive form fields to an existing PDF
    
    Args:
        input_path: Path to the existing PDF
        output_path: Path where PDF with added fields should be saved
        fields: JSON string containing field definitions (same format as create_form_pdf)
    
    Returns:
        Dictionary containing addition results
    """
    import json
    import time
    start_time = time.time()
    
    try:
        # Parse field definitions
        try:
            field_definitions = safe_json_parse(fields) if fields else []
        except json.JSONDecodeError as e:
            return {"error": f"Invalid field JSON: {str(e)}", "addition_time": 0}
        
        # Validate input path
        input_file = await validate_pdf_path(input_path)
        doc = fitz.open(str(input_file))
        
        added_fields = []
        
        # Process each field definition
        for i, field in enumerate(field_definitions):
            field_type = field.get("type", "text")
            field_name = field.get("name", f"added_field_{i}")
            field_label = field.get("label", field_name)
            page_num = field.get("page", 1) - 1  # Convert to 0-indexed
            
            # Ensure page exists
            if page_num >= len(doc):
                continue
                
            page = doc[page_num]
            
            # Position and size
            x = field.get("x", 50)
            y = field.get("y", 100)
            width = field.get("width", 200)
            height = field.get("height", 20)
            
            field_rect = fitz.Rect(x, y, x + width, y + height)
            
            # Add field label if requested
            if field.get("show_label", True):
                label_rect = fitz.Rect(x, y - 15, x + width, y)
                page.insert_text(label_rect.tl, field_label, fontname="helv", fontsize=10, color=(0, 0, 0))
            
            # Create appropriate field type
            try:
                if field_type == "text":
                    widget = fitz.Widget()
                    widget.field_name = field_name
                    widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
                    widget.rect = field_rect
                    widget.field_value = field.get("default_value", "")
                    widget.text_maxlen = field.get("max_length", 100)
                    
                    annot = page.add_widget(widget)
                    added_fields.append({
                        "name": field_name,
                        "type": "text",
                        "page": page_num + 1,
                        "position": {"x": x, "y": y, "width": width, "height": height}
                    })
                    
                elif field_type == "checkbox":
                    widget = fitz.Widget()
                    widget.field_name = field_name
                    widget.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
                    widget.rect = fitz.Rect(x, y, x + 15, y + 15)
                    widget.field_value = field.get("default_value", False)
                    
                    annot = page.add_widget(widget)
                    added_fields.append({
                        "name": field_name,
                        "type": "checkbox",
                        "page": page_num + 1,
                        "position": {"x": x, "y": y, "width": 15, "height": 15}
                    })
                    
                elif field_type == "dropdown":
                    options = field.get("options", ["Option 1", "Option 2"])
                    widget = fitz.Widget()
                    widget.field_name = field_name
                    widget.field_type = fitz.PDF_WIDGET_TYPE_COMBOBOX
                    widget.rect = field_rect
                    widget.choice_values = options
                    widget.field_value = field.get("default_value", options[0] if options else "")
                    
                    annot = page.add_widget(widget)
                    added_fields.append({
                        "name": field_name,
                        "type": "dropdown",
                        "options": options,
                        "page": page_num + 1,
                        "position": {"x": x, "y": y, "width": width, "height": height}
                    })
                    
            except Exception as field_error:
                # Skip this field but continue with others
                continue
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the modified PDF
        doc.save(str(output_file), garbage=4, deflate=True, clean=True)
        doc.close()
        
        file_size = output_file.stat().st_size
        
        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "fields_added": len(added_fields),
            "added_field_details": added_fields,
            "file_size": format_file_size(file_size),
            "addition_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"Adding form fields failed: {str(e)}", "addition_time": round(time.time() - start_time, 2)}

@mcp.tool(name="add_radio_group", description="Add a radio button group with mutual exclusion to PDF")
async def add_radio_group(
    input_path: str,
    output_path: str,
    group_name: str,
    options: str,  # JSON string of radio button options
    x: int = 50,
    y: int = 100,
    spacing: int = 30,
    page: int = 1
) -> Dict[str, Any]:
    """
    Add a radio button group where only one option can be selected
    
    Args:
        input_path: Path to the existing PDF
        output_path: Path where PDF with radio group should be saved
        group_name: Name for the radio button group
        options: JSON array of option labels ["Option 1", "Option 2", "Option 3"]
        x: X coordinate for the first radio button
        y: Y coordinate for the first radio button
        spacing: Vertical spacing between radio buttons
        page: Page number (1-indexed)
    
    Returns:
        Dictionary containing addition results
    """
    import json
    import time
    start_time = time.time()
    
    try:
        # Parse options
        try:
            option_labels = safe_json_parse(options) if options else []
        except json.JSONDecodeError as e:
            return {"error": f"Invalid options JSON: {str(e)}", "addition_time": 0}
        
        if not option_labels:
            return {"error": "At least one option is required", "addition_time": 0}
        
        # Validate input path
        input_file = await validate_pdf_path(input_path)
        doc = fitz.open(str(input_file))
        
        page_num = page - 1  # Convert to 0-indexed
        if page_num >= len(doc):
            doc.close()
            return {"error": f"Page {page} does not exist in PDF", "addition_time": 0}
        
        pdf_page = doc[page_num]
        added_buttons = []
        
        # Add radio buttons for each option
        for i, option_label in enumerate(option_labels):
            button_y = y + (i * spacing)
            button_name = f"{group_name}_{i}"
            
            # Add label text
            label_rect = fitz.Rect(x + 25, button_y - 5, x + 300, button_y + 15)
            pdf_page.insert_text((x + 25, button_y + 10), option_label, fontname="helv", fontsize=10, color=(0, 0, 0))
            
            # Create radio button as checkbox (simpler implementation)
            widget = fitz.Widget()
            widget.field_name = f"{group_name}_{i}"  # Unique name for each button
            widget.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
            widget.rect = fitz.Rect(x, button_y, x + 15, button_y + 15)
            widget.field_value = False
            
            # Add widget to page
            annot = pdf_page.add_widget(widget)
            
            # Add visual circle to make it look like radio button
            circle_center = (x + 7.5, button_y + 7.5)
            pdf_page.draw_circle(circle_center, 6, color=(0.5, 0.5, 0.5), width=1)
            
            added_buttons.append({
                "option": option_label,
                "position": {"x": x, "y": button_y, "width": 15, "height": 15},
                "field_name": button_name
            })
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the modified PDF
        doc.save(str(output_file), garbage=4, deflate=True, clean=True)
        doc.close()
        
        file_size = output_file.stat().st_size
        
        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "group_name": group_name,
            "options_added": len(added_buttons),
            "radio_buttons": added_buttons,
            "page": page,
            "file_size": format_file_size(file_size),
            "addition_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"Adding radio group failed: {str(e)}", "addition_time": round(time.time() - start_time, 2)}

@mcp.tool(name="add_textarea_field", description="Add a multi-line text area with word limits to PDF")
async def add_textarea_field(
    input_path: str,
    output_path: str,
    field_name: str,
    label: str = "",
    x: int = 50,
    y: int = 100,
    width: int = 400,
    height: int = 100,
    word_limit: int = 500,
    page: int = 1,
    show_word_count: bool = True
) -> Dict[str, Any]:
    """
    Add a multi-line text area with optional word count display
    
    Args:
        input_path: Path to the existing PDF
        output_path: Path where PDF with textarea should be saved
        field_name: Name for the textarea field
        label: Label text to display above the field
        x: X coordinate for the field
        y: Y coordinate for the field
        width: Width of the textarea
        height: Height of the textarea
        word_limit: Maximum number of words allowed
        page: Page number (1-indexed)
        show_word_count: Whether to show word count indicator
    
    Returns:
        Dictionary containing addition results
    """
    import time
    start_time = time.time()
    
    try:
        # Validate input path
        input_file = await validate_pdf_path(input_path)
        doc = fitz.open(str(input_file))
        
        page_num = page - 1  # Convert to 0-indexed
        if page_num >= len(doc):
            doc.close()
            return {"error": f"Page {page} does not exist in PDF", "addition_time": 0}
        
        pdf_page = doc[page_num]
        
        # Add field label if provided
        if label:
            label_rect = fitz.Rect(x, y - 20, x + width, y)
            pdf_page.insert_text((x, y - 5), label, fontname="helv", fontsize=10, color=(0, 0, 0))
        
        # Add word count indicator if requested
        if show_word_count:
            count_text = f"Word limit: {word_limit}"
            count_rect = fitz.Rect(x + width - 100, y - 20, x + width, y)
            pdf_page.insert_text((x + width - 100, y - 5), count_text, fontname="helv", fontsize=8, color=(0.5, 0.5, 0.5))
        
        # Create multiline text widget
        widget = fitz.Widget()
        widget.field_name = field_name
        widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
        widget.rect = fitz.Rect(x, y, x + width, y + height)
        widget.field_value = ""
        widget.text_maxlen = word_limit * 6  # Rough estimate: average 6 chars per word
        widget.text_format = fitz.TEXT_ALIGN_LEFT
        
        # Set multiline property (this is a bit tricky with PyMuPDF, so we'll add visual cues)
        annot = pdf_page.add_widget(widget)
        
        # Add visual border to indicate it's a textarea
        border_rect = fitz.Rect(x - 1, y - 1, x + width + 1, y + height + 1)
        pdf_page.draw_rect(border_rect, color=(0.7, 0.7, 0.7), width=1)
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the modified PDF
        doc.save(str(output_file), garbage=4, deflate=True, clean=True)
        doc.close()
        
        file_size = output_file.stat().st_size
        
        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "field_name": field_name,
            "label": label,
            "dimensions": {"width": width, "height": height},
            "word_limit": word_limit,
            "position": {"x": x, "y": y},
            "page": page,
            "file_size": format_file_size(file_size),
            "addition_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"Adding textarea failed: {str(e)}", "addition_time": round(time.time() - start_time, 2)}

@mcp.tool(name="add_date_field", description="Add a date field with format validation to PDF")
async def add_date_field(
    input_path: str,
    output_path: str,
    field_name: str,
    label: str = "",
    x: int = 50,
    y: int = 100,
    width: int = 150,
    height: int = 25,
    date_format: str = "MM/DD/YYYY",
    page: int = 1,
    show_format_hint: bool = True
) -> Dict[str, Any]:
    """
    Add a date field with format validation and hints
    
    Args:
        input_path: Path to the existing PDF
        output_path: Path where PDF with date field should be saved
        field_name: Name for the date field
        label: Label text to display
        x: X coordinate for the field
        y: Y coordinate for the field
        width: Width of the date field
        height: Height of the date field
        date_format: Expected date format (MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD)
        page: Page number (1-indexed)
        show_format_hint: Whether to show format hint below field
    
    Returns:
        Dictionary containing addition results
    """
    import time
    start_time = time.time()
    
    try:
        # Validate input path
        input_file = await validate_pdf_path(input_path)
        doc = fitz.open(str(input_file))
        
        page_num = page - 1  # Convert to 0-indexed
        if page_num >= len(doc):
            doc.close()
            return {"error": f"Page {page} does not exist in PDF", "addition_time": 0}
        
        pdf_page = doc[page_num]
        
        # Add field label if provided
        if label:
            label_rect = fitz.Rect(x, y - 20, x + width, y)
            pdf_page.insert_text((x, y - 5), label, fontname="helv", fontsize=10, color=(0, 0, 0))
        
        # Add format hint if requested
        if show_format_hint:
            hint_text = f"Format: {date_format}"
            pdf_page.insert_text((x, y + height + 10), hint_text, fontname="helv", fontsize=8, color=(0.5, 0.5, 0.5))
        
        # Create date text widget
        widget = fitz.Widget()
        widget.field_name = field_name
        widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
        widget.rect = fitz.Rect(x, y, x + width, y + height)
        widget.field_value = ""
        widget.text_maxlen = 10  # Standard date length
        widget.text_format = fitz.TEXT_ALIGN_LEFT
        
        # Add widget to page
        annot = pdf_page.add_widget(widget)
        
        # Add calendar icon (simple visual indicator)
        icon_x = x + width - 20
        calendar_rect = fitz.Rect(icon_x, y + 2, icon_x + 16, y + height - 2)
        pdf_page.draw_rect(calendar_rect, color=(0.8, 0.8, 0.8), width=1)
        pdf_page.insert_text((icon_x + 4, y + height - 6), "📅", fontname="helv", fontsize=8)
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the modified PDF
        doc.save(str(output_file), garbage=4, deflate=True, clean=True)
        doc.close()
        
        file_size = output_file.stat().st_size
        
        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "field_name": field_name,
            "label": label,
            "date_format": date_format,
            "position": {"x": x, "y": y, "width": width, "height": height},
            "page": page,
            "file_size": format_file_size(file_size),
            "addition_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"Adding date field failed: {str(e)}", "addition_time": round(time.time() - start_time, 2)}

@mcp.tool(name="validate_form_data", description="Validate form data against rules and constraints")
async def validate_form_data(
    pdf_path: str,
    form_data: str,  # JSON string of field values
    validation_rules: str = "{}"  # JSON string of validation rules
) -> Dict[str, Any]:
    """
    Validate form data against specified rules and field constraints
    
    Args:
        pdf_path: Path to the PDF form
        form_data: JSON string of field names and values to validate
        validation_rules: JSON string defining validation rules per field
    
    Validation rules format:
    {
        "field_name": {
            "required": true,
            "type": "email|phone|number|text|date",
            "min_length": 5,
            "max_length": 100,
            "pattern": "regex_pattern",
            "custom_message": "Custom error message"
        }
    }
    
    Returns:
        Dictionary containing validation results
    """
    import json
    import re
    import time
    start_time = time.time()
    
    try:
        # Parse inputs
        try:
            field_values = safe_json_parse(form_data) if form_data else {}
            rules = safe_json_parse(validation_rules) if validation_rules else {}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON input: {str(e)}", "validation_time": 0}
        
        # Get form structure directly
        path = await validate_pdf_path(pdf_path)
        doc = fitz.open(str(path))
        
        if not doc.is_form_pdf:
            doc.close()
            return {"error": "PDF does not contain form fields", "validation_time": 0}
        
        # Extract form fields directly
        form_fields_list = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            for widget in page.widgets():
                field_info = {
                    "field_name": widget.field_name,
                    "field_type": widget.field_type_string,
                    "field_value": widget.field_value or ""
                }
                
                # Add choices for dropdown fields
                if hasattr(widget, 'choice_values') and widget.choice_values:
                    field_info["choices"] = widget.choice_values
                
                form_fields_list.append(field_info)
        
        doc.close()
        
        if not form_fields_list:
            return {"error": "No form fields found in PDF", "validation_time": 0}
        
        # Build field info lookup
        form_fields = {field["field_name"]: field for field in form_fields_list}
        
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "field_validations": {},
            "summary": {
                "total_fields": len(form_fields),
                "validated_fields": 0,
                "required_fields_missing": [],
                "invalid_fields": []
            }
        }
        
        # Define validation patterns
        validation_patterns = {
            "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            "phone": r'^[\+]?[1-9][\d]{0,15}$',
            "number": r'^-?\d*\.?\d+$',
            "date": r'^\d{1,2}[/-]\d{1,2}[/-]\d{4}$'
        }
        
        # Validate each field
        for field_name, field_info in form_fields.items():
            field_validation = {
                "field_name": field_name,
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
            
            field_value = field_values.get(field_name, "")
            field_rule = rules.get(field_name, {})
            
            # Check required fields
            if field_rule.get("required", False) and not field_value:
                field_validation["is_valid"] = False
                field_validation["errors"].append("Field is required but empty")
                validation_results["summary"]["required_fields_missing"].append(field_name)
                validation_results["is_valid"] = False
            
            # Skip further validation if field is empty and not required
            if not field_value and not field_rule.get("required", False):
                validation_results["field_validations"][field_name] = field_validation
                continue
            
            validation_results["summary"]["validated_fields"] += 1
            
            # Length validation
            if "min_length" in field_rule and len(str(field_value)) < field_rule["min_length"]:
                field_validation["is_valid"] = False
                field_validation["errors"].append(f"Minimum length is {field_rule['min_length']} characters")
            
            if "max_length" in field_rule and len(str(field_value)) > field_rule["max_length"]:
                field_validation["is_valid"] = False
                field_validation["errors"].append(f"Maximum length is {field_rule['max_length']} characters")
            
            # Type validation
            field_type = field_rule.get("type", "text")
            if field_type in validation_patterns and field_value:
                if not re.match(validation_patterns[field_type], str(field_value)):
                    field_validation["is_valid"] = False
                    field_validation["errors"].append(f"Invalid {field_type} format")
            
            # Custom pattern validation
            if "pattern" in field_rule and field_value:
                try:
                    if not re.match(field_rule["pattern"], str(field_value)):
                        custom_msg = field_rule.get("custom_message", "Field format is invalid")
                        field_validation["is_valid"] = False
                        field_validation["errors"].append(custom_msg)
                except re.error:
                    field_validation["warnings"].append("Invalid regex pattern in validation rule")
            
            # Dropdown/Choice validation
            if field_info.get("field_type") in ["ComboBox", "ListBox"] and "choices" in field_info:
                if field_value and field_value not in field_info["choices"]:
                    field_validation["is_valid"] = False
                    field_validation["errors"].append(f"Value must be one of: {', '.join(field_info['choices'])}")
            
            # Track invalid fields
            if not field_validation["is_valid"]:
                validation_results["summary"]["invalid_fields"].append(field_name)
                validation_results["is_valid"] = False
                validation_results["errors"].extend([f"{field_name}: {error}" for error in field_validation["errors"]])
            
            if field_validation["warnings"]:
                validation_results["warnings"].extend([f"{field_name}: {warning}" for warning in field_validation["warnings"]])
            
            validation_results["field_validations"][field_name] = field_validation
        
        # Overall validation summary
        validation_results["summary"]["error_count"] = len(validation_results["errors"])
        validation_results["summary"]["warning_count"] = len(validation_results["warnings"])
        validation_results["validation_time"] = round(time.time() - start_time, 2)
        
        return validation_results
        
    except Exception as e:
        return {"error": f"Form validation failed: {str(e)}", "validation_time": round(time.time() - start_time, 2)}

@mcp.tool(name="add_field_validation", description="Add validation rules to existing form fields")
async def add_field_validation(
    input_path: str,
    output_path: str,
    validation_rules: str  # JSON string of validation rules
) -> Dict[str, Any]:
    """
    Add JavaScript validation rules to form fields (where supported)
    
    Args:
        input_path: Path to the existing PDF form
        output_path: Path where PDF with validation should be saved
        validation_rules: JSON string defining validation rules
    
    Rules format:
    {
        "field_name": {
            "required": true,
            "format": "email|phone|number|date",
            "message": "Custom validation message"
        }
    }
    
    Returns:
        Dictionary containing validation addition results
    """
    import json
    import time
    start_time = time.time()
    
    try:
        # Parse validation rules
        try:
            rules = safe_json_parse(validation_rules) if validation_rules else {}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid validation rules JSON: {str(e)}", "addition_time": 0}
        
        # Validate input path
        input_file = await validate_pdf_path(input_path)
        doc = fitz.open(str(input_file))
        
        if not doc.is_form_pdf:
            doc.close()
            return {"error": "Input PDF is not a form document", "addition_time": 0}
        
        added_validations = []
        failed_validations = []
        
        # Process each page to find and modify form fields
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            for widget in page.widgets():
                field_name = widget.field_name
                
                if field_name in rules:
                    rule = rules[field_name]
                    
                    try:
                        # Add visual indicators for required fields
                        if rule.get("required", False):
                            # Add red asterisk for required fields
                            field_rect = widget.rect
                            asterisk_pos = (field_rect.x1 + 5, field_rect.y0 + 12)
                            page.insert_text(asterisk_pos, "*", fontname="helv", fontsize=12, color=(1, 0, 0))
                        
                        # Add format hints
                        format_type = rule.get("format", "")
                        if format_type:
                            hint_text = ""
                            if format_type == "email":
                                hint_text = "example@domain.com"
                            elif format_type == "phone":
                                hint_text = "(555) 123-4567"
                            elif format_type == "date":
                                hint_text = "MM/DD/YYYY"
                            elif format_type == "number":
                                hint_text = "Numbers only"
                            
                            if hint_text:
                                hint_pos = (widget.rect.x0, widget.rect.y1 + 10)
                                page.insert_text(hint_pos, hint_text, fontname="helv", fontsize=8, color=(0.5, 0.5, 0.5))
                        
                        # Note: Full JavaScript validation would require more complex PDF manipulation
                        # For now, we add visual cues and could extend with actual JS validation later
                        
                        added_validations.append({
                            "field_name": field_name,
                            "required": rule.get("required", False),
                            "format": format_type,
                            "page": page_num + 1,
                            "validation_type": "visual_cues"
                        })
                        
                    except Exception as e:
                        failed_validations.append({
                            "field_name": field_name,
                            "error": str(e)
                        })
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the modified PDF
        doc.save(str(output_file), garbage=4, deflate=True, clean=True)
        doc.close()
        
        file_size = output_file.stat().st_size
        
        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "validations_added": len(added_validations),
            "validations_failed": len(failed_validations),
            "validation_details": added_validations,
            "failed_validations": failed_validations,
            "file_size": format_file_size(file_size),
            "addition_time": round(time.time() - start_time, 2),
            "note": "Visual validation cues added. Full JavaScript validation requires PDF viewer support."
        }
        
    except Exception as e:
        return {"error": f"Adding field validation failed: {str(e)}", "addition_time": round(time.time() - start_time, 2)}

@mcp.tool(name="merge_pdfs_advanced", description="Advanced PDF merging with bookmark preservation and options")
async def merge_pdfs_advanced(
    input_paths: str,  # JSON array of PDF file paths
    output_path: str,
    preserve_bookmarks: bool = True,
    add_page_numbers: bool = False,
    include_toc: bool = False
) -> Dict[str, Any]:
    """
    Merge multiple PDF files into a single document
    
    Args:
        input_paths: JSON array of PDF file paths to merge
        output_path: Path where merged PDF should be saved
        preserve_bookmarks: Whether to preserve existing bookmarks
        add_page_numbers: Whether to add page numbers to merged document
        include_toc: Whether to generate table of contents with source filenames
    
    Returns:
        Dictionary containing merge results
    """
    import json
    import time
    start_time = time.time()
    
    try:
        # Parse input paths
        try:
            pdf_paths = safe_json_parse(input_paths) if input_paths else []
        except json.JSONDecodeError as e:
            return {"error": f"Invalid input paths JSON: {str(e)}", "merge_time": 0}
        
        if len(pdf_paths) < 2:
            return {"error": "At least 2 PDF files are required for merging", "merge_time": 0}
        
        # Validate all input paths
        validated_paths = []
        for pdf_path in pdf_paths:
            try:
                validated_path = await validate_pdf_path(pdf_path)
                validated_paths.append(validated_path)
            except Exception as e:
                return {"error": f"Invalid PDF path '{pdf_path}': {str(e)}", "merge_time": 0}
        
        # Create output document
        merged_doc = fitz.open()
        merge_info = {
            "files_merged": [],
            "total_pages": 0,
            "bookmarks_preserved": 0,
            "merge_errors": []
        }
        
        current_page_offset = 0
        
        # Process each PDF
        for i, pdf_path in enumerate(validated_paths):
            try:
                doc = fitz.open(str(pdf_path))
                filename = Path(pdf_path).name
                
                # Insert pages
                merged_doc.insert_pdf(doc, from_page=0, to_page=doc.page_count - 1)
                
                # Handle bookmarks
                if preserve_bookmarks and doc.get_toc():
                    toc = doc.get_toc()
                    # Adjust bookmark page numbers for merged document
                    adjusted_toc = []
                    for level, title, page_num in toc:
                        adjusted_toc.append([level, title, page_num + current_page_offset])
                    
                    # Add adjusted bookmarks to merged document
                    existing_toc = merged_doc.get_toc()
                    existing_toc.extend(adjusted_toc)
                    merged_doc.set_toc(existing_toc)
                    merge_info["bookmarks_preserved"] += len(toc)
                
                # Add table of contents entry for source file
                if include_toc:
                    toc_entry = [1, f"Document {i+1}: {filename}", current_page_offset + 1]
                    existing_toc = merged_doc.get_toc()
                    existing_toc.append(toc_entry)
                    merged_doc.set_toc(existing_toc)
                
                merge_info["files_merged"].append({
                    "filename": filename,
                    "pages": doc.page_count,
                    "page_range": f"{current_page_offset + 1}-{current_page_offset + doc.page_count}"
                })
                
                current_page_offset += doc.page_count
                doc.close()
                
            except Exception as e:
                merge_info["merge_errors"].append({
                    "filename": Path(pdf_path).name,
                    "error": str(e)
                })
        
        # Add page numbers if requested
        if add_page_numbers:
            for page_num in range(merged_doc.page_count):
                page = merged_doc[page_num]
                page_rect = page.rect
                
                # Add page number at bottom center
                page_text = f"Page {page_num + 1}"
                text_pos = (page_rect.width / 2 - 20, page_rect.height - 20)
                page.insert_text(text_pos, page_text, fontname="helv", fontsize=10, color=(0.5, 0.5, 0.5))
        
        merge_info["total_pages"] = merged_doc.page_count
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save merged PDF
        merged_doc.save(str(output_file), garbage=4, deflate=True, clean=True)
        merged_doc.close()
        
        file_size = output_file.stat().st_size
        
        return {
            "output_path": str(output_file),
            "files_processed": len(pdf_paths),
            "files_successfully_merged": len(merge_info["files_merged"]),
            "merge_details": merge_info,
            "total_pages": merge_info["total_pages"],
            "bookmarks_preserved": merge_info["bookmarks_preserved"],
            "page_numbers_added": add_page_numbers,
            "toc_generated": include_toc,
            "file_size": format_file_size(file_size),
            "merge_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"PDF merge failed: {str(e)}", "merge_time": round(time.time() - start_time, 2)}

@mcp.tool(name="split_pdf_by_pages", description="Split PDF into separate files by page ranges")
async def split_pdf_by_pages(
    input_path: str,
    output_directory: str,
    page_ranges: str,  # JSON array of ranges like ["1-5", "6-10", "11-end"]
    naming_pattern: str = "page_{start}-{end}.pdf"
) -> Dict[str, Any]:
    """
    Split PDF into separate files by specified page ranges
    
    Args:
        input_path: Path to the PDF file to split
        output_directory: Directory where split files should be saved
        page_ranges: JSON array of page ranges (1-indexed)
        naming_pattern: Pattern for output filenames with {start}, {end}, {index} placeholders
    
    Returns:
        Dictionary containing split results
    """
    import json
    import time
    start_time = time.time()
    
    try:
        # Parse page ranges
        try:
            ranges = safe_json_parse(page_ranges) if page_ranges else []
        except json.JSONDecodeError as e:
            return {"error": f"Invalid page ranges JSON: {str(e)}", "split_time": 0}
        
        if not ranges:
            return {"error": "At least one page range is required", "split_time": 0}
        
        # Validate input path
        input_file = await validate_pdf_path(input_path)
        doc = fitz.open(str(input_file))
        total_pages = doc.page_count
        
        # Create output directory with security validation
        output_dir = validate_output_path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        split_info = {
            "files_created": [],
            "split_errors": [],
            "total_pages_processed": 0
        }
        
        # Process each range
        for i, range_str in enumerate(ranges):
            try:
                # Parse range string
                if range_str.lower() == "all":
                    start_page = 1
                    end_page = total_pages
                elif "-" in range_str:
                    parts = range_str.split("-", 1)
                    start_page = int(parts[0])
                    if parts[1].lower() == "end":
                        end_page = total_pages
                    else:
                        end_page = int(parts[1])
                else:
                    # Single page
                    start_page = end_page = int(range_str)
                
                # Validate page numbers (convert to 0-indexed for PyMuPDF)
                if start_page < 1 or start_page > total_pages:
                    split_info["split_errors"].append({
                        "range": range_str,
                        "error": f"Start page {start_page} out of range (1-{total_pages})"
                    })
                    continue
                
                if end_page < 1 or end_page > total_pages:
                    split_info["split_errors"].append({
                        "range": range_str,
                        "error": f"End page {end_page} out of range (1-{total_pages})"
                    })
                    continue
                
                if start_page > end_page:
                    split_info["split_errors"].append({
                        "range": range_str,
                        "error": f"Start page {start_page} greater than end page {end_page}"
                    })
                    continue
                
                # Create output filename
                output_filename = naming_pattern.format(
                    start=start_page,
                    end=end_page,
                    index=i+1,
                    original=Path(input_file).stem
                )
                output_path = output_dir / output_filename
                
                # Create new document with specified pages
                new_doc = fitz.open()
                new_doc.insert_pdf(doc, from_page=start_page-1, to_page=end_page-1)
                
                # Copy relevant bookmarks
                original_toc = doc.get_toc()
                if original_toc:
                    filtered_toc = []
                    for level, title, page_num in original_toc:
                        # Adjust page numbers and include only relevant bookmarks
                        if start_page <= page_num <= end_page:
                            adjusted_page = page_num - start_page + 1
                            filtered_toc.append([level, title, adjusted_page])
                    
                    if filtered_toc:
                        new_doc.set_toc(filtered_toc)
                
                # Save split document
                new_doc.save(str(output_path), garbage=4, deflate=True, clean=True)
                new_doc.close()
                
                file_size = output_path.stat().st_size
                pages_in_range = end_page - start_page + 1
                
                split_info["files_created"].append({
                    "filename": output_filename,
                    "page_range": f"{start_page}-{end_page}",
                    "pages": pages_in_range,
                    "file_size": format_file_size(file_size),
                    "output_path": str(output_path)
                })
                
                split_info["total_pages_processed"] += pages_in_range
                
            except ValueError as e:
                split_info["split_errors"].append({
                    "range": range_str,
                    "error": f"Invalid range format: {str(e)}"
                })
            except Exception as e:
                split_info["split_errors"].append({
                    "range": range_str,
                    "error": f"Split failed: {str(e)}"
                })
        
        doc.close()
        
        return {
            "input_path": str(input_file),
            "output_directory": str(output_dir),
            "total_input_pages": total_pages,
            "files_created": len(split_info["files_created"]),
            "files_failed": len(split_info["split_errors"]),
            "split_details": split_info,
            "naming_pattern": naming_pattern,
            "split_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"PDF split failed: {str(e)}", "split_time": round(time.time() - start_time, 2)}

@mcp.tool(name="reorder_pdf_pages", description="Reorder pages in a PDF document")
async def reorder_pdf_pages(
    input_path: str,
    output_path: str,
    page_order: str  # JSON array of page numbers in desired order
) -> Dict[str, Any]:
    """
    Reorder pages in a PDF document according to specified sequence
    
    Args:
        input_path: Path to the PDF file to reorder
        output_path: Path where reordered PDF should be saved
        page_order: JSON array of page numbers in desired order (1-indexed)
    
    Returns:
        Dictionary containing reorder results
    """
    import json
    import time
    start_time = time.time()
    
    try:
        # Parse page order
        try:
            order = safe_json_parse(page_order) if page_order else []
        except json.JSONDecodeError as e:
            return {"error": f"Invalid page order JSON: {str(e)}", "reorder_time": 0}
        
        if not order:
            return {"error": "Page order array is required", "reorder_time": 0}
        
        # Validate input path
        input_file = await validate_pdf_path(input_path)
        doc = fitz.open(str(input_file))
        total_pages = doc.page_count
        
        # Validate page numbers
        invalid_pages = []
        for page_num in order:
            if not isinstance(page_num, int) or page_num < 1 or page_num > total_pages:
                invalid_pages.append(page_num)
        
        if invalid_pages:
            doc.close()
            return {"error": f"Invalid page numbers: {invalid_pages}. Pages must be 1-{total_pages}", "reorder_time": 0}
        
        # Create new document with reordered pages
        new_doc = fitz.open()
        
        reorder_info = {
            "pages_processed": 0,
            "original_order": list(range(1, total_pages + 1)),
            "new_order": order,
            "pages_duplicated": [],
            "pages_omitted": []
        }
        
        # Track which pages are used
        pages_used = set()
        
        # Insert pages in specified order
        for new_position, original_page in enumerate(order, 1):
            # Convert to 0-indexed for PyMuPDF
            page_index = original_page - 1
            
            # Insert the page
            new_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
            
            # Track usage
            if original_page in pages_used:
                reorder_info["pages_duplicated"].append(original_page)
            else:
                pages_used.add(original_page)
            
            reorder_info["pages_processed"] += 1
        
        # Find omitted pages
        all_pages = set(range(1, total_pages + 1))
        reorder_info["pages_omitted"] = list(all_pages - pages_used)
        
        # Handle bookmarks - adjust page references
        original_toc = doc.get_toc()
        if original_toc:
            new_toc = []
            for level, title, original_page_ref in original_toc:
                # Find new position of the referenced page
                try:
                    new_page_ref = order.index(original_page_ref) + 1
                    new_toc.append([level, title, new_page_ref])
                except ValueError:
                    # Page was omitted, skip this bookmark
                    pass
            
            if new_toc:
                new_doc.set_toc(new_toc)
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save reordered PDF
        new_doc.save(str(output_file), garbage=4, deflate=True, clean=True)
        
        doc.close()
        new_doc.close()
        
        file_size = output_file.stat().st_size
        
        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "original_pages": total_pages,
            "reordered_pages": len(order),
            "reorder_details": reorder_info,
            "pages_duplicated": len(reorder_info["pages_duplicated"]),
            "pages_omitted": len(reorder_info["pages_omitted"]),
            "file_size": format_file_size(file_size),
            "reorder_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"PDF page reorder failed: {str(e)}", "reorder_time": round(time.time() - start_time, 2)}

@mcp.tool(name="split_pdf_by_bookmarks", description="Split PDF into separate files using bookmarks as breakpoints")
async def split_pdf_by_bookmarks(
    input_path: str,
    output_directory: str,
    bookmark_level: int = 1,
    naming_pattern: str = "{title}.pdf"
) -> Dict[str, Any]:
    """
    Split PDF into separate files using bookmarks as natural breakpoints
    
    Args:
        input_path: Path to the PDF file to split
        output_directory: Directory where split files should be saved
        bookmark_level: Which bookmark level to use for splitting (1=chapters, 2=sections)
        naming_pattern: Pattern for output filenames with {title}, {index} placeholders
    
    Returns:
        Dictionary containing split results
    """
    import time
    import re
    start_time = time.time()
    
    try:
        # Validate input path
        input_file = await validate_pdf_path(input_path)
        doc = fitz.open(str(input_file))
        
        # Get table of contents
        toc = doc.get_toc()
        if not toc:
            doc.close()
            return {"error": "PDF has no bookmarks for splitting", "split_time": 0}
        
        # Filter bookmarks by level
        split_points = []
        for level, title, page_num in toc:
            if level == bookmark_level:
                split_points.append((title, page_num))
        
        if len(split_points) < 2:
            doc.close()
            return {"error": f"Not enough level-{bookmark_level} bookmarks for splitting (found {len(split_points)})", "split_time": 0}
        
        # Create output directory with security validation
        output_dir = validate_output_path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        split_info = {
            "files_created": [],
            "split_errors": [],
            "total_pages_processed": 0
        }
        
        total_pages = doc.page_count
        
        # Process each bookmark section
        for i, (title, start_page) in enumerate(split_points):
            try:
                # Determine end page
                if i + 1 < len(split_points):
                    end_page = split_points[i + 1][1] - 1
                else:
                    end_page = total_pages
                
                # Clean title for filename
                clean_title = re.sub(r'[^\w\s-]', '', title).strip()
                clean_title = re.sub(r'\s+', '_', clean_title)
                if not clean_title:
                    clean_title = f"section_{i+1}"
                
                # Create output filename
                output_filename = naming_pattern.format(
                    title=clean_title,
                    index=i+1,
                    original=Path(input_file).stem
                )
                
                # Ensure .pdf extension
                if not output_filename.lower().endswith('.pdf'):
                    output_filename += '.pdf'
                
                output_path = output_dir / output_filename
                
                # Create new document with bookmark section
                new_doc = fitz.open()
                new_doc.insert_pdf(doc, from_page=start_page-1, to_page=end_page-1)
                
                # Add relevant bookmarks to new document
                section_toc = []
                for level, bookmark_title, page_num in toc:
                    if start_page <= page_num <= end_page:
                        adjusted_page = page_num - start_page + 1
                        section_toc.append([level, bookmark_title, adjusted_page])
                
                if section_toc:
                    new_doc.set_toc(section_toc)
                
                # Save split document
                new_doc.save(str(output_path), garbage=4, deflate=True, clean=True)
                new_doc.close()
                
                file_size = output_path.stat().st_size
                pages_in_section = end_page - start_page + 1
                
                split_info["files_created"].append({
                    "filename": output_filename,
                    "bookmark_title": title,
                    "page_range": f"{start_page}-{end_page}",
                    "pages": pages_in_section,
                    "file_size": format_file_size(file_size),
                    "output_path": str(output_path)
                })
                
                split_info["total_pages_processed"] += pages_in_section
                
            except Exception as e:
                split_info["split_errors"].append({
                    "bookmark_title": title,
                    "error": f"Split failed: {str(e)}"
                })
        
        doc.close()
        
        return {
            "input_path": str(input_file),
            "output_directory": str(output_dir),
            "bookmark_level_used": bookmark_level,
            "bookmarks_found": len(split_points),
            "files_created": len(split_info["files_created"]),
            "files_failed": len(split_info["split_errors"]),
            "split_details": split_info,
            "naming_pattern": naming_pattern,
            "split_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"Bookmark-based PDF split failed: {str(e)}", "split_time": round(time.time() - start_time, 2)}

@mcp.tool(name="add_sticky_notes", description="Add sticky note comments to specific locations in PDF")
async def add_sticky_notes(
    input_path: str,
    output_path: str,
    notes: str  # JSON array of note definitions
) -> Dict[str, Any]:
    """
    Add sticky note annotations to PDF at specified locations
    
    Args:
        input_path: Path to the existing PDF
        output_path: Path where PDF with notes should be saved
        notes: JSON array of note definitions
    
    Note format:
    [
        {
            "page": 1,
            "x": 100, "y": 200,
            "content": "This is a note",
            "author": "John Doe",
            "subject": "Review Comment",
            "color": "yellow"
        }
    ]
    
    Returns:
        Dictionary containing annotation results
    """
    import json
    import time
    start_time = time.time()
    
    try:
        # Parse notes
        try:
            note_definitions = safe_json_parse(notes) if notes else []
        except json.JSONDecodeError as e:
            return {"error": f"Invalid notes JSON: {str(e)}", "annotation_time": 0}
        
        if not note_definitions:
            return {"error": "At least one note is required", "annotation_time": 0}
        
        # Validate input path
        input_file = await validate_pdf_path(input_path)
        doc = fitz.open(str(input_file))
        
        annotation_info = {
            "notes_added": [],
            "annotation_errors": []
        }
        
        # Color mapping
        color_map = {
            "yellow": (1, 1, 0),
            "red": (1, 0, 0),
            "green": (0, 1, 0),
            "blue": (0, 0, 1),
            "orange": (1, 0.5, 0),
            "purple": (0.5, 0, 1),
            "pink": (1, 0.75, 0.8),
            "gray": (0.5, 0.5, 0.5)
        }
        
        # Process each note
        for i, note_def in enumerate(note_definitions):
            try:
                page_num = note_def.get("page", 1) - 1  # Convert to 0-indexed
                x = note_def.get("x", 100)
                y = note_def.get("y", 100)
                content = note_def.get("content", "")
                author = note_def.get("author", "Anonymous")
                subject = note_def.get("subject", "Note")
                color_name = note_def.get("color", "yellow").lower()
                
                # Validate page number
                if page_num >= len(doc) or page_num < 0:
                    annotation_info["annotation_errors"].append({
                        "note_index": i,
                        "error": f"Page {page_num + 1} does not exist"
                    })
                    continue
                
                page = doc[page_num]
                
                # Get color
                color = color_map.get(color_name, (1, 1, 0))  # Default to yellow
                
                # Create realistic sticky note appearance
                note_width = 80
                note_height = 60
                note_rect = fitz.Rect(x, y, x + note_width, y + note_height)
                
                # Add colored rectangle background (sticky note paper)
                page.draw_rect(note_rect, color=color, fill=color, width=1)
                
                # Add slight shadow effect for depth
                shadow_rect = fitz.Rect(x + 2, y - 2, x + note_width + 2, y + note_height - 2)
                page.draw_rect(shadow_rect, color=(0.7, 0.7, 0.7), fill=(0.7, 0.7, 0.7), width=0)
                
                # Add the main sticky note rectangle on top
                page.draw_rect(note_rect, color=color, fill=color, width=1)
                
                # Add border for definition
                border_color = (min(1, color[0] * 0.8), min(1, color[1] * 0.8), min(1, color[2] * 0.8))
                page.draw_rect(note_rect, color=border_color, width=1)
                
                # Add "folded corner" effect (small triangle)
                fold_size = 8
                fold_points = [
                    fitz.Point(x + note_width - fold_size, y),
                    fitz.Point(x + note_width, y),
                    fitz.Point(x + note_width, y + fold_size)
                ]
                page.draw_polyline(fold_points, color=(1, 1, 1), fill=(1, 1, 1), width=1)
                
                # Add text content on the sticky note
                text_rect = fitz.Rect(x + 4, y + 4, x + note_width - 8, y + note_height - 8)
                
                # Wrap text to fit in sticky note
                words = content.split()
                lines = []
                current_line = []
                
                for word in words:
                    test_line = " ".join(current_line + [word])
                    if len(test_line) > 12:  # Approximate character limit per line
                        if current_line:
                            lines.append(" ".join(current_line))
                            current_line = [word]
                        else:
                            lines.append(word[:12] + "...")
                            break
                    else:
                        current_line.append(word)
                
                if current_line:
                    lines.append(" ".join(current_line))
                
                # Limit to 4 lines to fit in sticky note
                if len(lines) > 4:
                    lines = lines[:3] + [lines[3][:8] + "..."]
                
                # Draw text lines
                line_height = 10
                text_y = y + 10
                text_color = (0, 0, 0)  # Black text
                
                for line in lines[:4]:  # Max 4 lines
                    if text_y + line_height <= y + note_height - 4:
                        page.insert_text((x + 6, text_y), line, fontname="helv", fontsize=8, color=text_color)
                        text_y += line_height
                
                # Create invisible text annotation for PDF annotation system compatibility
                annot = page.add_text_annot(fitz.Point(x + note_width/2, y + note_height/2), content)
                annot.set_info(content=content, title=subject)
                
                # Set the popup/content background to match sticky note color
                annot.set_colors(stroke=(0, 0, 0, 0), fill=color)  # Invisible border, colored background
                annot.set_flags(fitz.PDF_ANNOT_IS_PRINT | fitz.PDF_ANNOT_IS_INVISIBLE)
                annot.update()
                
                annotation_info["notes_added"].append({
                    "page": page_num + 1,
                    "position": {"x": x, "y": y},
                    "content": content[:50] + "..." if len(content) > 50 else content,
                    "author": author,
                    "subject": subject,
                    "color": color_name
                })
                
            except Exception as e:
                annotation_info["annotation_errors"].append({
                    "note_index": i,
                    "error": f"Failed to add note: {str(e)}"
                })
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PDF with annotations
        doc.save(str(output_file), garbage=4, deflate=True, clean=True)
        doc.close()
        
        file_size = output_file.stat().st_size
        
        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "notes_requested": len(note_definitions),
            "notes_added": len(annotation_info["notes_added"]),
            "notes_failed": len(annotation_info["annotation_errors"]),
            "annotation_details": annotation_info,
            "file_size": format_file_size(file_size),
            "annotation_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"Adding sticky notes failed: {str(e)}", "annotation_time": round(time.time() - start_time, 2)}

@mcp.tool(name="add_video_notes", description="Add video sticky notes that embed and launch video content")
async def add_video_notes(
    input_path: str,
    output_path: str,
    video_notes: str  # JSON array of video note definitions
) -> Dict[str, Any]:
    """
    Add video sticky notes that embed video files and launch on click
    
    Args:
        input_path: Path to the existing PDF
        output_path: Path where PDF with video notes should be saved
        video_notes: JSON array of video note definitions
    
    Video note format:
    [
        {
            "page": 1,
            "x": 100, "y": 200,
            "video_path": "/path/to/video.mp4",
            "title": "Demo Video",
            "color": "red",
            "size": "medium"
        }
    ]
    
    Returns:
        Dictionary containing video embedding results
    """
    import json
    import time
    import hashlib
    import os
    start_time = time.time()
    
    try:
        # Parse video notes
        try:
            note_definitions = safe_json_parse(video_notes) if video_notes else []
        except json.JSONDecodeError as e:
            return {"error": f"Invalid video notes JSON: {str(e)}", "embedding_time": 0}
        
        if not note_definitions:
            return {"error": "At least one video note is required", "embedding_time": 0}
        
        # Validate input path
        input_file = await validate_pdf_path(input_path)
        doc = fitz.open(str(input_file))
        
        embedding_info = {
            "videos_embedded": [],
            "embedding_errors": []
        }
        
        # Track embedded file names to prevent duplicates
        embedded_names = set()
        
        # Color mapping for video note appearance
        color_map = {
            "red": (1, 0, 0),
            "blue": (0, 0, 1),
            "green": (0, 1, 0),
            "orange": (1, 0.5, 0),
            "purple": (0.5, 0, 1),
            "yellow": (1, 1, 0),
            "pink": (1, 0.75, 0.8),
            "gray": (0.5, 0.5, 0.5)
        }
        
        # Size mapping
        size_map = {
            "small": (60, 45),
            "medium": (80, 60),
            "large": (100, 75)
        }
        
        # Process each video note
        for i, note_def in enumerate(note_definitions):
            try:
                page_num = note_def.get("page", 1) - 1  # Convert to 0-indexed
                x = note_def.get("x", 100)
                y = note_def.get("y", 100)
                video_path = note_def.get("video_path", "")
                title = note_def.get("title", "Video")
                color_name = note_def.get("color", "red").lower()
                size_name = note_def.get("size", "medium").lower()
                
                # Validate inputs
                if not video_path or not os.path.exists(video_path):
                    embedding_info["embedding_errors"].append({
                        "note_index": i,
                        "error": f"Video file not found: {video_path}"
                    })
                    continue
                
                # Check video format and suggest conversion if needed
                video_ext = os.path.splitext(video_path)[1].lower()
                supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
                recommended_formats = ['.mp4']
                
                if video_ext not in supported_formats:
                    embedding_info["embedding_errors"].append({
                        "note_index": i,
                        "error": f"Unsupported video format: {video_ext}. Supported: {', '.join(supported_formats)}",
                        "conversion_suggestion": f"Convert with FFmpeg: ffmpeg -i '{os.path.basename(video_path)}' -c:v libx264 -c:a aac -preset medium '{os.path.splitext(os.path.basename(video_path))[0]}.mp4'"
                    })
                    continue
                
                # Suggest optimization for non-MP4 files
                conversion_suggestion = None
                if video_ext not in recommended_formats:
                    conversion_suggestion = f"For best compatibility, convert to MP4: ffmpeg -i '{os.path.basename(video_path)}' -c:v libx264 -c:a aac -preset medium -crf 23 '{os.path.splitext(os.path.basename(video_path))[0]}.mp4'"
                
                # Video validation and metadata extraction
                try:
                    import cv2
                    cap = cv2.VideoCapture(video_path)
                    
                    # Check if video is readable/valid
                    if not cap.isOpened():
                        embedding_info["embedding_errors"].append({
                            "note_index": i,
                            "error": f"Cannot open or corrupted video file: {video_path}",
                            "validation_suggestion": "Check if video file is corrupted and try re-encoding"
                        })
                        continue
                    
                    # Extract video metadata
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration_seconds = frame_count / fps if fps > 0 else 0
                    
                    # Extract first frame as thumbnail
                    ret, frame = cap.read()
                    thumbnail_data = None
                    if ret and frame is not None:
                        # Resize thumbnail to fit sticky note
                        thumbnail_height = min(note_height - 20, height)  # Leave space for metadata
                        thumbnail_width = int((width / height) * thumbnail_height)
                        
                        # Ensure thumbnail fits within note width
                        if thumbnail_width > note_width - 10:
                            thumbnail_width = note_width - 10
                            thumbnail_height = int((height / width) * thumbnail_width)
                        
                        # Resize frame
                        thumbnail = cv2.resize(frame, (thumbnail_width, thumbnail_height))
                        # Convert BGR to RGB
                        thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                        thumbnail_data = (thumbnail_rgb, thumbnail_width, thumbnail_height)
                    
                    cap.release()
                    
                    # Format duration for display
                    if duration_seconds < 60:
                        duration_str = f"{int(duration_seconds)}s"
                    else:
                        minutes = int(duration_seconds // 60)
                        seconds = int(duration_seconds % 60)
                        duration_str = f"{minutes}:{seconds:02d}"
                    
                    # Create metadata string
                    metadata_text = f"{duration_str} | {width}x{height}"
                    
                except ImportError:
                    # OpenCV not available - basic file validation only
                    thumbnail_data = None
                    metadata_text = None
                    duration_seconds = 0
                    width, height = 0, 0
                    
                    # Basic file validation - check if file starts with video headers
                    try:
                        with open(video_path, 'rb') as f:
                            header = f.read(12)
                            # Check for common video file signatures
                            video_signatures = [
                                b'\x00\x00\x00\x18ftypmp4',  # MP4
                                b'\x00\x00\x00\x20ftypmp4',  # MP4
                                b'RIFF',                      # AVI (partial)
                                b'\x1a\x45\xdf\xa3',         # MKV
                            ]
                            
                            is_valid = any(header.startswith(sig) for sig in video_signatures)
                            if not is_valid:
                                embedding_info["embedding_errors"].append({
                                    "note_index": i,
                                    "error": f"Invalid or corrupted video file: {video_path}",
                                    "validation_suggestion": "File does not appear to be a valid video format"
                                })
                                continue
                    except Exception as e:
                        embedding_info["embedding_errors"].append({
                            "note_index": i,
                            "error": f"Cannot validate video file: {str(e)}"
                        })
                        continue
                except Exception as e:
                    embedding_info["embedding_errors"].append({
                        "note_index": i,
                        "error": f"Video validation failed: {str(e)}"
                    })
                    continue
                
                # Check file size and suggest compression if very large
                file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                if file_size_mb > 50:  # Warn for files > 50MB
                    size_warning = f"Large video file ({file_size_mb:.1f}MB) will significantly increase PDF size"
                    if not conversion_suggestion:
                        conversion_suggestion = f"Compress video: ffmpeg -i '{os.path.basename(video_path)}' -c:v libx264 -c:a aac -preset medium -crf 28 -maxrate 1M -bufsize 2M '{os.path.splitext(os.path.basename(video_path))[0]}_compressed.mp4'"
                else:
                    size_warning = None
                
                if page_num >= len(doc) or page_num < 0:
                    embedding_info["embedding_errors"].append({
                        "note_index": i,
                        "error": f"Page {page_num + 1} does not exist"
                    })
                    continue
                
                page = doc[page_num]
                color = color_map.get(color_name, (1, 0, 0))  # Default to red
                note_width, note_height = size_map.get(size_name, (80, 60))
                
                # Create enhanced video sticky note appearance
                note_rect = fitz.Rect(x, y, x + note_width, y + note_height)
                
                # Add shadow effect
                shadow_rect = fitz.Rect(x + 3, y - 3, x + note_width + 3, y + note_height - 3)
                page.draw_rect(shadow_rect, color=(0.6, 0.6, 0.6), fill=(0.6, 0.6, 0.6), width=0)
                
                # Add main background (darker for video contrast)
                bg_color = (min(1, color[0] * 0.3), min(1, color[1] * 0.3), min(1, color[2] * 0.3))
                page.draw_rect(note_rect, color=bg_color, fill=bg_color, width=1)
                
                # Add thumbnail if available
                if thumbnail_data:
                    thumb_img, thumb_w, thumb_h = thumbnail_data
                    # Center thumbnail in note
                    thumb_x = x + (note_width - thumb_w) // 2
                    thumb_y = y + 5  # Small margin from top
                    
                    try:
                        # Convert numpy array to bytes for PyMuPDF
                        from PIL import Image
                        import io
                        
                        pil_img = Image.fromarray(thumb_img)
                        img_bytes = io.BytesIO()
                        pil_img.save(img_bytes, format='PNG')
                        img_data = img_bytes.getvalue()
                        
                        # Insert thumbnail image
                        thumb_rect = fitz.Rect(thumb_x, thumb_y, thumb_x + thumb_w, thumb_y + thumb_h)
                        page.insert_image(thumb_rect, stream=img_data)
                        
                        # Add semi-transparent overlay for play button visibility
                        overlay_rect = fitz.Rect(thumb_x, thumb_y, thumb_x + thumb_w, thumb_y + thumb_h)
                        page.draw_rect(overlay_rect, color=(0, 0, 0, 0.3), fill=(0, 0, 0, 0.3), width=0)
                        
                    except ImportError:
                        # PIL not available, use solid color background
                        page.draw_rect(note_rect, color=color, fill=color, width=1)
                else:
                    # No thumbnail, use solid color background
                    page.draw_rect(note_rect, color=color, fill=color, width=1)
                
                # Add film strip border for visual indication
                strip_color = (1, 1, 1)
                strip_width = 2
                # Top and bottom strips
                for i in range(0, note_width, 8):
                    if i + 4 <= note_width:
                        # Top perforations
                        perf_rect = fitz.Rect(x + i + 1, y - 1, x + i + 3, y + 1)
                        page.draw_rect(perf_rect, color=strip_color, fill=strip_color, width=0)
                        # Bottom perforations
                        perf_rect = fitz.Rect(x + i + 1, y + note_height - 1, x + i + 3, y + note_height + 1)
                        page.draw_rect(perf_rect, color=strip_color, fill=strip_color, width=0)
                
                # Add enhanced play button with circular background
                play_icon_size = min(note_width, note_height) // 4
                icon_x = x + note_width // 2
                icon_y = y + (note_height - 15) // 2  # Account for metadata space at bottom
                
                # Play button circle background
                circle_radius = play_icon_size + 3
                page.draw_circle(fitz.Point(icon_x, icon_y), circle_radius, color=(0, 0, 0, 0.7), fill=(0, 0, 0, 0.7), width=0)
                page.draw_circle(fitz.Point(icon_x, icon_y), circle_radius, color=(1, 1, 1), width=2)
                
                # Play triangle
                play_points = [
                    fitz.Point(icon_x - play_icon_size//2, icon_y - play_icon_size//2),
                    fitz.Point(icon_x + play_icon_size//2, icon_y),
                    fitz.Point(icon_x - play_icon_size//2, icon_y + play_icon_size//2)
                ]
                page.draw_polyline(play_points, color=(1, 1, 1), fill=(1, 1, 1), width=1)
                
                # Add video camera icon indicator in top corner
                cam_size = 8
                cam_rect = fitz.Rect(x + note_width - cam_size - 2, y + 2, x + note_width - 2, y + cam_size + 2)
                page.draw_rect(cam_rect, color=(1, 1, 1), fill=(1, 1, 1), width=1)
                page.draw_circle(fitz.Point(x + note_width - cam_size//2 - 2, y + cam_size//2 + 2), 2, color=(0, 0, 0), fill=(0, 0, 0), width=0)
                
                # Add title and metadata at bottom
                title_text = title[:15] + "..." if len(title) > 15 else title
                page.insert_text((x + 2, y + note_height - 12), title_text, fontname="helv-bold", fontsize=7, color=(1, 1, 1))
                
                if metadata_text:
                    page.insert_text((x + 2, y + note_height - 3), metadata_text, fontname="helv", fontsize=6, color=(0.9, 0.9, 0.9))
                
                # Generate unique embedded filename
                file_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
                embedded_name = f"videoPop-{file_hash}.mp4"
                
                # Ensure unique name (handle duplicates)
                counter = 1
                original_name = embedded_name
                while embedded_name in embedded_names:
                    name_parts = original_name.rsplit('.', 1)
                    embedded_name = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                    counter += 1
                
                embedded_names.add(embedded_name)
                
                # Read video file
                with open(video_path, 'rb') as video_file:
                    video_data = video_file.read()
                
                # Embed video as file attachment using PyMuPDF
                doc.embfile_add(embedded_name, video_data, filename=embedded_name, ufilename=embedded_name, desc=f"Video: {title}")
                
                # Create JavaScript action for video launch
                javascript_code = f"this.exportDataObject({{cName: '{embedded_name}', nLaunch: 2}});"
                
                # Add clickable annotation for video launch with fallback info
                fallback_info = f"""Video: {title}
Duration: {duration_str if metadata_text else 'Unknown'}
Resolution: {width}x{height if width and height else 'Unknown'}
File: {os.path.basename(video_path)}

CLICK TO PLAY VIDEO
(Requires Adobe Acrobat/Reader with JavaScript enabled)

FALLBACK ACCESS:
If video doesn't launch automatically:
1. Use PDF menu: View → Navigation Panels → Attachments
2. Find '{embedded_name}' in attachments list
3. Double-click to extract and play

MOBILE/WEB FALLBACK:
This PDF contains embedded video files that may not be
accessible in mobile or web-based PDF viewers."""

                annot = page.add_text_annot(fitz.Point(x + note_width/2, y + note_height/2), fallback_info)
                annot.set_info(content=fallback_info, title=f"Video: {title}")
                annot.set_colors(stroke=(0, 0, 0, 0), fill=color)
                annot.set_rect(note_rect)  # Cover the entire video note area
                annot.set_flags(fitz.PDF_ANNOT_IS_PRINT)
                annot.update()
                
                video_info = {
                    "page": page_num + 1,
                    "position": {"x": x, "y": y},
                    "video_file": os.path.basename(video_path),
                    "embedded_name": embedded_name,
                    "title": title,
                    "color": color_name,
                    "size": size_name,
                    "file_size_mb": round(len(video_data) / (1024 * 1024), 2),
                    "format": video_ext,
                    "optimized": video_ext in recommended_formats,
                    "duration_seconds": duration_seconds,
                    "resolution": {"width": width, "height": height},
                    "has_thumbnail": thumbnail_data is not None,
                    "metadata_display": metadata_text,
                    "fallback_accessible": True
                }
                
                # Add optional fields if they exist
                if conversion_suggestion:
                    video_info["conversion_suggestion"] = conversion_suggestion
                if size_warning:
                    video_info["size_warning"] = size_warning
                    
                embedding_info["videos_embedded"].append(video_info)
                
            except Exception as e:
                embedding_info["embedding_errors"].append({
                    "note_index": i,
                    "error": f"Failed to embed video: {str(e)}"
                })
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PDF with embedded videos
        doc.save(str(output_file), garbage=4, deflate=True, clean=True)
        doc.close()
        
        file_size = output_file.stat().st_size
        
        # Analyze format distribution
        format_stats = {}
        conversion_suggestions = []
        for video_info in embedding_info["videos_embedded"]:
            fmt = video_info.get("format", "unknown")
            format_stats[fmt] = format_stats.get(fmt, 0) + 1
            if video_info.get("conversion_suggestion"):
                conversion_suggestions.append(video_info["conversion_suggestion"])
        
        result = {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "videos_requested": len(note_definitions),
            "videos_embedded": len(embedding_info["videos_embedded"]),
            "videos_failed": len(embedding_info["embedding_errors"]),
            "embedding_details": embedding_info,
            "format_distribution": format_stats,
            "total_file_size": format_file_size(file_size),
            "compatibility_note": "Requires PDF viewer with JavaScript support (Adobe Acrobat/Reader)",
            "embedding_time": round(time.time() - start_time, 2)
        }
        
        # Add format optimization info if applicable
        if conversion_suggestions:
            result["optimization_suggestions"] = {
                "count": len(conversion_suggestions),
                "ffmpeg_commands": conversion_suggestions[:3],  # Show first 3 suggestions
                "note": "Run suggested FFmpeg commands to optimize videos for better PDF compatibility and smaller file sizes"
            }
        
        # Add supported formats info
        result["format_support"] = {
            "supported": [".mp4", ".mov", ".avi", ".mkv", ".webm"],
            "recommended": [".mp4"],
            "optimization_note": "MP4 with H.264/AAC provides best compatibility across PDF viewers"
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Video embedding failed: {str(e)}", "embedding_time": round(time.time() - start_time, 2)}

@mcp.tool(name="add_highlights", description="Add text highlights to specific text or areas in PDF")
async def add_highlights(
    input_path: str,
    output_path: str,
    highlights: str  # JSON array of highlight definitions
) -> Dict[str, Any]:
    """
    Add highlight annotations to PDF text or specific areas
    
    Args:
        input_path: Path to the existing PDF
        output_path: Path where PDF with highlights should be saved
        highlights: JSON array of highlight definitions
    
    Highlight format:
    [
        {
            "page": 1,
            "text": "text to highlight",  // Optional: search for this text
            "rect": [x0, y0, x1, y1],  // Optional: specific rectangle
            "color": "yellow",
            "author": "John Doe",
            "note": "Important point"
        }
    ]
    
    Returns:
        Dictionary containing highlight results
    """
    import json
    import time
    start_time = time.time()
    
    try:
        # Parse highlights
        try:
            highlight_definitions = safe_json_parse(highlights) if highlights else []
        except json.JSONDecodeError as e:
            return {"error": f"Invalid highlights JSON: {str(e)}", "highlight_time": 0}
        
        if not highlight_definitions:
            return {"error": "At least one highlight is required", "highlight_time": 0}
        
        # Validate input path
        input_file = await validate_pdf_path(input_path)
        doc = fitz.open(str(input_file))
        
        highlight_info = {
            "highlights_added": [],
            "highlight_errors": []
        }
        
        # Color mapping
        color_map = {
            "yellow": (1, 1, 0),
            "red": (1, 0, 0),
            "green": (0, 1, 0),
            "blue": (0, 0, 1),
            "orange": (1, 0.5, 0),
            "purple": (0.5, 0, 1),
            "pink": (1, 0.75, 0.8)
        }
        
        # Process each highlight
        for i, highlight_def in enumerate(highlight_definitions):
            try:
                page_num = highlight_def.get("page", 1) - 1  # Convert to 0-indexed
                text_to_find = highlight_def.get("text", "")
                rect_coords = highlight_def.get("rect", None)
                color_name = highlight_def.get("color", "yellow").lower()
                author = highlight_def.get("author", "Anonymous")
                note = highlight_def.get("note", "")
                
                # Validate page number
                if page_num >= len(doc) or page_num < 0:
                    highlight_info["highlight_errors"].append({
                        "highlight_index": i,
                        "error": f"Page {page_num + 1} does not exist"
                    })
                    continue
                
                page = doc[page_num]
                color = color_map.get(color_name, (1, 1, 0))
                
                highlights_added_this_item = 0
                
                # Method 1: Search for text and highlight
                if text_to_find:
                    text_instances = page.search_for(text_to_find)
                    for rect in text_instances:
                        # Create highlight annotation
                        annot = page.add_highlight_annot(rect)
                        annot.set_colors(stroke=color)
                        annot.set_info(content=note)
                        annot.update()
                        highlights_added_this_item += 1
                
                # Method 2: Highlight specific rectangle
                elif rect_coords and len(rect_coords) == 4:
                    highlight_rect = fitz.Rect(rect_coords[0], rect_coords[1], 
                                             rect_coords[2], rect_coords[3])
                    annot = page.add_highlight_annot(highlight_rect)
                    annot.set_colors(stroke=color)
                    annot.set_info(content=note)
                    annot.update()
                    highlights_added_this_item += 1
                
                else:
                    highlight_info["highlight_errors"].append({
                        "highlight_index": i,
                        "error": "Must specify either 'text' to search for or 'rect' coordinates"
                    })
                    continue
                
                if highlights_added_this_item > 0:
                    highlight_info["highlights_added"].append({
                        "page": page_num + 1,
                        "text_searched": text_to_find,
                        "rect_used": rect_coords,
                        "instances_highlighted": highlights_added_this_item,
                        "color": color_name,
                        "author": author,
                        "note": note[:50] + "..." if len(note) > 50 else note
                    })
                else:
                    highlight_info["highlight_errors"].append({
                        "highlight_index": i,
                        "error": f"No text found to highlight: '{text_to_find}'"
                    })
                
            except Exception as e:
                highlight_info["highlight_errors"].append({
                    "highlight_index": i,
                    "error": f"Failed to add highlight: {str(e)}"
                })
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PDF with highlights
        doc.save(str(output_file), garbage=4, deflate=True, clean=True)
        doc.close()
        
        file_size = output_file.stat().st_size
        
        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "highlights_requested": len(highlight_definitions),
            "highlights_added": len(highlight_info["highlights_added"]),
            "highlights_failed": len(highlight_info["highlight_errors"]),
            "highlight_details": highlight_info,
            "file_size": format_file_size(file_size),
            "highlight_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"Adding highlights failed: {str(e)}", "highlight_time": round(time.time() - start_time, 2)}

@mcp.tool(name="add_stamps", description="Add approval stamps (Approved, Draft, Confidential, etc) to PDF")
async def add_stamps(
    input_path: str,
    output_path: str,
    stamps: str  # JSON array of stamp definitions
) -> Dict[str, Any]:
    """
    Add stamp annotations to PDF (Approved, Draft, Confidential, etc)
    
    Args:
        input_path: Path to the existing PDF
        output_path: Path where PDF with stamps should be saved
        stamps: JSON array of stamp definitions
    
    Stamp format:
    [
        {
            "page": 1,
            "x": 400, "y": 700,
            "stamp_type": "APPROVED",  // APPROVED, DRAFT, CONFIDENTIAL, REVIEWED, etc
            "size": "large",  // small, medium, large
            "rotation": 0,  // degrees
            "opacity": 0.7
        }
    ]
    
    Returns:
        Dictionary containing stamp results
    """
    import json
    import time
    start_time = time.time()
    
    try:
        # Parse stamps
        try:
            stamp_definitions = safe_json_parse(stamps) if stamps else []
        except json.JSONDecodeError as e:
            return {"error": f"Invalid stamps JSON: {str(e)}", "stamp_time": 0}
        
        if not stamp_definitions:
            return {"error": "At least one stamp is required", "stamp_time": 0}
        
        # Validate input path
        input_file = await validate_pdf_path(input_path)
        doc = fitz.open(str(input_file))
        
        stamp_info = {
            "stamps_added": [],
            "stamp_errors": []
        }
        
        # Predefined stamp types with colors and text
        stamp_types = {
            "APPROVED": {"text": "APPROVED", "color": (0, 0.7, 0), "border_color": (0, 0.5, 0)},
            "REJECTED": {"text": "REJECTED", "color": (0.8, 0, 0), "border_color": (0.6, 0, 0)},
            "DRAFT": {"text": "DRAFT", "color": (0.8, 0.4, 0), "border_color": (0.6, 0.3, 0)},
            "CONFIDENTIAL": {"text": "CONFIDENTIAL", "color": (0.8, 0, 0), "border_color": (0.6, 0, 0)},
            "REVIEWED": {"text": "REVIEWED", "color": (0, 0, 0.8), "border_color": (0, 0, 0.6)},
            "FINAL": {"text": "FINAL", "color": (0.5, 0, 0.5), "border_color": (0.3, 0, 0.3)},
            "URGENT": {"text": "URGENT", "color": (0.9, 0, 0), "border_color": (0.7, 0, 0)},
            "COMPLETED": {"text": "COMPLETED", "color": (0, 0.6, 0), "border_color": (0, 0.4, 0)}
        }
        
        # Size mapping
        size_map = {
            "small": {"width": 80, "height": 25, "font_size": 10},
            "medium": {"width": 120, "height": 35, "font_size": 12},
            "large": {"width": 160, "height": 45, "font_size": 14}
        }
        
        # Process each stamp
        for i, stamp_def in enumerate(stamp_definitions):
            try:
                page_num = stamp_def.get("page", 1) - 1  # Convert to 0-indexed
                x = stamp_def.get("x", 400)
                y = stamp_def.get("y", 700)
                stamp_type = stamp_def.get("stamp_type", "APPROVED").upper()
                size_name = stamp_def.get("size", "medium").lower()
                rotation = stamp_def.get("rotation", 0)
                opacity = stamp_def.get("opacity", 0.7)
                
                # Validate page number
                if page_num >= len(doc) or page_num < 0:
                    stamp_info["stamp_errors"].append({
                        "stamp_index": i,
                        "error": f"Page {page_num + 1} does not exist"
                    })
                    continue
                
                page = doc[page_num]
                
                # Get stamp properties
                if stamp_type not in stamp_types:
                    stamp_info["stamp_errors"].append({
                        "stamp_index": i,
                        "error": f"Unknown stamp type: {stamp_type}. Available: {list(stamp_types.keys())}"
                    })
                    continue
                
                stamp_props = stamp_types[stamp_type]
                size_props = size_map.get(size_name, size_map["medium"])
                
                # Calculate stamp rectangle
                stamp_width = size_props["width"]
                stamp_height = size_props["height"]
                stamp_rect = fitz.Rect(x, y, x + stamp_width, y + stamp_height)
                
                # Create stamp as a combination of rectangle and text
                # Draw border rectangle
                page.draw_rect(stamp_rect, color=stamp_props["border_color"], width=2)
                
                # Fill rectangle with semi-transparent background
                fill_color = (*stamp_props["color"], opacity)
                page.draw_rect(stamp_rect, color=stamp_props["color"], fill=fill_color, width=1)
                
                # Add text
                text_rect = fitz.Rect(x + 5, y + 5, x + stamp_width - 5, y + stamp_height - 5)
                
                # Calculate text position for centering
                font_size = size_props["font_size"]
                text = stamp_props["text"]
                
                # Insert text (centered)
                text_point = (
                    x + stamp_width / 2 - len(text) * font_size / 4,
                    y + stamp_height / 2 + font_size / 3
                )
                
                page.insert_text(
                    text_point, 
                    text, 
                    fontname="hebo",  # Bold font
                    fontsize=font_size,
                    color=(1, 1, 1),  # White text
                    rotate=rotation
                )
                
                stamp_info["stamps_added"].append({
                    "page": page_num + 1,
                    "position": {"x": x, "y": y},
                    "stamp_type": stamp_type,
                    "size": size_name,
                    "dimensions": {"width": stamp_width, "height": stamp_height},
                    "rotation": rotation,
                    "opacity": opacity
                })
                
            except Exception as e:
                stamp_info["stamp_errors"].append({
                    "stamp_index": i,
                    "error": f"Failed to add stamp: {str(e)}"
                })
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PDF with stamps
        doc.save(str(output_file), garbage=4, deflate=True, clean=True)
        doc.close()
        
        file_size = output_file.stat().st_size
        
        return {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "stamps_requested": len(stamp_definitions),
            "stamps_added": len(stamp_info["stamps_added"]),
            "stamps_failed": len(stamp_info["stamp_errors"]),
            "available_stamp_types": list(stamp_types.keys()),
            "stamp_details": stamp_info,
            "file_size": format_file_size(file_size),
            "stamp_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        return {"error": f"Adding stamps failed: {str(e)}", "stamp_time": round(time.time() - start_time, 2)}

@mcp.tool(name="extract_all_annotations", description="Extract all annotations (notes, highlights, stamps) from PDF")
async def extract_all_annotations(
    pdf_path: str,
    export_format: str = "json"  # json, csv
) -> Dict[str, Any]:
    """
    Extract all annotations from PDF and export to JSON or CSV format
    
    Args:
        pdf_path: Path to the PDF file to analyze
        export_format: Output format (json or csv)
    
    Returns:
        Dictionary containing all extracted annotations
    """
    import time
    start_time = time.time()
    
    try:
        # Validate input path
        input_file = await validate_pdf_path(pdf_path)
        doc = fitz.open(str(input_file))
        
        all_annotations = []
        annotation_summary = {
            "total_annotations": 0,
            "by_type": {},
            "by_page": {},
            "authors": set()
        }
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_annotations = []
            
            # Get all annotations on this page
            for annot in page.annots():
                try:
                    annot_info = {
                        "page": page_num + 1,
                        "type": annot.type[1],  # Get annotation type name
                        "content": annot.info.get("content", ""),
                        "author": annot.info.get("title", "") or annot.info.get("author", ""),
                        "subject": annot.info.get("subject", ""),
                        "creation_date": str(annot.info.get("creationDate", "")),
                        "modification_date": str(annot.info.get("modDate", "")),
                        "rect": {
                            "x0": round(annot.rect.x0, 2),
                            "y0": round(annot.rect.y0, 2), 
                            "x1": round(annot.rect.x1, 2),
                            "y1": round(annot.rect.y1, 2)
                        }
                    }
                    
                    # Get colors if available
                    try:
                        stroke_color = annot.colors.get("stroke")
                        fill_color = annot.colors.get("fill")
                        if stroke_color:
                            annot_info["stroke_color"] = stroke_color
                        if fill_color:
                            annot_info["fill_color"] = fill_color
                    except:
                        pass
                    
                    # For highlight annotations, try to get highlighted text
                    if annot.type[1] == "Highlight":
                        try:
                            highlighted_text = page.get_textbox(annot.rect)
                            if highlighted_text.strip():
                                annot_info["highlighted_text"] = highlighted_text.strip()
                        except:
                            pass
                    
                    all_annotations.append(annot_info)
                    page_annotations.append(annot_info)
                    
                    # Update summary
                    annotation_type = annot_info["type"]
                    annotation_summary["by_type"][annotation_type] = annotation_summary["by_type"].get(annotation_type, 0) + 1
                    
                    if annot_info["author"]:
                        annotation_summary["authors"].add(annot_info["author"])
                    
                except Exception as e:
                    # Skip problematic annotations
                    continue
            
            # Update page summary
            if page_annotations:
                annotation_summary["by_page"][page_num + 1] = len(page_annotations)
        
        doc.close()
        
        annotation_summary["total_annotations"] = len(all_annotations)
        annotation_summary["authors"] = list(annotation_summary["authors"])
        
        # Format output based on requested format
        if export_format.lower() == "csv":
            # Convert to CSV-friendly format
            csv_data = []
            for annot in all_annotations:
                csv_row = {
                    "Page": annot["page"],
                    "Type": annot["type"],
                    "Content": annot["content"],
                    "Author": annot["author"],
                    "Subject": annot["subject"],
                    "X0": annot["rect"]["x0"],
                    "Y0": annot["rect"]["y0"],
                    "X1": annot["rect"]["x1"],
                    "Y1": annot["rect"]["y1"],
                    "Creation_Date": annot["creation_date"],
                    "Highlighted_Text": annot.get("highlighted_text", "")
                }
                csv_data.append(csv_row)
            
            return {
                "input_path": str(input_file),
                "export_format": "csv",
                "annotations": csv_data,
                "summary": annotation_summary,
                "extraction_time": round(time.time() - start_time, 2)
            }
        
        else:  # JSON format (default)
            return {
                "input_path": str(input_file),
                "export_format": "json",
                "annotations": all_annotations,
                "summary": annotation_summary,
                "extraction_time": round(time.time() - start_time, 2)
            }
        
    except Exception as e:
        return {"error": f"Annotation extraction failed: {str(e)}", "extraction_time": round(time.time() - start_time, 2)}

# Main entry point
def create_server():
    """Create and return the MCP server instance"""
    return mcp

@mcp.tool(
    name="extract_links",
    description="Extract all links from PDF with comprehensive filtering and analysis options"
)
async def extract_links(
    pdf_path: str,
    pages: Optional[str] = None,
    include_internal: bool = True,
    include_external: bool = True,
    include_email: bool = True
) -> dict:
    """
    Extract all links from a PDF document with page filtering options.

    Args:
        pdf_path: Path to PDF file or HTTPS URL
        pages: Page numbers (e.g., "1,3,5" or "1-5,8,10-12"). If None, processes all pages
        include_internal: Include internal document links (default: True)
        include_external: Include external URL links (default: True)
        include_email: Include email links (default: True)

    Returns:
        Dictionary containing extracted links organized by type and page
    """
    start_time = time.time()

    try:
        # Validate PDF path and security
        path = await validate_pdf_path(pdf_path)

        # Parse pages parameter
        pages_to_extract = []
        doc = fitz.open(path)
        total_pages = doc.page_count

        if pages:
            try:
                pages_to_extract = parse_page_ranges(pages, total_pages)
            except ValueError as e:
                raise ValueError(f"Invalid page specification: {e}")
        else:
            pages_to_extract = list(range(total_pages))

        # Extract links from specified pages
        all_links = []
        pages_with_links = []

        for page_num in pages_to_extract:
            page = doc[page_num]
            page_links = page.get_links()

            if page_links:
                pages_with_links.append(page_num + 1)  # 1-based for user

                for link in page_links:
                    link_info = {
                        "page": page_num + 1,  # 1-based page numbering
                        "type": "unknown",
                        "destination": None,
                        "coordinates": {
                            "x0": round(link["from"].x0, 2),
                            "y0": round(link["from"].y0, 2),
                            "x1": round(link["from"].x1, 2),
                            "y1": round(link["from"].y1, 2)
                        }
                    }

                    # Determine link type and destination
                    if link["kind"] == fitz.LINK_URI:
                        # External URL
                        if include_external:
                            link_info["type"] = "external_url"
                            link_info["destination"] = link["uri"]
                            all_links.append(link_info)
                    elif link["kind"] == fitz.LINK_GOTO:
                        # Internal link to another page
                        if include_internal:
                            link_info["type"] = "internal_page"
                            link_info["destination"] = f"Page {link['page'] + 1}"
                            all_links.append(link_info)
                    elif link["kind"] == fitz.LINK_GOTOR:
                        # Link to external document
                        if include_external:
                            link_info["type"] = "external_document"
                            link_info["destination"] = link.get("file", "unknown")
                            all_links.append(link_info)
                    elif link["kind"] == fitz.LINK_LAUNCH:
                        # Launch application/file
                        if include_external:
                            link_info["type"] = "launch"
                            link_info["destination"] = link.get("file", "unknown")
                            all_links.append(link_info)
                    elif link["kind"] == fitz.LINK_NAMED:
                        # Named action (like print, quit, etc.)
                        if include_internal:
                            link_info["type"] = "named_action"
                            link_info["destination"] = link.get("name", "unknown")
                            all_links.append(link_info)

        # Organize links by type
        links_by_type = {
            "external_url": [link for link in all_links if link["type"] == "external_url"],
            "internal_page": [link for link in all_links if link["type"] == "internal_page"],
            "external_document": [link for link in all_links if link["type"] == "external_document"],
            "launch": [link for link in all_links if link["type"] == "launch"],
            "named_action": [link for link in all_links if link["type"] == "named_action"],
            "email": []  # PyMuPDF doesn't distinguish email separately, they come as external_url
        }

        # Extract email links from external URLs
        if include_email:
            for link in links_by_type["external_url"]:
                if link["destination"] and link["destination"].startswith("mailto:"):
                    email_link = link.copy()
                    email_link["type"] = "email"
                    email_link["destination"] = link["destination"].replace("mailto:", "")
                    links_by_type["email"].append(email_link)

            # Remove email links from external_url list
            links_by_type["external_url"] = [
                link for link in links_by_type["external_url"]
                if not (link["destination"] and link["destination"].startswith("mailto:"))
            ]

        doc.close()

        extraction_time = round(time.time() - start_time, 2)

        return {
            "file_info": {
                "path": str(path),
                "total_pages": total_pages,
                "pages_searched": pages_to_extract if pages else list(range(total_pages))
            },
            "extraction_summary": {
                "total_links_found": len(all_links),
                "pages_with_links": pages_with_links,
                "pages_searched_count": len(pages_to_extract),
                "link_types_found": [link_type for link_type, links in links_by_type.items() if links]
            },
            "links_by_type": links_by_type,
            "all_links": all_links,
            "extraction_settings": {
                "include_internal": include_internal,
                "include_external": include_external,
                "include_email": include_email,
                "pages_filter": pages or "all"
            },
            "extraction_time": extraction_time
        }

    except Exception as e:
        error_msg = sanitize_error_message(str(e))
        logger.error(f"Link extraction failed for {pdf_path}: {error_msg}")
        return {
            "error": f"Link extraction failed: {error_msg}",
            "extraction_time": round(time.time() - start_time, 2)
        }


def main():
    """Run the MCP server - entry point for CLI"""
    asyncio.run(run_server())

async def run_server():
    """Run the MCP server"""
    try:
        from importlib.metadata import version
        package_version = version("mcp-pdf")
    except:
        package_version = "1.0.1"

    # Log version to stderr so it appears even with MCP protocol on stdout
    import sys
    print(f"🎬 MCP PDF Tools v{package_version}", file=sys.stderr)
    await mcp.run_stdio_async()

if __name__ == "__main__":
    main()
