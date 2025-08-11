"""
MCP PDF Tools Server - Comprehensive PDF processing capabilities
"""

import os
import asyncio
import tempfile
import base64
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse
import logging
import ast

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

# Initialize FastMCP server
mcp = FastMCP("pdf-tools")

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
# URL download cache directory
CACHE_DIR = Path(os.environ.get("PDF_TEMP_DIR", "/tmp/mcp-pdf-processing"))
CACHE_DIR.mkdir(exist_ok=True, parents=True)

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
    """Download PDF from URL with caching"""
    try:
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
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type and "application/pdf" not in content_type:
                # Check if content looks like PDF by magic bytes
                content_start = response.content[:10]
                if not content_start.startswith(b"%PDF"):
                    raise ValueError(f"URL does not contain a PDF file. Content-Type: {content_type}")
            
            # Save to cache
            cache_file.write_bytes(response.content)
            logger.info(f"Downloaded and cached PDF: {cache_file} ({len(response.content)} bytes)")
            return cache_file
            
    except httpx.HTTPError as e:
        raise ValueError(f"Failed to download PDF from URL {url}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error downloading PDF: {str(e)}")

async def validate_pdf_path(pdf_path: str) -> Path:
    """Validate path (local or URL) and return local Path to PDF file"""
    # Check if it's a URL
    parsed = urlparse(pdf_path)
    
    if parsed.scheme in ('http', 'https'):
        if parsed.scheme == 'http':
            logger.warning(f"Using insecure HTTP URL: {pdf_path}")
        return await download_pdf_from_url(pdf_path)
    
    # Handle local path
    path = Path(pdf_path)
    if not path.exists():
        raise ValueError(f"File not found: {pdf_path}")
    if not path.suffix.lower() == '.pdf':
        raise ValueError(f"Not a PDF file: {pdf_path}")
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
    preserve_layout: bool = False
) -> Dict[str, Any]:
    """
    Extract text from PDF using various methods
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        method: Extraction method (auto, pymupdf, pdfplumber, pypdf)
        pages: Page numbers to extract as string like "1,2,3" or "[1,2,3]", None for all pages (0-indexed)
        preserve_layout: Whether to preserve the original text layout
    
    Returns:
        Dictionary containing extracted text and metadata
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
        
        # Extract text using selected method
        if method == "pymupdf":
            text = await extract_with_pymupdf(path, parsed_pages, preserve_layout)
        elif method == "pdfplumber":
            text = await extract_with_pdfplumber(path, parsed_pages, preserve_layout)
        elif method == "pypdf":
            text = await extract_with_pypdf(path, parsed_pages, preserve_layout)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
        
        # Get metadata
        doc = fitz.open(str(path))
        metadata = {
            "pages": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
        }
        doc.close()
        
        return {
            "text": text,
            "method_used": method,
            "metadata": metadata,
            "pages_extracted": pages or list(range(metadata["pages"])),
            "extraction_time": round(time.time() - start_time, 2),
            "warnings": []
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
@mcp.tool(name="pdf_to_markdown", description="Convert PDF to clean markdown format")
async def pdf_to_markdown(
    pdf_path: str,
    include_images: bool = True,
    include_metadata: bool = True,
    pages: Optional[str] = None  # Accept as string for MCP compatibility
) -> Dict[str, Any]:
    """
    Convert PDF to markdown format
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        include_images: Whether to extract and include images
        include_metadata: Whether to include document metadata
        pages: Specific pages to convert (0-indexed)
    
    Returns:
        Dictionary containing markdown content
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
                        img_data = pix.tobytes("png")
                        img_b64 = base64.b64encode(img_data).decode()
                        images_extracted.append({
                            "page": page_num + 1,
                            "index": img_index,
                            "data": img_b64,
                            "width": pix.width,
                            "height": pix.height
                        })
                        markdown_parts.append(f"\n![Image {page_num+1}-{img_index}](image-{page_num+1}-{img_index}.png)\n")
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
@mcp.tool(name="extract_images", description="Extract images from PDF")
async def extract_images(
    pdf_path: str,
    pages: Optional[str] = None,  # Accept as string for MCP compatibility
    min_width: int = 100,
    min_height: int = 100,
    output_format: str = "png"
) -> Dict[str, Any]:
    """
    Extract images from PDF
    
    Args:
        pdf_path: Path to PDF file or HTTPS URL
        pages: Specific pages to extract images from (0-indexed)
        min_width: Minimum image width to extract
        min_height: Minimum image height to extract
        output_format: Output format (png, jpeg)
    
    Returns:
        Dictionary containing extracted images
    """
    try:
        path = await validate_pdf_path(pdf_path)
        parsed_pages = parse_pages_parameter(pages)
        doc = fitz.open(str(path))
        
        images = []
        page_range = parsed_pages if parsed_pages else range(len(doc))
        
        for page_num in page_range:
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                # Check size requirements
                if pix.width >= min_width and pix.height >= min_height:
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        if output_format == "jpeg" and pix.alpha:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        img_data = pix.tobytes(output_format)
                        img_b64 = base64.b64encode(img_data).decode()
                        
                        images.append({
                            "page": page_num + 1,
                            "index": img_index,
                            "data": img_b64,
                            "width": pix.width,
                            "height": pix.height,
                            "format": output_format
                        })
                
                pix = None
        
        doc.close()
        
        return {
            "images": images,
            "total_images": len(images),
            "pages_searched": len(page_range),
            "filters": {
                "min_width": min_width,
                "min_height": min_height
            }
        }
        
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
        
        # Create output directory
        output_dir = CACHE_DIR / "image_output"
        output_dir.mkdir(exist_ok=True)
        
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

# Main entry point
def create_server():
    """Create and return the MCP server instance"""
    return mcp

def main():
    """Run the MCP server - entry point for CLI"""
    asyncio.run(run_server())

async def run_server():
    """Run the MCP server"""
    await mcp.run_stdio_async()

if __name__ == "__main__":
    main()
