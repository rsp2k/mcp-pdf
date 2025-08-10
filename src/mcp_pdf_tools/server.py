"""
MCP PDF Tools Server - Comprehensive PDF processing capabilities
"""

import os
import asyncio
import tempfile
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager
import logging

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
from PIL import Image
import pandas as pd
import json
import markdown

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
async def validate_pdf_path(pdf_path: str) -> Path:
    """Validate that the path exists and is a PDF file"""
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
@mcp.tool(name="extract_text", description="Extract text from PDF with intelligent method selection")
async def extract_text(
    pdf_path: str,
    method: str = "auto",
    pages: Optional[List[int]] = None,
    preserve_layout: bool = False
) -> Dict[str, Any]:
    """
    Extract text from PDF using various methods
    
    Args:
        pdf_path: Path to the PDF file
        method: Extraction method (auto, pymupdf, pdfplumber, pypdf)
        pages: List of page numbers to extract (0-indexed), None for all pages
        preserve_layout: Whether to preserve the original text layout
    
    Returns:
        Dictionary containing extracted text and metadata
    """
    import time
    start_time = time.time()
    
    try:
        path = await validate_pdf_path(pdf_path)
        
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
            text = await extract_with_pymupdf(path, pages, preserve_layout)
        elif method == "pdfplumber":
            text = await extract_with_pdfplumber(path, pages, preserve_layout)
        elif method == "pypdf":
            text = await extract_with_pypdf(path, pages, preserve_layout)
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
    except:
        pass
    
    # Fall back to stream mode (for borderless tables)
    try:
        tables = camelot.read_pdf(str(pdf_path), pages=page_str, flavor='stream')
        return [table.df for table in tables]
    except:
        return []

async def extract_tables_tabula(pdf_path: Path, pages: Optional[List[int]] = None) -> List[pd.DataFrame]:
    """Extract tables using Tabula"""
    page_list = [p+1 for p in pages] if pages else 'all'
    
    try:
        tables = tabula.read_pdf(str(pdf_path), pages=page_list, multiple_tables=True)
        return tables
    except:
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
    pages: Optional[List[int]] = None,
    method: str = "auto",
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    Extract tables from PDF using various methods
    
    Args:
        pdf_path: Path to the PDF file
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
        all_tables = []
        methods_tried = []
        
        # Auto method: try methods in order until we find tables
        if method == "auto":
            for try_method in ["camelot", "pdfplumber", "tabula"]:
                methods_tried.append(try_method)
                
                if try_method == "camelot":
                    tables = await extract_tables_camelot(path, pages)
                elif try_method == "pdfplumber":
                    tables = await extract_tables_pdfplumber(path, pages)
                elif try_method == "tabula":
                    tables = await extract_tables_tabula(path, pages)
                
                if tables:
                    method = try_method
                    all_tables = tables
                    break
        else:
            # Use specific method
            methods_tried.append(method)
            if method == "camelot":
                all_tables = await extract_tables_camelot(path, pages)
            elif method == "pdfplumber":
                all_tables = await extract_tables_pdfplumber(path, pages)
            elif method == "tabula":
                all_tables = await extract_tables_tabula(path, pages)
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
    pages: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Perform OCR on a scanned PDF
    
    Args:
        pdf_path: Path to the PDF file
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
        
        # Convert PDF pages to images
        with tempfile.TemporaryDirectory() as temp_dir:
            if pages:
                images = []
                for page_num in pages:
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
        pdf_path: Path to the PDF file
    
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
    pages: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Convert PDF to markdown format
    
    Args:
        pdf_path: Path to the PDF file
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
        page_range = pages if pages else range(len(doc))
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
    pages: Optional[List[int]] = None,
    min_width: int = 100,
    min_height: int = 100,
    output_format: str = "png"
) -> Dict[str, Any]:
    """
    Extract images from PDF
    
    Args:
        pdf_path: Path to the PDF file
        pages: Specific pages to extract images from (0-indexed)
        min_width: Minimum image width to extract
        min_height: Minimum image height to extract
        output_format: Output format (png, jpeg)
    
    Returns:
        Dictionary containing extracted images
    """
    try:
        path = await validate_pdf_path(pdf_path)
        doc = fitz.open(str(path))
        
        images = []
        page_range = pages if pages else range(len(doc))
        
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
        pdf_path: Path to the PDF file
    
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
        except:
            pass
            
        try:
            for page in doc:
                if page.get_links():
                    has_links = True
                    break
        except:
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
        except:
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
