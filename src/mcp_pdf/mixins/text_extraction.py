"""
Text Extraction Mixin - PDF text extraction and OCR capabilities
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# PDF processing libraries
import fitz  # PyMuPDF
import pdfplumber
import pypdf
import pytesseract
from pdf2image import convert_from_path

from .base import MCPMixin, mcp_tool
from ..security import validate_pdf_path, parse_pages_parameter, sanitize_error_message

logger = logging.getLogger(__name__)


class TextExtractionMixin(MCPMixin):
    """
    Handles all PDF text extraction and OCR operations.

    Tools provided:
    - extract_text: Intelligent text extraction with method selection
    - ocr_pdf: OCR processing for scanned documents
    - is_scanned_pdf: Detect if PDF is scanned/image-based
    """

    def get_mixin_name(self) -> str:
        return "TextExtraction"

    def get_required_permissions(self) -> List[str]:
        return ["read_files", "ocr_processing"]

    def _setup(self):
        """Initialize text extraction specific configuration"""
        self.max_chunk_pages = int(os.getenv("PDF_CHUNK_PAGES", "10"))
        self.max_tokens_per_chunk = int(os.getenv("PDF_MAX_TOKENS_CHUNK", "20000"))

    @mcp_tool(
        name="extract_text",
        description="Extract text from PDF with intelligent method selection and automatic chunking for large files"
    )
    async def extract_text(
        self,
        pdf_path: str,
        method: str = "auto",
        pages: Optional[str] = None,
        preserve_layout: bool = False,
        max_tokens: int = 20000,
        chunk_pages: int = 10
    ) -> Dict[str, Any]:
        """
        Extract text from PDF with intelligent method selection and automatic chunking.

        Args:
            pdf_path: Path to PDF file or URL
            method: Extraction method ("auto", "pymupdf", "pdfplumber", "pypdf")
            pages: Page specification (e.g., "1-5,10,15-20" or "all")
            preserve_layout: Whether to preserve text layout and formatting
            max_tokens: Maximum tokens to prevent MCP overflow (default 20000)
            chunk_pages: Number of pages per chunk for large PDFs

        Returns:
            Dictionary with extracted text, metadata, and processing info
        """
        start_time = time.time()

        try:
            # Validate inputs using centralized security functions
            path = await validate_pdf_path(pdf_path)
            parsed_pages = parse_pages_parameter(pages)

            # Auto-select method based on PDF characteristics
            if method == "auto":
                is_scanned = self._detect_scanned_pdf(str(path))
                if is_scanned:
                    return {
                        "success": False,
                        "error": "Scanned PDF detected. Please use the OCR tool for this file.",
                        "is_scanned": True,
                        "processing_time": round(time.time() - start_time, 2)
                    }
                method = "pymupdf"  # Default to PyMuPDF for text-based PDFs

            # Get PDF metadata and size analysis
            doc = fitz.open(str(path))
            total_pages = len(doc)
            file_size_bytes = path.stat().st_size if path.is_file() else 0
            file_size_mb = file_size_bytes / (1024 * 1024) if file_size_bytes > 0 else 0

            # Sample content for analysis
            sample_pages = min(3, total_pages)
            sample_text = ""
            for page_num in range(sample_pages):
                page = doc[page_num]
                sample_text += page.get_text()

            avg_chars_per_page = len(sample_text) / sample_pages if sample_pages > 0 else 0
            estimated_total_chars = avg_chars_per_page * total_pages
            estimated_tokens_by_density = int(estimated_total_chars / 4)

            metadata = {
                "pages": total_pages,
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "file_size_mb": round(file_size_mb, 2),
                "avg_chars_per_page": int(avg_chars_per_page),
                "estimated_total_chars": int(estimated_total_chars),
                "estimated_tokens_by_density": estimated_tokens_by_density
            }
            doc.close()

            # Enforce MCP hard limit
            effective_max_tokens = min(max_tokens, 24000)

            # Determine pages to extract
            if parsed_pages:
                pages_to_extract = parsed_pages
            else:
                pages_to_extract = list(range(total_pages))

            # Extract text using selected method
            if method == "pymupdf":
                text = self._extract_with_pymupdf(path, pages_to_extract, preserve_layout)
            elif method == "pdfplumber":
                text = self._extract_with_pdfplumber(path, pages_to_extract, preserve_layout)
            elif method == "pypdf":
                text = self._extract_with_pypdf(path, pages_to_extract, preserve_layout)
            else:
                raise ValueError(f"Unknown extraction method: {method}")

            # Estimate token count
            estimated_tokens = len(text) // 4

            # Handle large responses with intelligent chunking
            if estimated_tokens > effective_max_tokens:
                chars_per_chunk = effective_max_tokens * 4

                if len(pages_to_extract) > chunk_pages:
                    # Multiple page chunks
                    chunk_page_ranges = []
                    for i in range(0, len(pages_to_extract), chunk_pages):
                        chunk_pages_list = pages_to_extract[i:i + chunk_pages]
                        chunk_page_ranges.append(chunk_pages_list)

                    # Extract first chunk
                    if method == "pymupdf":
                        chunk_text = self._extract_with_pymupdf(path, chunk_page_ranges[0], preserve_layout)
                    elif method == "pdfplumber":
                        chunk_text = self._extract_with_pdfplumber(path, chunk_page_ranges[0], preserve_layout)
                    elif method == "pypdf":
                        chunk_text = self._extract_with_pypdf(path, chunk_page_ranges[0], preserve_layout)

                    return {
                        "success": True,
                        "text": chunk_text,
                        "method_used": method,
                        "metadata": metadata,
                        "pages_extracted": chunk_page_ranges[0],
                        "processing_time": round(time.time() - start_time, 2),
                        "chunking_info": {
                            "is_chunked": True,
                            "current_chunk": 1,
                            "total_chunks": len(chunk_page_ranges),
                            "chunk_page_ranges": chunk_page_ranges,
                            "reason": "Large PDF automatically chunked to prevent token overflow",
                            "next_chunk_command": f"Use pages parameter: \"{','.join(map(str, chunk_page_ranges[1]))}\" for chunk 2" if len(chunk_page_ranges) > 1 else None
                        }
                    }
                else:
                    # Single chunk but too much text - truncate
                    truncated_text = text[:chars_per_chunk]
                    last_sentence = truncated_text.rfind('. ')
                    if last_sentence > chars_per_chunk * 0.8:
                        truncated_text = truncated_text[:last_sentence + 1]

                    return {
                        "success": True,
                        "text": truncated_text,
                        "method_used": method,
                        "metadata": metadata,
                        "pages_extracted": pages_to_extract,
                        "processing_time": round(time.time() - start_time, 2),
                        "chunking_info": {
                            "is_truncated": True,
                            "original_estimated_tokens": estimated_tokens,
                            "returned_estimated_tokens": len(truncated_text) // 4,
                            "truncation_percentage": round((len(truncated_text) / len(text)) * 100, 1)
                        }
                    }

            # Normal response
            return {
                "success": True,
                "text": text,
                "method_used": method,
                "metadata": metadata,
                "pages_extracted": pages_to_extract,
                "character_count": len(text),
                "word_count": len(text.split()),
                "processing_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Text extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "method_attempted": method,
                "processing_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="ocr_pdf",
        description="Perform OCR on scanned PDFs with preprocessing options"
    )
    async def ocr_pdf(
        self,
        pdf_path: str,
        languages: List[str] = ["eng"],
        preprocess: bool = True,
        dpi: int = 300,
        pages: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform OCR on scanned PDF documents.

        Args:
            pdf_path: Path to PDF file or URL
            languages: List of language codes for OCR (e.g., ["eng", "fra"])
            preprocess: Whether to preprocess images for better OCR
            dpi: DPI for PDF to image conversion
            pages: Specific pages to OCR

        Returns:
            Dictionary containing OCR text and metadata
        """
        start_time = time.time()

        try:
            # Validate inputs using centralized security functions
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
                        # Convert to grayscale for better OCR
                        image = image.convert('L')

                    # Join languages for tesseract
                    lang_string = '+'.join(languages)

                    # Perform OCR
                    try:
                        text = pytesseract.image_to_string(image, lang=lang_string)
                        ocr_texts.append(text)
                    except Exception as e:
                        logger.warning(f"OCR failed for page {i+1}: {e}")
                        ocr_texts.append("")

                full_text = "\n\n".join(ocr_texts)

                return {
                    "success": True,
                    "text": full_text,
                    "pages_processed": len(images),
                    "languages": languages,
                    "dpi": dpi,
                    "preprocessed": preprocess,
                    "character_count": len(full_text),
                    "processing_time": round(time.time() - start_time, 2)
                }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"OCR processing failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="is_scanned_pdf",
        description="Detect if a PDF is scanned/image-based rather than text-based"
    )
    async def is_scanned_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze PDF to determine if it's scanned/image-based.

        Args:
            pdf_path: Path to PDF file or URL

        Returns:
            Dictionary with scan detection results and recommendations
        """
        try:
            # Validate inputs using centralized security functions
            path = await validate_pdf_path(pdf_path)
            is_scanned = self._detect_scanned_pdf(str(path))

            doc_info = self._get_document_info(path)

            return {
                "success": True,
                "is_scanned": is_scanned,
                "confidence": "high" if is_scanned else "medium",
                "recommendation": "Use OCR extraction" if is_scanned else "Use text extraction",
                "page_count": doc_info.get("page_count", 0),
                "file_size": doc_info.get("file_size", 0)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            return {
                "success": False,
                "error": error_msg
            }

    # Private helper methods (all synchronous for proper async pattern)
    def _detect_scanned_pdf(self, pdf_path: str) -> bool:
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

    def _extract_with_pymupdf(self, pdf_path: Path, pages: Optional[List[int]] = None, preserve_layout: bool = False) -> str:
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

    def _extract_with_pdfplumber(self, pdf_path: Path, pages: Optional[List[int]] = None, preserve_layout: bool = False) -> str:
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

    def _extract_with_pypdf(self, pdf_path: Path, pages: Optional[List[int]] = None, preserve_layout: bool = False) -> str:
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

    def _get_document_info(self, pdf_path: Path) -> Dict[str, Any]:
        """Get basic document information"""
        try:
            doc = fitz.open(str(pdf_path))
            info = {
                "page_count": len(doc),
                "file_size": pdf_path.stat().st_size
            }
            doc.close()
            return info
        except Exception:
            return {"page_count": 0, "file_size": 0}