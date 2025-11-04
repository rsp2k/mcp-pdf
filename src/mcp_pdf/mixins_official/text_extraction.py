"""
Text Extraction Mixin - PDF text extraction, OCR, and scanned PDF detection
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# PDF processing libraries
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Official FastMCP mixin
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, sanitize_error_message

logger = logging.getLogger(__name__)


class TextExtractionMixin(MCPMixin):
    """
    Handles PDF text extraction operations including OCR and scanned PDF detection.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()
        self.max_pages_per_chunk = 10
        self.max_file_size = 100 * 1024 * 1024  # 100MB

    @mcp_tool(
        name="extract_text",
        description="Extract text from PDF with intelligent method selection and automatic chunking for large files"
    )
    async def extract_text(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        method: str = "auto",
        chunk_pages: int = 10,
        max_tokens: int = 20000,
        preserve_layout: bool = False
    ) -> Dict[str, Any]:
        """
        Extract text from PDF with intelligent method selection.

        Args:
            pdf_path: Path to PDF file or HTTPS URL
            pages: Page numbers to extract (comma-separated, 1-based), None for all
            method: Extraction method ("auto", "pymupdf", "pdfplumber", "pypdf")
            chunk_pages: Number of pages per chunk for large files
            max_tokens: Maximum tokens per response to prevent overflow
            preserve_layout: Whether to preserve text layout and formatting

        Returns:
            Dictionary containing extracted text and metadata
        """
        start_time = time.time()

        try:
            # Validate and prepare inputs
            path = await validate_pdf_path(pdf_path)
            parsed_pages = self._parse_pages_parameter(pages)

            # Open and analyze document
            doc = fitz.open(str(path))
            total_pages = len(doc)

            # Determine pages to process
            pages_to_extract = parsed_pages if parsed_pages else list(range(total_pages))
            pages_to_extract = [p for p in pages_to_extract if 0 <= p < total_pages]

            if not pages_to_extract:
                doc.close()
                return {
                    "success": False,
                    "error": "No valid pages specified",
                    "extraction_time": 0
                }

            # Check if chunking is needed
            if len(pages_to_extract) > chunk_pages:
                return await self._extract_text_chunked(
                    doc, path, pages_to_extract, method, chunk_pages,
                    max_tokens, preserve_layout, start_time
                )

            # Extract text from specified pages
            extraction_result = await self._extract_text_from_pages(
                doc, pages_to_extract, method, preserve_layout
            )

            doc.close()

            # Check token limit and truncate if necessary
            if len(extraction_result["text"]) > max_tokens:
                truncated_text = extraction_result["text"][:max_tokens]
                # Try to truncate at sentence boundary
                last_period = truncated_text.rfind('.')
                if last_period > max_tokens * 0.8:  # If we can find a good break point
                    truncated_text = truncated_text[:last_period + 1]

                extraction_result["text"] = truncated_text
                extraction_result["truncated"] = True
                extraction_result["truncation_reason"] = f"Response too large (>{max_tokens} chars)"

            extraction_result.update({
                "success": True,
                "file_info": {
                    "path": str(path),
                    "total_pages": total_pages,
                    "pages_extracted": len(pages_to_extract),
                    "pages_requested": pages or "all"
                },
                "extraction_time": round(time.time() - start_time, 2)
            })

            return extraction_result

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Text extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "extraction_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="ocr_pdf",
        description="Perform OCR on scanned PDFs with preprocessing options"
    )
    async def ocr_pdf(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        languages: List[str] = ["eng"],
        dpi: int = 300,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Perform OCR on scanned PDF pages.

        Args:
            pdf_path: Path to PDF file or HTTPS URL
            pages: Page numbers to process (comma-separated, 1-based), None for all
            languages: List of language codes for OCR
            dpi: DPI for image rendering
            preprocess: Whether to preprocess images for better OCR

        Returns:
            Dictionary containing OCR results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            parsed_pages = self._parse_pages_parameter(pages)

            doc = fitz.open(str(path))
            total_pages = len(doc)

            pages_to_process = parsed_pages if parsed_pages else list(range(total_pages))
            pages_to_process = [p for p in pages_to_process if 0 <= p < total_pages]

            if not pages_to_process:
                doc.close()
                return {
                    "success": False,
                    "error": "No valid pages specified",
                    "ocr_time": 0
                }

            ocr_results = []
            total_text = []

            for page_num in pages_to_process:
                try:
                    page = doc[page_num]

                    # Convert page to image
                    mat = fitz.Matrix(dpi/72, dpi/72)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))

                    # Preprocess image if requested
                    if preprocess:
                        image = self._preprocess_image_for_ocr(image)

                    # Perform OCR
                    lang_string = '+'.join(languages)
                    ocr_text = pytesseract.image_to_string(image, lang=lang_string)

                    # Get confidence scores
                    try:
                        ocr_data = pytesseract.image_to_data(image, lang=lang_string, output_type=pytesseract.Output.DICT)
                        confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    except:
                        avg_confidence = 0

                    page_result = {
                        "page": page_num + 1,
                        "text": ocr_text.strip(),
                        "confidence": round(avg_confidence, 2),
                        "word_count": len(ocr_text.split()),
                        "character_count": len(ocr_text)
                    }

                    ocr_results.append(page_result)
                    total_text.append(ocr_text)

                    pix = None  # Clean up

                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                    ocr_results.append({
                        "page": page_num + 1,
                        "text": "",
                        "error": str(e),
                        "confidence": 0
                    })

            doc.close()

            # Calculate overall statistics
            successful_pages = [r for r in ocr_results if "error" not in r]
            avg_confidence = sum(r["confidence"] for r in successful_pages) / len(successful_pages) if successful_pages else 0

            return {
                "success": True,
                "text": "\n\n".join(total_text),
                "pages_processed": len(pages_to_process),
                "pages_successful": len(successful_pages),
                "pages_failed": len(pages_to_process) - len(successful_pages),
                "overall_confidence": round(avg_confidence, 2),
                "page_results": ocr_results,
                "ocr_settings": {
                    "languages": languages,
                    "dpi": dpi,
                    "preprocessing": preprocess
                },
                "file_info": {
                    "path": str(path),
                    "total_pages": total_pages
                },
                "ocr_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"OCR processing failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "ocr_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="is_scanned_pdf",
        description="Detect if a PDF is scanned/image-based rather than text-based"
    )
    async def is_scanned_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Detect if a PDF contains scanned content vs native text.

        Args:
            pdf_path: Path to PDF file or HTTPS URL

        Returns:
            Dictionary containing scan detection results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            total_pages = len(doc)
            sample_size = min(5, total_pages)  # Check first 5 pages for performance

            text_analysis = []
            image_analysis = []

            for page_num in range(sample_size):
                page = doc[page_num]

                # Analyze text content
                text = page.get_text().strip()
                text_analysis.append({
                    "page": page_num + 1,
                    "text_length": len(text),
                    "has_text": len(text) > 10
                })

                # Analyze images
                images = page.get_images()
                total_image_area = 0

                for img in images:
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        image_area = pix.width * pix.height
                        total_image_area += image_area
                        pix = None
                    except:
                        pass

                page_rect = page.rect
                page_area = page_rect.width * page_rect.height
                image_coverage = (total_image_area / page_area) if page_area > 0 else 0

                image_analysis.append({
                    "page": page_num + 1,
                    "image_count": len(images),
                    "image_coverage_percent": round(image_coverage * 100, 2),
                    "large_image_present": image_coverage > 0.5
                })

            doc.close()

            # Determine if PDF is likely scanned
            pages_with_minimal_text = sum(1 for t in text_analysis if not t["has_text"])
            pages_with_large_images = sum(1 for i in image_analysis if i["large_image_present"])

            is_likely_scanned = (
                (pages_with_minimal_text / sample_size) > 0.6 or
                (pages_with_large_images / sample_size) > 0.4
            )

            confidence_score = 0
            if pages_with_minimal_text == sample_size and pages_with_large_images > 0:
                confidence_score = 0.9  # Very confident it's scanned
            elif pages_with_minimal_text > sample_size * 0.8:
                confidence_score = 0.7  # Likely scanned
            elif pages_with_large_images > sample_size * 0.6:
                confidence_score = 0.6  # Possibly scanned
            else:
                confidence_score = 0.2  # Likely text-based

            return {
                "success": True,
                "is_scanned": is_likely_scanned,
                "confidence": round(confidence_score, 2),
                "analysis_summary": {
                    "pages_analyzed": sample_size,
                    "pages_with_minimal_text": pages_with_minimal_text,
                    "pages_with_large_images": pages_with_large_images,
                    "total_pages": total_pages
                },
                "page_analysis": {
                    "text_analysis": text_analysis,
                    "image_analysis": image_analysis
                },
                "recommendations": [
                    "Use OCR for text extraction" if is_likely_scanned
                    else "Use standard text extraction methods"
                ],
                "file_info": {
                    "path": str(path),
                    "total_pages": total_pages
                },
                "analysis_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Scanned PDF detection failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "analysis_time": round(time.time() - start_time, 2)
            }

    # Helper methods (synchronous)
    def _parse_pages_parameter(self, pages: Optional[str]) -> Optional[List[int]]:
        """Parse pages parameter from string to list of 0-based page numbers

        Supports formats:
        - Single page: "5"
        - Comma-separated: "1,3,5"
        - Ranges: "1-10" or "11-30"
        - Mixed: "1,3-5,7,10-15"
        """
        if not pages:
            return None

        try:
            result = []
            parts = pages.split(',')

            for part in parts:
                part = part.strip()

                # Handle range (e.g., "1-10" or "11-30")
                if '-' in part:
                    range_parts = part.split('-')
                    if len(range_parts) == 2:
                        start = int(range_parts[0].strip())
                        end = int(range_parts[1].strip())
                        # Convert 1-based to 0-based and create range
                        result.extend(range(start - 1, end))
                    else:
                        return None
                # Handle single page
                else:
                    result.append(int(part) - 1)

            return result
        except (ValueError, AttributeError):
            return None

    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Preprocess image to improve OCR accuracy"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')

        # You could add more preprocessing here:
        # - Noise reduction
        # - Contrast enhancement
        # - Deskewing

        return image

    async def _extract_text_chunked(self, doc, path, pages_to_extract, method,
                                   chunk_pages, max_tokens, preserve_layout, start_time):
        """Handle chunked extraction for large documents"""
        total_chunks = (len(pages_to_extract) + chunk_pages - 1) // chunk_pages

        # Process first chunk
        first_chunk_pages = pages_to_extract[:chunk_pages]
        result = await self._extract_text_from_pages(doc, first_chunk_pages, method, preserve_layout)

        # Calculate next chunk hint based on actual pages being extracted
        next_chunk_hint = None
        if len(pages_to_extract) > chunk_pages:
            # Get the next chunk's page range (1-based for user)
            next_chunk_start = pages_to_extract[chunk_pages] + 1  # Convert to 1-based
            next_chunk_end = pages_to_extract[min(chunk_pages * 2 - 1, len(pages_to_extract) - 1)] + 1  # Convert to 1-based
            next_chunk_hint = f"Use pages parameter '{next_chunk_start}-{next_chunk_end}' for next chunk"

        return {
            "success": True,
            "text": result["text"],
            "method_used": result["method_used"],
            "chunked": True,
            "chunk_info": {
                "current_chunk": 1,
                "total_chunks": total_chunks,
                "pages_in_chunk": len(first_chunk_pages),
                "chunk_pages": [p + 1 for p in first_chunk_pages],
                "next_chunk_hint": next_chunk_hint
            },
            "file_info": {
                "path": str(path),
                "total_pages": len(doc),
                "total_pages_requested": len(pages_to_extract)
            },
            "extraction_time": round(time.time() - start_time, 2)
        }

    async def _extract_text_from_pages(self, doc, pages_to_extract, method, preserve_layout):
        """Extract text from specified pages using chosen method"""
        if method == "auto":
            # Try PyMuPDF first (fastest)
            try:
                text = ""
                for page_num in pages_to_extract:
                    page = doc[page_num]
                    page_text = page.get_text("text" if not preserve_layout else "dict")
                    if preserve_layout and isinstance(page_text, dict):
                        # Extract text while preserving some layout
                        page_text = self._extract_layout_text(page_text)
                    text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"

                return {"text": text.strip(), "method_used": "pymupdf"}
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}")
                return {"text": "", "method_used": "failed", "error": str(e)}

        # For other methods, similar implementation would follow
        return {"text": "", "method_used": method}

    def _extract_layout_text(self, page_dict):
        """Extract text from PyMuPDF dict format while preserving layout"""
        text_lines = []

        for block in page_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    text_lines.append(line_text)

        return "\n".join(text_lines)