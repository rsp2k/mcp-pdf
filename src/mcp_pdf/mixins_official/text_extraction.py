"""
Text Extraction Mixin - PDF text extraction, OCR, and scanned PDF detection
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional
import logging

# PDF processing libraries
import pymupdf
import pytesseract
from PIL import Image
import io

# Official FastMCP mixin
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, validate_output_path, sanitize_error_message

logger = logging.getLogger(__name__)


class TextExtractionMixin(MCPMixin):
    """
    Handles PDF text extraction operations including OCR and scanned PDF detection.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()
        self.max_pages_per_chunk = 10

    @mcp_tool(
        name="extract_text",
        description=(
            "Extract text from PDF and write to a .txt file. Returns the output "
            "file path and a short preview — full text is in the file, not in the "
            "response. Use output_directory to control where the file is saved, "
            "or set inline=True to get full text in the response instead.\n"
            "\n"
            "`method` defaults to \"auto\", which is what you want: it tries "
            "PyMuPDF, then pdfplumber, then pypdf, and returns the first "
            "engine that finds text. `method_used` names the winner, and "
            "`methods_attempted` appears when earlier engines were skipped.\n"
            "\n"
            "Name an engine to force it: \"pymupdf\" (fastest), \"pdfplumber\" "
            "(slower, reconstructs multi-column layout when combined with "
            "preserve_layout=True), \"pypdf\" (slowest and layout-blind, but "
            "most tolerant of malformed PDFs). A named engine that cannot read "
            "the file raises rather than quietly returning nothing.\n"
            "\n"
            "Empty text with success=true means every engine ran and found no "
            "text, which is the signature of a scanned PDF; the response "
            "carries an extraction_warning pointing at ocr_pdf."
        ),
        annotations={
            "readOnlyHint": False,       # writes a .txt file unless inline=True
            "destructiveHint": True,     # overwrites <stem>.txt in output_directory
            "idempotentHint": True,      # same args produce the same file
            "openWorldHint": True,       # pdf_path may be an http(s) URL
        },
    )
    async def extract_text(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        method: Literal["auto", "pymupdf", "pdfplumber", "pypdf"] = "auto",
        preserve_layout: bool = False,
        output_directory: Optional[str] = None,
        inline: bool = False,
        chunk_pages: int = 10,
        max_tokens: int = 20000
    ) -> Dict[str, Any]:
        """
        Extract text from PDF with intelligent method selection.

        By default, writes extracted text to a file and returns the path with
        a short preview. This prevents large extractions from filling the MCP
        context window. Set inline=True for the old behavior (full text in response).

        Args:
            pdf_path: Path to the PDF, or an http(s) URL to fetch.
            pages: 1-based page selection, e.g. "5", "1,3,5", "1-10",
                "1,3-5,12-20". None (the default) extracts every page.
            method: "auto" (default) cascades PyMuPDF -> pdfplumber -> pypdf
                and keeps the first result containing text; an engine that
                raises, or that returns nothing, falls through to the next.
                Name an engine to force it, in which case a failure raises
                instead of falling through. "pdfplumber" is the one worth
                naming deliberately, paired with preserve_layout=True, for
                multi-column pages.
            preserve_layout: Whether to preserve text layout and formatting
            output_directory: Directory to save the text file (default: temp directory)
            inline: Return full text in response instead of writing to file
            chunk_pages: Pages per chunk when inline=True (ignored for file output)
            max_tokens: Max chars when inline=True (ignored for file output)

        Returns:
            Dictionary with output_file path and summary, or full text if inline=True
        """
        start_time = time.time()

        try:
            # Validate and prepare inputs
            path = await validate_pdf_path(pdf_path)
            parsed_pages = self._parse_pages_parameter(pages)

            # Open and analyze document
            doc = pymupdf.open(str(path))
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

            # Inline mode: old behavior with chunking/truncation
            if inline:
                if len(pages_to_extract) > chunk_pages:
                    return await self._extract_text_chunked(
                        doc, path, pages_to_extract, method, chunk_pages,
                        max_tokens, preserve_layout, start_time
                    )

                extraction_result = await self._extract_text_from_pages(
                    doc, path, pages_to_extract, method, preserve_layout
                )
                doc.close()

                if len(extraction_result["text"]) > max_tokens:
                    truncated_text = extraction_result["text"][:max_tokens]
                    last_period = truncated_text.rfind('.')
                    if last_period > max_tokens * 0.8:
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

            # File output mode (default): extract all requested pages, write to file
            extraction_result = await self._extract_text_from_pages(
                doc, path, pages_to_extract, method, preserve_layout
            )
            doc.close()

            full_text = extraction_result["text"]

            # Setup output directory
            if output_directory:
                output_dir = validate_output_path(output_directory)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = Path(tempfile.mkdtemp(prefix="pdf_text_"))

            # Write text to file
            output_filename = f"{path.stem}.txt"
            output_path = output_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)

            # Build preview (first ~500 chars at sentence boundary)
            preview = full_text[:500]
            if len(full_text) > 500:
                last_period = preview.rfind('.')
                if last_period > 300:
                    preview = preview[:last_period + 1]
                preview += " [...]"

            word_count = len(full_text.split())
            char_count = len(full_text)
            file_size = output_path.stat().st_size

            return {
                "success": True,
                "output_file": str(output_path),
                "text_preview": preview,
                "extraction_summary": {
                    "word_count": word_count,
                    "character_count": char_count,
                    "file_size_bytes": file_size,
                    "file_size_kb": round(file_size / 1024, 1),
                    "pages_extracted": len(pages_to_extract),
                    "total_pages": total_pages,
                    "method_used": extraction_result.get("method_used", method)
                },
                "file_info": {
                    "input_path": str(path),
                    "total_pages": total_pages,
                    "pages_requested": pages or "all"
                },
                "extraction_time": round(time.time() - start_time, 2)
            }

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
        description=(
            "Read text off the PIXELS of a scanned PDF: renders each page to an "
            "image and runs Tesseract on it. Writes the recognised text to "
            "'<stem>_ocr.txt' in output_directory (a fresh temp directory when "
            "you do not give one) and returns that path plus a ~500-character "
            "preview and per-page confidence. Set inline=True to get the full "
            "text and per-page results in the response and NO file.\n"
            "\n"
            "Only use this when the page has no text layer. Run is_scanned_pdf "
            "first: if the PDF already has real text, extract_text is faster by "
            "orders of magnitude and exact, while OCR invents plausible "
            "misreadings. This is the slow tool in the set — every page is "
            "rasterised at `dpi` and passed through Tesseract, so budget "
            "roughly a second or more per page and pass `pages` rather than "
            "OCRing a long document to see one page.\n"
            "\n"
            "`pages` is a 1-based string: \"5\", \"1,3,5\", \"1-10\", or mixed "
            "\"1,3-5,12-20\". Omit for every page. `languages` is a LIST of "
            "Tesseract codes, e.g. [\"eng\"] or [\"eng\", \"spa\"]; each one "
            "needs its tesseract-ocr-<lang> data pack installed on the host or "
            "the call fails. `dpi` 300 is the sweet spot; 150 is faster and "
            "loses small type, 600 is slower and rarely better. `preprocess` "
            "just converts to greyscale in this build.\n"
            "\n"
            "Pages that fail are recorded individually and the call still "
            "returns success=true, so compare ocr_summary.pages_successful "
            "against pages_processed rather than trusting success. "
            "overall_confidence is Tesseract's mean word confidence on a 0-100 "
            "scale; below about 70 the text needs a human's eyes, and 0 means "
            "confidence could not be measured. A dynamic XFA form will OCR "
            "nothing but Adobe's \"please upgrade your reader\" placeholder — "
            "check analyze_pdf_health or is_xfa_pdf if the output looks like "
            "that."
        ),
        annotations={
            "readOnlyHint": False,       # writes a .txt file unless inline=True
            "destructiveHint": True,     # overwrites <stem>_ocr.txt in output_directory
            "idempotentHint": True,      # Tesseract is deterministic for fixed args
            "openWorldHint": True,       # pdf_path may be an http(s) URL
        },
    )
    async def ocr_pdf(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        languages: List[str] = ["eng"],
        dpi: int = 300,
        preprocess: bool = True,
        output_directory: Optional[str] = None,
        inline: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform OCR on scanned PDF pages.

        Args:
            pdf_path: Path to the PDF, or an http(s) URL to fetch.
            pages: 1-based page selection, e.g. "5", "1,3,5", "1-10",
                "1,3-5,12-20". None (the default) processes every page.
            languages: Tesseract language codes, joined with "+" internally.
                Each needs its language data installed on the host.
            dpi: Render resolution before OCR. 300 is a good default; lower
                is faster and loses small type, higher is slower.
            preprocess: Convert the rendered page to greyscale first.
            output_directory: Where to write "<stem>_ocr.txt". Overwritten if
                it already exists. Defaults to a fresh temp directory.
            inline: If True, return full OCR text plus per-page results in the
                response and write no file. Default False.

        Returns:
            File mode (default): success, output_file, text_preview, and
            ocr_summary with word_count, character_count, pages_processed,
            pages_successful, pages_failed and overall_confidence (0-100).
            Inline mode: success, text, pages_processed, pages_successful,
            overall_confidence and page_results — one entry per page with
            page, text, confidence, word_count, character_count, and an
            "error" key on the pages that failed. Inline mode returns no
            output_file and no ocr_summary.
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            parsed_pages = self._parse_pages_parameter(pages)

            doc = pymupdf.open(str(path))
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
                    mat = pymupdf.Matrix(dpi/72, dpi/72)
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
            full_text = "\n\n".join(total_text)
            word_count = len(full_text.split())
            elapsed = round(time.time() - start_time, 2)

            # ── Inline mode: return everything in the response ──
            if inline:
                return {
                    "success": True,
                    "text": full_text,
                    "pages_processed": len(pages_to_process),
                    "pages_successful": len(successful_pages),
                    "overall_confidence": round(avg_confidence, 2),
                    "page_results": ocr_results,
                    "ocr_time": elapsed,
                }

            # ── File-first mode (default): write text, return summary ──
            if output_directory:
                out_dir = Path(validate_output_path(output_directory))
            else:
                out_dir = Path(tempfile.mkdtemp(prefix="pdf_ocr_"))
            out_dir.mkdir(parents=True, exist_ok=True)

            output_filename = f"{path.stem}_ocr.txt"
            output_path = out_dir / output_filename
            output_path.write_text(full_text, encoding="utf-8")

            # Build preview (first ~500 chars at sentence boundary)
            preview = full_text[:500]
            if len(full_text) > 500:
                last_period = preview.rfind(".")
                if last_period > 300:
                    preview = preview[:last_period + 1]
                preview += " [...]"

            return {
                "success": True,
                "output_file": str(output_path),
                "text_preview": preview,
                "ocr_summary": {
                    "word_count": word_count,
                    "character_count": len(full_text),
                    "pages_processed": len(pages_to_process),
                    "pages_successful": len(successful_pages),
                    "pages_failed": len(pages_to_process) - len(successful_pages),
                    "overall_confidence": round(avg_confidence, 2),
                },
                "ocr_time": elapsed,
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
        description=(
            "Decide whether a PDF is page images with no text layer (scanned) "
            "or real selectable text. Read-only and cheap, writes nothing. Call "
            "it before extract_text or ocr_pdf so you pick the right one: "
            "is_scanned true means extract_text will come back empty and you "
            "want ocr_pdf; false means extract_text is exact and OCR would only "
            "add errors.\n"
            "\n"
            "Samples the FIRST 5 PAGES only, so a document that is text up "
            "front and scanned appendices later reads as not scanned. A page "
            "counts as textless when it yields 10 characters or fewer. The "
            "verdict is true when more than 60% of sampled pages are textless "
            "OR more than 40% carry a \"large\" image.\n"
            "\n"
            "image_coverage_percent is the share of the page area the images "
            "actually occupy (0-100), measured from their placement "
            "rectangles, and large_image_present means over half the page. "
            "`confidence` (0.6-0.9) is confidence IN THE VERDICT, whichever "
            "way is_scanned went, so a low value means the signals were mixed "
            "rather than that the document is text-based; read is_scanned for "
            "the direction and confidence for how much to trust it.\n"
            "\n"
            "This is a heuristic over a 10-page sample, not a guarantee. A "
            "born-digital report that is mostly full-bleed figures can read as "
            "scanned, and a scan with an OCR text layer already applied reads "
            "as text-based, which is usually the answer you want."
        ),
        annotations={
            "readOnlyHint": True,        # opens the PDF, writes nothing
            "idempotentHint": True,
            "openWorldHint": True,       # pdf_path may be an http(s) URL
        },
    )
    async def is_scanned_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Detect if a PDF contains scanned content vs native text.

        Args:
            pdf_path: Path to the PDF, or an http(s) URL to fetch.

        Returns:
            Dict with success plus:
              - is_scanned: the verdict
              - confidence: 0.6-0.9, confidence IN THAT VERDICT, whichever way
                it went. A low number means the signals were mixed, not that
                the document is text-based; read is_scanned for the direction.
              - analysis_summary: pages_analyzed, pages_with_minimal_text,
                pages_with_large_images, total_pages
              - page_analysis.text_analysis: per page, text_length and has_text
              - page_analysis.image_analysis: per page, image_count,
                image_coverage_percent (fraction of the page area the images
                actually occupy, 0-100) and large_image_present (>50%)
              - recommendations: one line naming OCR or standard extraction
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = pymupdf.open(str(path))

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
                        # Measure the area the image OCCUPIES ON THE PAGE, in
                        # points, not its pixel dimensions. This used to be
                        # pix.width * pix.height, a pixel count, divided below
                        # by a page area in points. The ratio of those two
                        # units means nothing: a 2000x1500 photo placed in a
                        # 4x3 inch box on a Letter page scored 3,000,000 /
                        # 484,704 = 619% "coverage", so large_image_present
                        # (threshold 0.5) fired on essentially any photo.
                        for rect in page.get_image_rects(xref):
                            total_image_area += abs(rect.width * rect.height)
                    except Exception:
                        # get_image_rects can fail on malformed xrefs; a page
                        # we cannot measure contributes 0 rather than aborting
                        # the whole scan.
                        pass

                page_rect = page.rect
                page_area = page_rect.width * page_rect.height
                # Clamp: overlapping or repeated placements can sum past the
                # page area, and a "312% covered" page helps nobody.
                image_coverage = min(total_image_area / page_area, 1.0) if page_area > 0 else 0

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

            text_ratio = pages_with_minimal_text / sample_size
            image_ratio = pages_with_large_images / sample_size

            is_likely_scanned = text_ratio > 0.6 or image_ratio > 0.4

            # Confidence is confidence IN THE VERDICT above, whichever way it
            # went. It used to be a separate ladder with its own thresholds
            # that disagreed with the verdict's: image_ratio of 0.5 makes
            # is_scanned True (> 0.4) but clears none of the confidence rungs
            # (needs > 0.6), so the tool returned is_scanned=True alongside
            # confidence 0.2 and the comment "Likely text-based". A caller
            # thresholding on confidence would discard a correct positive.
            if is_likely_scanned:
                if text_ratio == 1.0 and image_ratio > 0:
                    confidence_score = 0.9   # no text anywhere, images present
                elif text_ratio > 0.8:
                    confidence_score = 0.8   # nearly no text
                elif text_ratio > 0.6 and image_ratio > 0.4:
                    confidence_score = 0.75  # both signals agree
                else:
                    confidence_score = 0.6   # one signal only, near threshold
            else:
                if text_ratio == 0 and image_ratio == 0:
                    confidence_score = 0.9   # text on every page, no big images
                elif text_ratio < 0.2:
                    confidence_score = 0.8
                else:
                    confidence_score = 0.6   # mixed; some pages look scanned

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
        result = await self._extract_text_from_pages(doc, path, first_chunk_pages, method, preserve_layout)

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

    def _extract_with_pymupdf(self, doc, pages_to_extract, preserve_layout):
        """PyMuPDF extraction. Fastest, and correct on most PDFs."""
        parts = []
        for page_num in pages_to_extract:
            page = doc[page_num]
            page_text = page.get_text("text" if not preserve_layout else "dict")
            if preserve_layout and isinstance(page_text, dict):
                page_text = self._extract_layout_text(page_text)
            parts.append(f"\n\n--- Page {page_num + 1} ---\n\n{page_text}")
        return "".join(parts).strip()

    def _extract_with_pdfplumber(self, path, pages_to_extract, preserve_layout):
        """pdfplumber extraction. Slower, better on multi-column layouts.

        ``layout=True`` asks pdfplumber to reconstruct visual position with
        whitespace, which is the whole reason to reach for it over PyMuPDF.
        """
        import pdfplumber

        parts = []
        with pdfplumber.open(str(path)) as pdf:
            for page_num in pages_to_extract:
                if page_num >= len(pdf.pages):
                    continue
                page_text = pdf.pages[page_num].extract_text(
                    layout=preserve_layout
                ) or ""
                parts.append(f"\n\n--- Page {page_num + 1} ---\n\n{page_text}")
        return "".join(parts).strip()

    def _extract_with_pypdf(self, path, pages_to_extract, preserve_layout):
        """pypdf extraction. Slowest and layout-blind, but the most tolerant
        of malformed structure, so it is the last resort in the cascade.

        preserve_layout is accepted and ignored: pypdf exposes no layout mode.
        """
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        parts = []
        for page_num in pages_to_extract:
            if page_num >= len(reader.pages):
                continue
            page_text = reader.pages[page_num].extract_text() or ""
            parts.append(f"\n\n--- Page {page_num + 1} ---\n\n{page_text}")
        return "".join(parts).strip()

    async def _extract_text_from_pages(self, doc, path, pages_to_extract, method,
                                       preserve_layout):
        """Extract text from the given pages using the requested method.

        `method="auto"` walks the cascade README.md has always advertised,
        PyMuPDF then pdfplumber then pypdf, taking the first engine that
        returns non-empty text. An engine that raises, or that succeeds but
        finds nothing (the signature of a scanned page), falls through to the
        next. Only when all three come back empty do we report empty, and
        then with `method_used: "none"` and an `extraction_warning` rather
        than silently attributing the emptiness to whichever engine ran last.

        A named method runs only that engine and RAISES on failure. It used to
        hit an unimplemented stub returning `{"text": "", "method_used":
        method}`, so `method="pdfplumber"` reported success:true with zero
        characters and wrote a zero-byte file. Empty output and a broken
        engine looked identical, and the schema invited the mistake: "auto"
        reports `method_used: "pymupdf"`, so pinning that for determinism was
        the natural next step and silently returned nothing.
        """
        engines = {
            "pymupdf": lambda: self._extract_with_pymupdf(
                doc, pages_to_extract, preserve_layout),
            "pdfplumber": lambda: self._extract_with_pdfplumber(
                path, pages_to_extract, preserve_layout),
            "pypdf": lambda: self._extract_with_pypdf(
                path, pages_to_extract, preserve_layout),
        }

        if method != "auto":
            run = engines.get(method)
            if run is None:
                raise ValueError(
                    f"Unknown extraction method {method!r}. "
                    f"Valid: auto, {', '.join(engines)}"
                )
            # Let it raise. A named engine that cannot read the file is a
            # failure, not an empty document.
            text = run()
            result = {"text": text, "method_used": method}
            if not text:
                result["extraction_warning"] = (
                    f"{method} ran successfully but found no text on the "
                    f"requested pages. The PDF is likely scanned; try ocr_pdf, "
                    f"or method='auto' to fall through to the other engines."
                )
            return result

        attempted = []
        for name, run in engines.items():
            try:
                text = run()
            except Exception as e:
                logger.warning(f"{name} extraction failed: {e}")
                attempted.append(f"{name}: {type(e).__name__}")
                continue
            if text:
                return {
                    "text": text,
                    "method_used": name,
                    **({"methods_attempted": attempted} if attempted else {}),
                }
            attempted.append(f"{name}: no text found")

        return {
            "text": "",
            "method_used": "none",
            "methods_attempted": attempted,
            "extraction_warning": (
                "All extraction engines ran and none found text. This is the "
                "signature of a scanned or image-only PDF: use ocr_pdf, or "
                "is_scanned_pdf to confirm first."
            ),
        }

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