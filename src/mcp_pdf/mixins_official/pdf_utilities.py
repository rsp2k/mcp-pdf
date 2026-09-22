"""
PDF Utilities Mixin - Additional PDF processing tools
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import time
from typing import Dict, Any, Literal, Optional
import logging

# PDF processing libraries
import pymupdf

# Official FastMCP mixin
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, sanitize_error_message
from ..xfa import is_xfa_pdf as _detect_xfa
from .utils import parse_pages_parameter

logger = logging.getLogger(__name__)


class PDFUtilitiesMixin(MCPMixin):
    """
    Handles additional PDF utility operations including comparison, optimization, and repair.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()

    @mcp_tool(
        name="compare_pdfs",
        description=(
            "Report whether two PDFs differ, and roughly how much, across "
            "page count, file size, text, metadata and bookmarks. "
            "Read-only: nothing is written and neither file is modified.\n"
            "\n"
            "This is a SAMENESS CHECK, NOT A DIFF. It never tells you WHAT "
            "changed in the text and never produces a marked-up document. "
            "Two limits matter before you trust the number:\n"
            "  - text is read from AT MOST THE FIRST 10 PAGES of each file "
            "(fewer if either is shorter), so a change on page 11 is "
            "invisible;\n"
            "  - similarity_score compares the two texts position by "
            "position, so inserting or deleting a few words near the "
            "start shifts everything after it and can drive the score "
            "toward 0.0 even for near-identical documents.\n"
            "Treat a score of 1.0 / documents_identical as meaningful, and "
            "any lower score as 'differs, magnitude unreliable'. To see the "
            "actual differences, extract_text from both and compare the text "
            "yourself.\n"
            "\n"
            "comparison_type selects which sections are computed: 'text', "
            "'metadata' (title/author/producer/dates), 'structure' "
            "(bookmark outline), or 'all' (the default, runs all three). "
            "Page count and file size are always reported. overall_similarity "
            "is the mean of whichever sections ran, where metadata and "
            "structure each contribute a flat 1.0 or 0.0."
        ),
        annotations={
            "readOnlyHint": True,        # reads both files, writes nothing
            "idempotentHint": True,
            "openWorldHint": True,       # either path may be an HTTPS URL
        },
    )
    async def compare_pdfs(
        self,
        pdf_path1: str,
        pdf_path2: str,
        comparison_type: Literal["text", "structure", "metadata", "all"] = "all"
    ) -> Dict[str, Any]:
        """
        Compare two PDF files for differences.

        Args:
            pdf_path1: Path to the first PDF, or an HTTPS URL to fetch.
            pdf_path2: Path to the second PDF, or an HTTPS URL to fetch.
            comparison_type: Which comparisons to run.
                - "text": first 10 pages only, positional similarity score
                - "metadata": document info dictionary differences
                - "structure": bookmark/TOC equality and counts
                - "all": all three (default)

        Returns:
            Dict with success, comparison_summary (overall_similarity,
            documents_identical), basic_comparison (page counts, file
            sizes), and whichever of text_comparison /
            metadata_comparison / structure_comparison were requested.
        """
        start_time = time.time()

        try:
            # Validate both PDF paths
            path1 = await validate_pdf_path(pdf_path1)
            path2 = await validate_pdf_path(pdf_path2)

            doc1 = pymupdf.open(str(path1))
            doc2 = pymupdf.open(str(path2))

            comparison_results = {}

            # Basic document info comparison
            basic_comparison = {
                "pages": {"doc1": len(doc1), "doc2": len(doc2), "equal": len(doc1) == len(doc2)},
                "file_sizes": {
                    "doc1_bytes": path1.stat().st_size,
                    "doc2_bytes": path2.stat().st_size,
                    "size_diff_bytes": abs(path1.stat().st_size - path2.stat().st_size)
                }
            }

            # Text comparison
            if comparison_type in ["text", "all"]:
                text1 = ""
                text2 = ""

                # Extract text from both documents
                max_pages = min(len(doc1), len(doc2), 10)  # Limit for performance
                for page_num in range(max_pages):
                    if page_num < len(doc1):
                        text1 += doc1[page_num].get_text() + "\n"
                    if page_num < len(doc2):
                        text2 += doc2[page_num].get_text() + "\n"

                # Simple text comparison
                text_equal = text1.strip() == text2.strip()
                text_similarity = self._calculate_text_similarity(text1, text2)

                comparison_results["text_comparison"] = {
                    "texts_equal": text_equal,
                    "similarity_score": text_similarity,
                    "text1_chars": len(text1),
                    "text2_chars": len(text2),
                    "char_difference": abs(len(text1) - len(text2))
                }

            # Metadata comparison
            if comparison_type in ["metadata", "all"]:
                meta1 = doc1.metadata
                meta2 = doc2.metadata

                metadata_differences = {}
                all_keys = set(meta1.keys()) | set(meta2.keys())

                for key in all_keys:
                    val1 = meta1.get(key, "")
                    val2 = meta2.get(key, "")
                    if val1 != val2:
                        metadata_differences[key] = {"doc1": val1, "doc2": val2}

                comparison_results["metadata_comparison"] = {
                    "metadata_equal": len(metadata_differences) == 0,
                    "differences": metadata_differences,
                    "total_differences": len(metadata_differences)
                }

            # Structure comparison
            if comparison_type in ["structure", "all"]:
                toc1 = doc1.get_toc()
                toc2 = doc2.get_toc()

                structure_equal = toc1 == toc2

                comparison_results["structure_comparison"] = {
                    "bookmarks_equal": structure_equal,
                    "toc1_count": len(toc1),
                    "toc2_count": len(toc2),
                    "bookmark_difference": abs(len(toc1) - len(toc2))
                }

            doc1.close()
            doc2.close()

            # Overall similarity assessment
            similarities = []
            if "text_comparison" in comparison_results:
                similarities.append(comparison_results["text_comparison"]["similarity_score"])
            if "metadata_comparison" in comparison_results:
                similarities.append(1.0 if comparison_results["metadata_comparison"]["metadata_equal"] else 0.0)
            if "structure_comparison" in comparison_results:
                similarities.append(1.0 if comparison_results["structure_comparison"]["bookmarks_equal"] else 0.0)

            overall_similarity = sum(similarities) / len(similarities) if similarities else 0.0

            return {
                "success": True,
                "comparison_summary": {
                    "overall_similarity": round(overall_similarity, 2),
                    "comparison_type": comparison_type,
                    "documents_identical": overall_similarity == 1.0
                },
                "basic_comparison": basic_comparison,
                **comparison_results,
                "file_info": {
                    "file1": str(path1),
                    "file2": str(path2)
                },
                "comparison_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"PDF comparison failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "comparison_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="optimize_pdf",
        description=(
            "Rewrite a PDF more compactly by garbage-collecting unused "
            "objects and deflating streams. Writes a NEW file named "
            "{original_stem}_optimized.pdf IN THE SAME DIRECTORY as the "
            "input; there is no output-path parameter, so you cannot choose "
            "the name or location, and an existing file with that name is "
            "overwritten. The original is left untouched.\n"
            "\n"
            "All three levels are lossless structural cleanups. NOTHING is "
            "re-encoded: images are never downsampled and fonts are never "
            "subsetted, so on a file that is already mostly image data the "
            "saving can be near zero or even slightly negative. Check "
            "optimization_summary.reduction_percent rather than assuming a "
            "win.\n"
            "  'light'      — drop unused objects, deflate streams\n"
            "  'balanced'   — (default) the above plus clean/rebuild the "
            "content streams\n"
            "  'aggressive' — the most thorough object collection plus "
            "clean; still lossless, still no image recompression\n"
            "\n"
            "preserve_quality is accepted for backward compatibility and "
            "currently has NO effect at any level (no level is lossy), so "
            "leave it alone. To actually shrink a scan, rasterize with "
            "convert_to_images at a lower dpi instead. Use repair_pdf, not "
            "this, when the goal is to fix a damaged file."
        ),
        annotations={
            "readOnlyHint": False,       # writes {stem}_optimized.pdf
            "destructiveHint": True,     # clobbers {stem}_optimized.pdf beside the input
            "idempotentHint": True,      # re-running rewrites the same output
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
    )
    async def optimize_pdf(
        self,
        pdf_path: str,
        optimization_level: Literal["light", "balanced", "aggressive"] = "balanced",
        preserve_quality: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize PDF file for smaller size and better performance.

        Args:
            pdf_path: Path to the PDF to optimize, or an HTTPS URL. The
                output is written alongside it as {stem}_optimized.pdf.
            optimization_level: "light", "balanced" (default) or
                "aggressive". All are lossless; see the tool description for
                what each one actually does.
            preserve_quality: Ignored. Retained for signature compatibility;
                no level re-encodes image data, so there is no quality
                trade-off to control.

        Returns:
            Dict with success, optimization_summary (original and optimized
            sizes, size_reduction_bytes, reduction_percent, level used) and
            output_info with the optimized_path.
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)

            # Generate optimized filename
            optimized_path = path.parent / f"{path.stem}_optimized.pdf"

            doc = pymupdf.open(str(path))
            original_size = path.stat().st_size

            # Apply optimization based on level
            if optimization_level == "light":
                # Light optimization: remove unused objects
                doc.save(str(optimized_path), garbage=3, deflate=True)
            elif optimization_level == "balanced":
                # Balanced optimization: compression + cleanup
                doc.save(str(optimized_path), garbage=3, deflate=True, clean=True)
            elif optimization_level == "aggressive":
                # Aggressive optimization: maximum compression
                doc.save(str(optimized_path), garbage=4, deflate=True, clean=True, ascii=False)

            doc.close()

            # Check if optimization was successful
            if optimized_path.exists():
                optimized_size = optimized_path.stat().st_size
                size_reduction = original_size - optimized_size
                reduction_percent = (size_reduction / original_size) * 100 if original_size > 0 else 0

                return {
                    "success": True,
                    "optimization_summary": {
                        "original_size_bytes": original_size,
                        "optimized_size_bytes": optimized_size,
                        "size_reduction_bytes": size_reduction,
                        "reduction_percent": round(reduction_percent, 1),
                        "optimization_level": optimization_level
                    },
                    "output_info": {
                        "optimized_path": str(optimized_path),
                        "original_path": str(path)
                    },
                    "optimization_time": round(time.time() - start_time, 2)
                }
            else:
                return {
                    "success": False,
                    "error": "Optimization failed - output file not created",
                    "optimization_time": round(time.time() - start_time, 2)
                }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"PDF optimization failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "optimization_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="repair_pdf",
        description=(
            "Salvage a damaged PDF by opening it, testing every page, and "
            "rewriting the readable ones into a clean file named "
            "{original_stem}_repaired.pdf IN THE SAME DIRECTORY as the "
            "input. There is no output-path parameter, and an existing file "
            "with that name is overwritten. The original is left untouched. "
            "Pages that cannot be read are dropped, and their 1-based "
            "numbers are listed in corrupted_page_numbers, so the repaired "
            "file may have FEWER pages than the original.\n"
            "\n"
            "Two hard limits on what it can rescue:\n"
            "  - the file must still start with a valid %PDF- header, or it "
            "is rejected before repair is even attempted;\n"
            "  - the document must still be openable. If the parser cannot "
            "open it at all the call returns success=false with "
            "repair_summary.repair_successful=false and writes nothing.\n"
            "So this fixes structural damage in a file that mostly still "
            "works; it cannot reconstruct a truncated or header-less file. "
            "Run analyze_pdf_health first if you only want to know whether "
            "a file is damaged. Use optimize_pdf, not this, when the file "
            "is fine and you just want it smaller."
        ),
        annotations={
            "readOnlyHint": False,       # writes {stem}_repaired.pdf
            "destructiveHint": True,     # clobbers {stem}_repaired.pdf beside the input
            "idempotentHint": True,      # re-running rewrites the same output
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
    )
    async def repair_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Attempt to repair a corrupted or damaged PDF file.

        Args:
            pdf_path: Path to the damaged PDF, or an HTTPS URL. The repaired
                copy is written alongside it as {stem}_repaired.pdf. The file
                must have a valid %PDF- header and must still be openable.

        Returns:
            Dict with success, repair_summary (original_pages,
            recovered_pages, corrupted_pages, recovery_rate_percent),
            file_info with the repaired_path, repair_notes, and
            corrupted_page_numbers (1-based) for pages that were dropped.
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)

            # Generate repaired filename
            repaired_path = path.parent / f"{path.stem}_repaired.pdf"

            # Attempt to open and repair the PDF
            try:
                doc = pymupdf.open(str(path))

                # Check if document can be read
                total_pages = len(doc)
                readable_pages = 0
                corrupted_pages = []

                for page_num in range(total_pages):
                    try:
                        page = doc[page_num]
                        # Try to get text to verify page integrity
                        page.get_text()
                        readable_pages += 1
                    except Exception:
                        corrupted_pages.append(page_num + 1)

                # If document is readable, save a clean copy
                if readable_pages > 0:
                    # Save with repair options
                    doc.save(str(repaired_path), garbage=4, deflate=True, clean=True)

                    repair_success = True
                    repair_notes = f"Successfully repaired: {readable_pages}/{total_pages} pages recovered"
                else:
                    repair_success = False
                    repair_notes = "Document appears to be severely corrupted - no readable pages found"

                doc.close()

            except Exception as open_error:
                # Document can't be opened normally, try recovery
                repair_success = False
                repair_notes = f"Cannot open document: {str(open_error)[:100]}"

            # Check repair results
            if repair_success and repaired_path.exists():
                repaired_size = repaired_path.stat().st_size
                original_size = path.stat().st_size

                return {
                    "success": True,
                    "repair_summary": {
                        "repair_successful": True,
                        "original_pages": total_pages,
                        "recovered_pages": readable_pages,
                        "corrupted_pages": len(corrupted_pages),
                        "recovery_rate_percent": round((readable_pages / total_pages) * 100, 1) if total_pages > 0 else 0
                    },
                    "file_info": {
                        "original_path": str(path),
                        "repaired_path": str(repaired_path),
                        "original_size_bytes": original_size,
                        "repaired_size_bytes": repaired_size
                    },
                    "repair_notes": repair_notes,
                    "corrupted_page_numbers": corrupted_pages,
                    "repair_time": round(time.time() - start_time, 2)
                }
            else:
                return {
                    "success": False,
                    "repair_summary": {
                        "repair_successful": False,
                        "error_details": repair_notes
                    },
                    "file_info": {
                        "original_path": str(path)
                    },
                    "repair_time": round(time.time() - start_time, 2)
                }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"PDF repair failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "repair_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="rotate_pages",
        description=(
            "Set the display rotation of some or all pages, writing a NEW "
            "PDF. Fixes sideways scans and landscape pages.\n"
            "\n"
            "rotation is ABSOLUTE, not additive: it sets each chosen page's "
            "rotation to that value, so rotating by 90 twice still leaves "
            "the page at 90, not 180. Only 90, 180 and 270 are accepted "
            "(clockwise). There is no 0 and no negative, so you cannot "
            "un-rotate a page with this tool — pick the absolute angle you "
            "want instead.\n"
            "\n"
            "`pages` is a plain comma/range STRING of 1-based page numbers, "
            "NOT JSON: \"3\", \"1,3,5\", \"2-7\", or mixed \"1,4-6,9\". "
            "Leave it out (or null) to rotate EVERY page. A range is "
            "inclusive on both ends. Page numbers outside the document are "
            "silently dropped rather than raising, so compare "
            "rotation_summary.pages_requested with pages_rotated; a badly "
            "formed string does fail the call.\n"
            "\n"
            "output_filename is a BARE FILENAME, not a path: the file is "
            "written into the INPUT PDF's own directory. The default "
            "\"rotated_document.pdf\" is generic, so two calls on different "
            "PDFs in the same directory would overwrite each other — pass a "
            "distinct name."
        ),
        annotations={
            "readOnlyHint": False,       # writes a new PDF
            "destructiveHint": True,     # clobbers output_filename in the input's directory
            "idempotentHint": True,      # rotation is absolute, so re-running is a no-op change
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
    )
    async def rotate_pages(
        self,
        pdf_path: str,
        rotation: Literal[90, 180, 270] = 90,
        pages: Optional[str] = None,
        output_filename: str = "rotated_document.pdf"
    ) -> Dict[str, Any]:
        """
        Rotate specific pages in a PDF document.

        Args:
            pdf_path: Path to the source PDF, or an HTTPS URL. The output is
                written into this file's parent directory.
            rotation: Absolute clockwise rotation to apply: 90, 180 or 270.
                Anything else is rejected. Not cumulative across calls.
            pages: Comma/range string of 1-based page numbers, e.g. "1,4-6,9".
                Ranges are inclusive. None (the default) rotates all pages.
            output_filename: Filename only (no directory), created next to
                the input. Defaults to "rotated_document.pdf".

        Returns:
            Dict with success, rotation_summary (rotation_degrees,
            total_pages, pages_requested, pages_rotated, pages_failed),
            output_info with the full output_path, and rotated_pages as
            1-based page numbers.
        """
        start_time = time.time()

        try:
            # Validate inputs
            if rotation not in [90, 180, 270]:
                return {
                    "success": False,
                    "error": "Rotation must be 90, 180, or 270 degrees",
                    "rotation_time": round(time.time() - start_time, 2)
                }

            path = await validate_pdf_path(pdf_path)
            output_path = path.parent / output_filename

            doc = pymupdf.open(str(path))
            total_pages = len(doc)

            # Parse pages parameter
            parsed_pages = parse_pages_parameter(pages)
            if pages and parsed_pages is None:
                doc.close()
                return {
                    "success": False,
                    "error": "Invalid page numbers specified",
                    "rotation_time": round(time.time() - start_time, 2)
                }

            page_numbers = parsed_pages if parsed_pages else list(range(total_pages))
            page_numbers = [p for p in page_numbers if 0 <= p < total_pages]

            # Rotate specified pages
            pages_rotated = 0
            for page_num in page_numbers:
                try:
                    page = doc[page_num]
                    page.set_rotation(rotation)
                    pages_rotated += 1
                except Exception as e:
                    logger.warning(f"Failed to rotate page {page_num + 1}: {e}")

            # Save rotated document
            doc.save(str(output_path))
            output_size = output_path.stat().st_size
            doc.close()

            return {
                "success": True,
                "rotation_summary": {
                    "rotation_degrees": rotation,
                    "total_pages": total_pages,
                    "pages_requested": len(page_numbers),
                    "pages_rotated": pages_rotated,
                    "pages_failed": len(page_numbers) - pages_rotated
                },
                "output_info": {
                    "output_path": str(output_path),
                    "output_size_bytes": output_size
                },
                "rotated_pages": [p + 1 for p in page_numbers],
                "rotation_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Page rotation failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "rotation_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="convert_to_images",
        description=(
            "RENDER whole PDF pages to raster image files, one image per "
            "page, exactly as the page would look on screen (text, vectors "
            "and pictures flattened together). This is what you want to LOOK "
            "AT a page or feed it to a vision model.\n"
            "\n"
            "Do not confuse it with the extract_* tools, which pull out what "
            "is stored inside the file rather than a picture of the page: "
            "extract_images pulls the EMBEDDED bitmaps out on their own, and "
            "extract_vector_graphics exports pages as SVG for schematics and "
            "line art.\n"
            "\n"
            "Files are written into the INPUT PDF's own directory as "
            "{output_prefix}_{3-digit 1-based page}.{format} — for example "
            "page_001.png, page_002.png. There is no output-directory "
            "parameter, and same-named files are overwritten, so give each "
            "run a distinct output_prefix.\n"
            "\n"
            "`pages` is a plain comma/range STRING of 1-based page numbers, "
            "NOT JSON: \"3\", \"1,3,5\", \"2-7\", or mixed \"1,4-6,9\"; "
            "ranges are inclusive. Omitting it renders EVERY page, which on "
            "a long document at the default 300 dpi means a lot of large "
            "files — pass a range and consider dpi 100-150 for previews. "
            "dpi scales both dimensions (300 dpi turns a Letter page into "
            "roughly 2550x3300 px). Use 'jpg'/'jpeg' for photographic pages "
            "and smaller files; 'png' (the default) is lossless and better "
            "for text and line art. A page that fails to render is skipped "
            "with the call still reporting success, so compare "
            "conversion_summary.pages_requested with pages_converted.\n"
            "\n"
            "For a dynamic XFA form the render is Adobe's 'please open in "
            "Reader' placeholder, not the real form; the response then sets "
            "is_xfa and a warning, and extract_xfa_fields is what you "
            "actually want."
        ),
        annotations={
            "readOnlyHint": False,       # writes one image file per page
            "destructiveHint": True,     # clobbers same-named images in the input's directory
            "idempotentHint": True,      # same args re-render the same files
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
    )
    async def convert_to_images(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        dpi: int = 300,
        format: Literal["png", "jpg", "jpeg"] = "png",
        output_prefix: str = "page"
    ) -> Dict[str, Any]:
        """
        Convert PDF pages to image files.

        Args:
            pdf_path: Path to the source PDF, or an HTTPS URL. Images are
                written into this file's parent directory.
            pages: Comma/range string of 1-based page numbers, e.g.
                "1,4-6,9". Ranges are inclusive. None (the default) renders
                every page.
            dpi: Render resolution, default 300. Lower it (100-150) for
                previews or vision-model input; higher values grow both
                dimensions and the file size quadratically.
            format: "png" (default, lossless) or "jpg"/"jpeg" (smaller,
                lossy). Also becomes the file extension.
            output_prefix: Filename stem for the outputs, default "page",
                giving page_001.png, page_002.png, ...

        Returns:
            Dict with success, conversion_summary (pages_requested,
            pages_converted, pages_failed, format, dpi, total size) and
            converted_images listing each file's page, filename, absolute
            path, byte size and pixel dimensions. For a dynamic XFA form it
            also carries is_xfa, xfa_type and a warning.
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)

            # XFA early-detect. Dynamic XFA renders to the Adobe "Open in
            # Reader" placeholder page, not the real form. We still produce
            # the rendered image (the caller may want it) but flag what they
            # are actually getting so they do not trust it as the form layout.
            #
            # Locally guarded: this probe is incidental to the conversion, so
            # a pypdf failure on a file MuPDF can render must not abort the
            # render.
            try:
                xfa_info = _detect_xfa(str(path))
            except Exception as e:
                logger.warning(f"XFA probe failed, continuing: {e}")
                xfa_info = {"is_xfa": None, "detection_failed": True}

            xfa_warning = None
            if xfa_info.get("is_xfa") and xfa_info.get("xfa_type") == "dynamic":
                xfa_warning = (
                    "Dynamic XFA form. The rendered image is the Adobe "
                    "placeholder page, NOT the real form layout. Use "
                    "extract_xfa_fields for the form schema; only Adobe "
                    "Reader can render the actual form."
                )

            doc = pymupdf.open(str(path))
            total_pages = len(doc)

            # Parse pages parameter
            parsed_pages = parse_pages_parameter(pages)
            if pages and parsed_pages is None:
                doc.close()
                return {
                    "success": False,
                    "error": "Invalid page numbers specified",
                    "conversion_time": round(time.time() - start_time, 2)
                }

            page_numbers = parsed_pages if parsed_pages else list(range(total_pages))
            page_numbers = [p for p in page_numbers if 0 <= p < total_pages]

            # Convert pages to images
            converted_images = []
            pages_converted = 0

            for page_num in page_numbers:
                try:
                    page = doc[page_num]

                    # Create image from page
                    mat = pymupdf.Matrix(dpi/72, dpi/72)
                    pix = page.get_pixmap(matrix=mat)

                    # Generate filename
                    image_filename = f"{output_prefix}_{page_num + 1:03d}.{format}"
                    image_path = path.parent / image_filename

                    # Save image
                    if format.lower() in ["jpg", "jpeg"]:
                        pix.save(str(image_path), "JPEG")
                    else:
                        pix.save(str(image_path), "PNG")

                    image_size = image_path.stat().st_size

                    converted_images.append({
                        "page": page_num + 1,
                        "filename": image_filename,
                        "path": str(image_path),
                        "size_bytes": image_size,
                        "dimensions": f"{pix.width}x{pix.height}"
                    })

                    pages_converted += 1
                    pix = None

                except Exception as e:
                    logger.warning(f"Failed to convert page {page_num + 1}: {e}")

            doc.close()

            total_size = sum(img["size_bytes"] for img in converted_images)

            response = {
                "success": True,
                "conversion_summary": {
                    "pages_requested": len(page_numbers),
                    "pages_converted": pages_converted,
                    "pages_failed": len(page_numbers) - pages_converted,
                    "output_format": format,
                    "dpi": dpi,
                    "total_output_size_bytes": total_size
                },
                "converted_images": converted_images,
                "file_info": {
                    "input_path": str(path),
                    "total_pages": total_pages
                },
                "conversion_time": round(time.time() - start_time, 2)
            }
            if xfa_warning:
                response["is_xfa"] = True
                response["xfa_type"] = "dynamic"
                response["warning"] = xfa_warning
            return response

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"PDF to images conversion failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "conversion_time": round(time.time() - start_time, 2)
            }

    # Helper methods
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simplified)"""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        # Simple character-based similarity
        common_chars = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
        max_length = max(len(text1), len(text2))

        return common_chars / max_length if max_length > 0 else 1.0