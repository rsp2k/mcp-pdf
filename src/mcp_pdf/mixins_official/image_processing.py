"""
Image Processing Mixin - PDF image extraction and markdown conversion
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Literal, Optional, List
import logging

# PDF and image processing libraries
import pymupdf

# Official FastMCP mixin
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, validate_output_path, sanitize_error_message
from .utils import parse_pages_parameter

logger = logging.getLogger(__name__)


class ImageProcessingMixin(MCPMixin):
    """
    Handles PDF image extraction and markdown conversion operations.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()

    @mcp_tool(
        name="extract_images",
        description=(
            "Extract the EMBEDDED raster images from a PDF and write each one as "
            "a separate file. Returns a summary plus one entry per image (filename, "
            "absolute path, page, pixel dimensions, byte size). Use pdf_to_markdown "
            "instead when you want the page text with the images referenced inline, "
            "and extract_vector_graphics for charts/schematics drawn as vector paths "
            "rather than stored as bitmaps.\n"
            "\n"
            "Files land in output_directory (created if missing); when it is omitted "
            "a fresh temp directory is made and its path is returned in "
            "extraction_summary.output_directory. Filenames are "
            "'{pdf_stem}_page_{N}_img_{M}.{output_format}' and an existing file with "
            "the same name is overwritten.\n"
            "\n"
            "min_width/min_height (both default 100 px) skip any image smaller than "
            "the limit in EITHER dimension, which drops logos, rules and spacer "
            "GIFs. Raise them to cut noise; set both to 1 to keep everything. "
            "Skipped and failed images are only counted in images_skipped, so a "
            "successful call with images_extracted == 0 means everything was "
            "filtered out, not that the PDF has no images.\n"
            "\n"
            "CAVEAT on include_context: the 'context' string is NOT text adjacent to "
            "the image. It is a context_chars-wide slice taken from the MIDDLE of the "
            "page's text, identical for every image on that page. Treat it as a rough "
            "page hint only; leave include_context=False if you need real captions "
            "and get them from extract_text or pdf_to_markdown."
        ),
        annotations={
            "readOnlyHint": False,       # writes one image file per extracted image
            "destructiveHint": True,     # overwrites same-named files in output_directory
            "idempotentHint": True,      # deterministic filenames; re-running rewrites the same set
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
    )
    async def extract_images(
        self,
        pdf_path: str,
        output_directory: Optional[str] = None,
        min_width: int = 100,
        min_height: int = 100,
        output_format: Literal["png", "jpg", "jpeg"] = "png",
        pages: Optional[str] = None,
        include_context: bool = True,
        context_chars: int = 200
    ) -> Dict[str, Any]:
        """
        Extract images from PDF with custom output directory and clean summary.

        Args:
            pdf_path: Path to the PDF, or an HTTPS URL to fetch.
            output_directory: Directory to write the images into, created if it
                does not exist. Defaults to a new temp directory whose path is
                returned as extraction_summary.output_directory.
            min_width: Skip images narrower than this many pixels (default 100).
            min_height: Skip images shorter than this many pixels (default 100).
                An image is skipped if EITHER dimension is under its limit.
            output_format: "png" (default), "jpg" or "jpeg". "jpg"/"jpeg" encode
                as JPEG, anything else as PNG; the value is also used verbatim
                as the file extension.
            pages: Pages to scan, 1-based. Accepts single pages, comma lists and
                ranges: "5", "1,3,5", "1-10", "1,3-5,7". None (default) means
                every page. A string that fails to parse is treated as None.
            include_context: Attach a "context" string to each image entry.
                See the caveat in the tool description: it is a slice from the
                middle of the page text, not text near the image.
            context_chars: Width in characters of that page-text slice
                (default 200).

        Returns:
            Dict with success, extraction_summary (images_extracted,
            images_skipped, pages_processed, total size, output_directory), an
            "images" list of per-file metadata, the filter settings used, and
            file_info. Images below the size filter or that failed to decode
            are counted in images_skipped, not raised.
        """
        start_time = time.time()

        try:
            # Validate PDF path
            input_pdf_path = await validate_pdf_path(pdf_path)

            # Setup output directory
            if output_directory:
                output_dir = validate_output_path(output_directory)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = Path(tempfile.mkdtemp(prefix="pdf_images_"))

            # Parse pages parameter
            parsed_pages = parse_pages_parameter(pages)

            # Open PDF document
            doc = pymupdf.open(str(input_pdf_path))
            total_pages = len(doc)

            # Determine pages to process
            pages_to_process = parsed_pages if parsed_pages else list(range(total_pages))
            pages_to_process = [p for p in pages_to_process if 0 <= p < total_pages]

            if not pages_to_process:
                doc.close()
                return {
                    "success": False,
                    "error": "No valid pages specified",
                    "extraction_time": round(time.time() - start_time, 2)
                }

            extracted_images = []
            images_extracted = 0
            images_skipped = 0

            for page_num in pages_to_process:
                try:
                    page = doc[page_num]
                    image_list = page.get_images()

                    # Get page text for context if requested
                    page_text = page.get_text() if include_context else ""

                    for img_index, img in enumerate(image_list):
                        try:
                            # Get image data
                            xref = img[0]
                            pix = pymupdf.Pixmap(doc, xref)

                            # Check image dimensions
                            if pix.width < min_width or pix.height < min_height:
                                images_skipped += 1
                                pix = None
                                continue

                            # Convert CMYK to RGB if necessary
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                pass
                            else:  # CMYK: convert to RGB first
                                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

                            # Generate filename
                            base_name = input_pdf_path.stem
                            filename = f"{base_name}_page_{page_num + 1}_img_{img_index + 1}.{output_format}"
                            output_path = output_dir / filename

                            # Save image
                            if output_format.lower() in ["jpg", "jpeg"]:
                                pix.save(str(output_path), "JPEG")
                            else:
                                pix.save(str(output_path), "PNG")

                            # Get file size
                            file_size = output_path.stat().st_size

                            # Extract context if requested
                            context_text = ""
                            if include_context and page_text:
                                # Simple context extraction - could be enhanced
                                start_pos = max(0, len(page_text)//2 - context_chars//2)
                                context_text = page_text[start_pos:start_pos + context_chars].strip()

                            # Add to results
                            image_info = {
                                "filename": filename,
                                "path": str(output_path),
                                "page": page_num + 1,
                                "image_index": img_index + 1,
                                "width": pix.width,
                                "height": pix.height,
                                "format": output_format.upper(),
                                "size_bytes": file_size,
                                "size_kb": round(file_size / 1024, 1)
                            }

                            if include_context and context_text:
                                image_info["context"] = context_text

                            extracted_images.append(image_info)
                            images_extracted += 1

                            pix = None  # Clean up

                        except Exception as e:
                            logger.warning(f"Failed to extract image {img_index + 1} from page {page_num + 1}: {e}")
                            images_skipped += 1

                except Exception as e:
                    logger.warning(f"Failed to process page {page_num + 1}: {e}")

            doc.close()

            # Calculate total output size
            total_size = sum(img["size_bytes"] for img in extracted_images)

            return {
                "success": True,
                "extraction_summary": {
                    "images_extracted": images_extracted,
                    "images_skipped": images_skipped,
                    "pages_processed": len(pages_to_process),
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "output_directory": str(output_dir)
                },
                "images": extracted_images,
                "filter_settings": {
                    "min_width": min_width,
                    "min_height": min_height,
                    "output_format": output_format,
                    "include_context": include_context
                },
                "file_info": {
                    "input_path": str(input_pdf_path),
                    "total_pages": total_pages,
                    "pages_processed": pages or "all"
                },
                "extraction_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Image extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "extraction_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="pdf_to_markdown",
        description=(
            "Convert PDF to markdown and write to a .md file. Raster images are "
            "extracted to {output_directory}/images/ and vector graphics (charts, "
            "schematics, diagrams) to {output_directory}/vectors/ as SVG. Returns "
            "the output file path and a short preview — full markdown is in the file. "
            "Set inline=True to get full markdown in the response instead. "
            "Use output_filename to override the default .md filename. "
            "Set vector_fallback_raster=True to render pages with sub-threshold "
            "drawings as raster images instead of skipping them entirely."
        ),
        annotations={
            "readOnlyHint": False,       # writes the .md plus images/ and vectors/ files
            "destructiveHint": True,     # overwrites same-named files under output_directory
            "idempotentHint": True,      # deterministic filenames; re-running rewrites the same set
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
    )
    async def pdf_to_markdown(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        include_images: bool = True,
        include_metadata: bool = True,
        output_directory: Optional[str] = None,
        output_filename: Optional[str] = None,
        min_width: int = 100,
        min_height: int = 100,
        image_format: Literal["png", "jpg", "jpeg"] = "png",
        inline: bool = False,
        include_vectors: bool = True,
        vector_min_drawings: int = 5,
        vector_min_complexity: int = 50,
        vector_fallback_raster: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert PDF to clean markdown format and write to file.

        By default, writes markdown to a file, extracts raster images to an images/
        subdirectory, and extracts significant vector graphics (charts, schematics,
        diagrams) to a vectors/ subdirectory as SVG. Returns file path + summary to
        avoid filling the MCP context window. Set inline=True for full markdown in
        response.

        Args:
            pdf_path: Path to the PDF, or an HTTPS URL to fetch.
            pages: Pages to convert, 1-based. Accepts single pages, comma lists
                and ranges: "5", "1,3,5", "1-10", "1,3-5,7". None (default)
                means every page. An unparseable string is treated as None.
                A "## Page N" header is inserted only when more than one page
                is converted.
            include_images: Whether to include raster images in markdown
            include_metadata: Prepend a "# Document Metadata" block built from
                the PDF's own metadata dict. Skipped when every value is empty.
            output_directory: Directory for output .md file and images/ subdirectory.
                Defaults to a temp directory if not specified.
            output_filename: Custom filename for the output .md file (e.g., "chapter_1.md").
                A missing ".md" suffix is appended. Defaults to the PDF filename
                with .md extension.
            min_width: Minimum image width in pixels (default 100). An image is
                skipped when EITHER dimension is below its limit.
            min_height: Minimum image height in pixels (default 100).
            image_format: "png" (default), "jpg" or "jpeg". "jpg"/"jpeg" encode
                as JPEG, anything else as PNG; the value is also the file
                extension.
            inline: Return the full markdown in the response instead of writing
                the .md file. NOTE: images and vectors are still extracted to
                disk under output_directory, so this is not a dry run.
            include_vectors: Extract significant vector graphics as SVG (default: True).
                Detects charts, schematics, and technical drawings automatically.
            vector_min_drawings: Minimum drawing count per page to consider (default: 5)
            vector_min_complexity: Minimum total path items for extraction (default: 50)
            vector_fallback_raster: When True, pages with drawings below the vector
                complexity threshold are rendered as full-page raster images (PNG at
                150 DPI) instead of being skipped. Captures charts and diagrams that
                are too simple for SVG extraction but still visually meaningful.

        Returns:
            Dictionary with output_file path and summary, or full markdown if inline=True
        """
        start_time = time.time()

        try:
            # Validate PDF path
            input_pdf_path = await validate_pdf_path(pdf_path)

            # Parse pages parameter
            parsed_pages = parse_pages_parameter(pages)

            # Open PDF document
            doc = pymupdf.open(str(input_pdf_path))
            total_pages = len(doc)

            # Determine pages to process
            pages_to_process = parsed_pages if parsed_pages else list(range(total_pages))
            pages_to_process = [p for p in pages_to_process if 0 <= p < total_pages]

            # Setup output directory — always needed (file output is the default)
            images_extracted = 0
            images_skipped = 0
            vectors_extracted = 0
            raster_fallbacks = 0
            extracted_image_info = []
            extracted_vector_info = []
            vector_diagnostics = []

            if output_directory:
                output_dir = validate_output_path(output_directory)
            else:
                output_dir = Path(tempfile.mkdtemp(prefix="pdf_markdown_"))
            output_dir.mkdir(parents=True, exist_ok=True)
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            if include_vectors:
                vectors_dir = output_dir / "vectors"
                vectors_dir.mkdir(parents=True, exist_ok=True)

            markdown_parts = []

            # Add metadata if requested
            if include_metadata:
                metadata = doc.metadata
                if any(metadata.values()):
                    markdown_parts.append("# Document Metadata\n")
                    for key, value in metadata.items():
                        if value:
                            clean_key = key.replace("Date", " Date").title()
                            markdown_parts.append(f"**{clean_key}:** {value}\n")
                    markdown_parts.append("\n---\n\n")

            # Extract content from each page
            for page_num in pages_to_process:
                try:
                    page = doc[page_num]

                    # Add page header
                    if len(pages_to_process) > 1:
                        markdown_parts.append(f"## Page {page_num + 1}\n\n")

                    # Extract text content
                    page_text = page.get_text()
                    if page_text.strip():
                        # Clean up text formatting
                        cleaned_text = self._clean_text_for_markdown(page_text)
                        markdown_parts.append(cleaned_text)
                        markdown_parts.append("\n\n")

                    # Extract images if requested
                    if include_images:
                        image_list = page.get_images()

                        for img_index, img in enumerate(image_list):
                            try:
                                alt_text = f"Image {img_index + 1} from page {page_num + 1}"
                                xref = img[0]
                                pix = pymupdf.Pixmap(doc, xref)

                                if pix.width < min_width or pix.height < min_height:
                                    images_skipped += 1
                                    pix = None
                                    continue

                                # Convert CMYK to RGB if necessary
                                if pix.n - pix.alpha >= 4:
                                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

                                base_name = input_pdf_path.stem
                                filename = f"{base_name}_page_{page_num + 1}_img_{img_index + 1}.{image_format}"
                                img_path = images_dir / filename

                                if image_format.lower() in ["jpg", "jpeg"]:
                                    pix.save(str(img_path), "JPEG")
                                else:
                                    pix.save(str(img_path), "PNG")

                                file_size = img_path.stat().st_size
                                extracted_image_info.append({
                                    "filename": filename,
                                    "path": str(img_path),
                                    "page": page_num + 1,
                                    "width": pix.width,
                                    "height": pix.height,
                                    "size_bytes": file_size
                                })
                                images_extracted += 1
                                pix = None

                                markdown_parts.append(f"![{alt_text}](./images/{filename})\n\n")

                            except Exception as e:
                                logger.warning(f"Failed to process image {img_index + 1} on page {page_num + 1}: {e}")
                                images_skipped += 1

                    # Extract significant vector graphics as SVG
                    if include_vectors:
                        try:
                            drawings = page.get_drawings()
                            if self._is_vector_significant(
                                drawings, vector_min_drawings, vector_min_complexity
                            ):
                                base_name = input_pdf_path.stem
                                svg_content = page.get_svg_image(text_as_path=False)
                                svg_filename = f"{base_name}_page_{page_num + 1}.svg"
                                svg_path = vectors_dir / svg_filename
                                with open(svg_path, 'w', encoding='utf-8') as f:
                                    f.write(svg_content)
                                file_size = svg_path.stat().st_size
                                extracted_vector_info.append({
                                    "filename": svg_filename,
                                    "path": str(svg_path),
                                    "page": page_num + 1,
                                    "drawing_count": len(drawings),
                                    "total_items": sum(
                                        len(d.get("items", [])) for d in drawings
                                    ),
                                    "size_bytes": file_size,
                                })
                                vectors_extracted += 1
                                markdown_parts.append(
                                    f"![Page {page_num + 1} diagram](./vectors/{svg_filename})\n\n"
                                )
                            elif drawings:
                                # Page has drawings but below SVG complexity threshold
                                diag_entry = {
                                    "page": page_num + 1,
                                    "drawing_count": len(drawings),
                                    "total_path_items": sum(len(d.get("items", [])) for d in drawings),
                                    "raster_images_on_page": len(page.get_images()),
                                }

                                if vector_fallback_raster:
                                    # Render full page as raster image at 150 DPI
                                    try:
                                        base_name = input_pdf_path.stem
                                        pix = page.get_pixmap(dpi=150)
                                        fallback_filename = f"{base_name}_page_{page_num + 1}_fallback.png"
                                        fallback_path = images_dir / fallback_filename
                                        pix.save(str(fallback_path))
                                        file_size = fallback_path.stat().st_size
                                        extracted_image_info.append({
                                            "filename": fallback_filename,
                                            "path": str(fallback_path),
                                            "page": page_num + 1,
                                            "width": pix.width,
                                            "height": pix.height,
                                            "size_bytes": file_size,
                                            "type": "vector_fallback",
                                        })
                                        raster_fallbacks += 1
                                        pix = None
                                        markdown_parts.append(
                                            f"![Page {page_num + 1} content](./images/{fallback_filename})\n\n"
                                        )
                                        diag_entry["reason"] = "raster_fallback_rendered"
                                    except Exception as fb_exc:
                                        logger.warning(
                                            "Raster fallback failed for page %d: %s",
                                            page_num + 1, fb_exc,
                                        )
                                        diag_entry["reason"] = "raster_fallback_failed"
                                else:
                                    diag_entry["reason"] = "below_complexity_threshold"

                                vector_diagnostics.append(diag_entry)
                        except Exception as e:
                            logger.warning(f"Failed to extract vectors from page {page_num + 1}: {e}")

                except Exception as e:
                    logger.warning(f"Failed to process page {page_num + 1}: {e}")
                    markdown_parts.append(f"*[Error processing page {page_num + 1}: {str(e)[:100]}]*\n\n")

            doc.close()

            # Combine all markdown parts
            full_markdown = "".join(markdown_parts)

            # Calculate statistics
            word_count = len(full_markdown.split())
            line_count = len(full_markdown.split('\n'))
            char_count = len(full_markdown)

            conversion_summary = {
                "pages_converted": len(pages_to_process),
                "total_pages": total_pages,
                "word_count": word_count,
                "line_count": line_count,
                "character_count": char_count,
                "images_extracted": images_extracted,
                "images_skipped": images_skipped,
                "vectors_extracted": vectors_extracted,
                "raster_fallbacks": raster_fallbacks,
            }

            # Inline mode: return full markdown in response
            if inline:
                result = {
                    "success": True,
                    "markdown": full_markdown,
                    "conversion_summary": conversion_summary,
                    "image_output": {
                        "images_directory": str(images_dir),
                        "images": extracted_image_info,
                    },
                    "file_info": {
                        "input_path": str(input_pdf_path),
                        "pages_processed": pages or "all",
                    },
                    "conversion_time": round(time.time() - start_time, 2),
                }
                if include_vectors and extracted_vector_info:
                    result["vector_output"] = {
                        "vectors_directory": str(vectors_dir),
                        "vectors_extracted": vectors_extracted,
                        "vectors": extracted_vector_info,
                    }
                if include_vectors:
                    result["vector_diagnostics"] = {
                        "pages_with_vectors": vectors_extracted,
                        "pages_with_drawings_skipped": len(vector_diagnostics),
                        "pages_analyzed": len(pages_to_process),
                        "skipped_pages": vector_diagnostics[:20],
                    }
                return result

            # File output mode (default): write .md file, return path + summary
            if output_filename:
                if not output_filename.endswith('.md'):
                    output_filename += '.md'
                md_path = output_dir / output_filename
            else:
                md_path = output_dir / f"{input_pdf_path.stem}.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(full_markdown)

            # Build preview (first ~500 chars at sentence boundary)
            preview = full_markdown[:500]
            if len(full_markdown) > 500:
                last_period = preview.rfind('.')
                if last_period > 300:
                    preview = preview[:last_period + 1]
                preview += " [...]"

            result = {
                "success": True,
                "output_file": str(md_path),
                "markdown_preview": preview,
                "conversion_summary": conversion_summary,
                "image_output": {
                    "images_directory": str(images_dir),
                    "images_extracted": images_extracted,
                    "images_skipped": images_skipped,
                    "filter_settings": {
                        "min_width": min_width,
                        "min_height": min_height,
                        "image_format": image_format,
                    },
                    "images": extracted_image_info,
                },
                "file_info": {
                    "input_path": str(input_pdf_path),
                    "output_directory": str(output_dir),
                    "pages_processed": pages or "all",
                },
                "conversion_time": round(time.time() - start_time, 2),
            }
            if include_vectors and extracted_vector_info:
                result["vector_output"] = {
                    "vectors_directory": str(vectors_dir),
                    "vectors_extracted": vectors_extracted,
                    "vectors": extracted_vector_info,
                }
            if include_vectors:
                result["vector_diagnostics"] = {
                    "pages_with_vectors": vectors_extracted,
                    "pages_with_drawings_skipped": len(vector_diagnostics),
                    "pages_analyzed": len(pages_to_process),
                    "skipped_pages": vector_diagnostics[:20],
                }
            return result

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"PDF to markdown conversion failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "conversion_time": round(time.time() - start_time, 2)
            }

    # Helper methods
    # Note: Now using shared parse_pages_parameter from utils.py

    def _clean_text_for_markdown(self, text: str) -> str:
        """Clean and format text for markdown output"""
        # Basic text cleaning
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line:
                # Escape markdown special characters if they appear to be literal
                # (This is a basic implementation - could be enhanced)
                if not self._looks_like_markdown_formatting(line):
                    line = line.replace('*', '\\*').replace('_', '\\_').replace('#', '\\#')

                cleaned_lines.append(line)

        # Join lines with proper spacing
        result = '\n'.join(cleaned_lines)

        # Clean up excessive whitespace
        while '\n\n\n' in result:
            result = result.replace('\n\n\n', '\n\n')

        return result

    def _looks_like_markdown_formatting(self, line: str) -> bool:
        """Simple heuristic to detect if line contains intentional markdown formatting"""
        # Very basic check - could be enhanced
        markdown_patterns = ['# ', '## ', '### ', '* ', '- ', '1. ', '**', '__']
        return any(pattern in line for pattern in markdown_patterns)

    def _is_vector_significant(self, drawings, min_drawings=5, min_complexity=50):
        """Detect if a page's drawings represent meaningful vector content (charts, schematics).

        Uses a multi-tier heuristic adapted from extract_charts:
        1. Drawing count gate — filters pages with only border lines
        2. Total path complexity — charts and schematics have many path items
        3. Single complex drawing — catches large diagrams even on sparse pages
        """
        if len(drawings) < min_drawings:
            return False
        total_items = sum(len(d.get("items", [])) for d in drawings)
        if total_items >= min_complexity:
            return True
        for d in drawings:
            items = d.get("items", [])
            rect = d.get("rect", pymupdf.Rect(0, 0, 0, 0))
            if len(items) > 20 and (rect.width > 200 or rect.height > 150):
                return True
        return False

    @mcp_tool(
        name="extract_vector_graphics",
        description=(
            "Export PDF pages as SVG, capturing content drawn as vector paths: "
            "circuit schematics, IC block diagrams, response curves, dimensioned "
            "package outlines, PCB layouts. Use extract_images instead for "
            "content stored as bitmaps, and pdf_to_markdown when you want text "
            "plus auto-detected diagrams in one pass.\n"
            "\n"
            "UNCONDITIONAL, unlike pdf_to_markdown's vector step: every requested "
            "page is exported with no complexity threshold, so a text-only page "
            "still produces an SVG. Check the per-page drawing_count in the result "
            "to tell a real diagram from a page whose only 'vectors' are table "
            "rules.\n"
            "\n"
            "Modes:\n"
            "  full_page (default) — the whole page rendered by PyMuPDF, layout, "
            "colours and text preserved. This is the one you want for diagrams.\n"
            "  drawings_only — a hand-built SVG of just the path geometry. Only "
            "lines, rectangles, quads and cubic beziers are converted, and TEXT IS "
            "ALWAYS DROPPED regardless of include_text, so labels, pin names and "
            "axis values disappear. A page with zero drawings yields no file and is "
            "reported as skipped.\n"
            "  both — write both files per page.\n"
            "\n"
            "Files go to output_directory (created if missing; a temp directory when "
            "omitted) as '{pdf_stem}_page_{N}.svg' and "
            "'{pdf_stem}_page_{N}_drawings.svg'; same-named files are overwritten. "
            "An invalid mode returns success=false without writing anything, while a "
            "page that fails mid-run is reported as a per-page error with the overall "
            "call still success=true."
        ),
        annotations={
            "readOnlyHint": False,       # writes one or two SVG files per page
            "destructiveHint": True,     # overwrites same-named files in output_directory
            "idempotentHint": True,      # deterministic filenames; re-running rewrites the same set
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
    )
    async def extract_vector_graphics(
        self,
        pdf_path: str,
        output_directory: Optional[str] = None,
        pages: Optional[str] = None,
        mode: Literal["full_page", "drawings_only", "both"] = "full_page",
        include_text: bool = True,
        simplify_paths: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract vector graphics from PDF pages as SVG files.

        Perfect for extracting:
        - IC functional diagrams from datasheets
        - Frequency response charts and line graphs
        - Package outline drawings (dimensioned technical drawings)
        - Circuit schematics
        - PCB layout diagrams

        Args:
            pdf_path: Path to the PDF, or an HTTPS URL to fetch.
            output_directory: Directory to write the SVG files into, created if
                missing. Defaults to a new temp directory whose path comes back
                as extraction_summary.output_directory.
            pages: Pages to export, 1-based. Accepts single pages, comma lists
                and ranges: "5", "1,3,5", "1-10", "1,3-5,7". None (default)
                means every page. An unparseable string is treated as None.
            mode: "full_page" (default) renders the complete page; "drawings_only"
                rebuilds only the path geometry and always omits text;
                "both" writes one file of each. Any other value returns an error.
            include_text: Only affects full_page mode. True (default) keeps text
                as selectable SVG <text> elements; False converts glyphs to
                outlined paths — the text still LOOKS present but is no longer
                selectable or searchable. drawings_only ignores this flag and
                never carries text.
            simplify_paths: Only affects full_page mode. Rounds coordinates with
                three or more decimals down to one decimal place to shrink the
                file. It does not remove or merge paths, and does not touch the
                drawings_only output.

        Returns:
            Dict with success, extraction_summary (pages_processed,
            pages_successful, mode, total size, output_directory), an svg_files
            list carrying per-page page/has_text/drawing_count plus a
            "full_page" and/or "drawings_only" entry (or "error" for a page that
            failed), the settings used, and viewing hints.
        """
        start_time = time.time()

        try:
            # Validate PDF path
            input_pdf_path = await validate_pdf_path(pdf_path)

            # Setup output directory
            if output_directory:
                output_dir = validate_output_path(output_directory)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = Path(tempfile.mkdtemp(prefix="pdf_vectors_"))

            # Parse pages parameter
            parsed_pages = parse_pages_parameter(pages)

            # Validate mode
            valid_modes = ["full_page", "drawings_only", "both"]
            if mode not in valid_modes:
                return {
                    "success": False,
                    "error": f"Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}",
                    "extraction_time": round(time.time() - start_time, 2)
                }

            # Open PDF document
            doc = pymupdf.open(str(input_pdf_path))
            total_pages = len(doc)

            # Determine pages to process
            pages_to_process = parsed_pages if parsed_pages else list(range(total_pages))
            pages_to_process = [p for p in pages_to_process if 0 <= p < total_pages]

            if not pages_to_process:
                doc.close()
                return {
                    "success": False,
                    "error": "No valid pages specified",
                    "extraction_time": round(time.time() - start_time, 2)
                }

            svg_files = []
            total_size = 0
            base_name = input_pdf_path.stem

            for page_num in pages_to_process:
                try:
                    page = doc[page_num]
                    page_results = {}

                    # Full page SVG extraction
                    if mode in ["full_page", "both"]:
                        svg_content = page.get_svg_image(
                            text_as_path=not include_text
                        )

                        # Optionally simplify paths (basic implementation)
                        if simplify_paths:
                            svg_content = self._simplify_svg_paths(svg_content)

                        filename = f"{base_name}_page_{page_num + 1}.svg"
                        output_path = output_dir / filename

                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(svg_content)

                        file_size = output_path.stat().st_size
                        total_size += file_size

                        page_results["full_page"] = {
                            "filename": filename,
                            "path": str(output_path),
                            "size_bytes": file_size,
                            "size_kb": round(file_size / 1024, 1)
                        }

                    # Individual drawings extraction
                    if mode in ["drawings_only", "both"]:
                        drawings = page.get_drawings()
                        drawing_count = len(drawings)

                        if drawing_count > 0:
                            # Convert drawings to SVG
                            drawings_svg = self._drawings_to_svg(
                                drawings,
                                page.rect.width,
                                page.rect.height
                            )

                            filename = f"{base_name}_page_{page_num + 1}_drawings.svg"
                            output_path = output_dir / filename

                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(drawings_svg)

                            file_size = output_path.stat().st_size
                            total_size += file_size

                            page_results["drawings_only"] = {
                                "filename": filename,
                                "path": str(output_path),
                                "size_bytes": file_size,
                                "size_kb": round(file_size / 1024, 1),
                                "drawing_count": drawing_count
                            }
                        else:
                            page_results["drawings_only"] = {
                                "skipped": True,
                                "reason": "No vector drawings found on page"
                            }

                    # Get drawing statistics for the page
                    all_drawings = page.get_drawings()

                    svg_files.append({
                        "page": page_num + 1,
                        "has_text": bool(page.get_text().strip()),
                        "drawing_count": len(all_drawings),
                        **page_results
                    })

                except Exception as e:
                    logger.warning(f"Failed to extract vectors from page {page_num + 1}: {e}")
                    svg_files.append({
                        "page": page_num + 1,
                        "error": sanitize_error_message(str(e))
                    })

            doc.close()

            # Count successful extractions
            successful_pages = sum(1 for f in svg_files if "error" not in f)

            return {
                "success": True,
                "extraction_summary": {
                    "pages_processed": len(pages_to_process),
                    "pages_successful": successful_pages,
                    "mode": mode,
                    "total_size_bytes": total_size,
                    "total_size_kb": round(total_size / 1024, 1),
                    "output_directory": str(output_dir)
                },
                "svg_files": svg_files,
                "settings": {
                    "include_text": include_text,
                    "simplify_paths": simplify_paths,
                    "mode": mode
                },
                "file_info": {
                    "input_path": str(input_pdf_path),
                    "total_pages": total_pages,
                    "pages_processed": pages or "all"
                },
                "extraction_time": round(time.time() - start_time, 2),
                "hints": {
                    "viewing": "Open SVG files in browser, Inkscape, or Illustrator for editing",
                    "full_page_vs_drawings": "full_page preserves layout; drawings_only extracts raw vector paths"
                }
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Vector graphics extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "extraction_time": round(time.time() - start_time, 2)
            }

    def _drawings_to_svg(
        self,
        drawings: List[Dict],
        width: float,
        height: float
    ) -> str:
        """
        Convert PyMuPDF drawings to standalone SVG.

        Drawings contain: rect, items (path operations), color, fill, width, etc.
        """
        svg_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" ',
            f'viewBox="0 0 {width:.2f} {height:.2f}" ',
            f'width="{width:.2f}" height="{height:.2f}">',
            '',
            '  <!-- Extracted vector drawings from PDF -->',
            ''
        ]

        for idx, drawing in enumerate(drawings):
            try:
                path_data = self._drawing_to_path(drawing)
                if not path_data:
                    continue

                # Extract style attributes
                stroke_color = self._color_to_svg(drawing.get('color'))
                fill_color = self._color_to_svg(drawing.get('fill'))
                stroke_width = drawing.get('width', 1)

                # Build style string
                style_parts = []
                if fill_color:
                    style_parts.append(f'fill:{fill_color}')
                else:
                    style_parts.append('fill:none')

                if stroke_color:
                    style_parts.append(f'stroke:{stroke_color}')
                    style_parts.append(f'stroke-width:{stroke_width:.2f}')

                style = ';'.join(style_parts)

                svg_parts.append(f'  <path d="{path_data}" style="{style}" />')

            except Exception as e:
                logger.debug(f"Failed to convert drawing {idx}: {e}")
                continue

        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)

    def _drawing_to_path(self, drawing: Dict) -> Optional[str]:
        """Convert a single drawing to SVG path data string."""
        items = drawing.get('items', [])
        if not items:
            return None

        path_parts = []

        for item in items:
            if not item:
                continue

            # Item format: (type, points...)
            item_type = item[0]

            try:
                if item_type == 'l':  # Line
                    # ('l', Point, Point)
                    p1, p2 = item[1], item[2]
                    path_parts.append(f'M {p1.x:.2f} {p1.y:.2f}')
                    path_parts.append(f'L {p2.x:.2f} {p2.y:.2f}')

                elif item_type == 're':  # Rectangle
                    # ('re', Rect)
                    rect = item[1]
                    path_parts.append(f'M {rect.x0:.2f} {rect.y0:.2f}')
                    path_parts.append(f'L {rect.x1:.2f} {rect.y0:.2f}')
                    path_parts.append(f'L {rect.x1:.2f} {rect.y1:.2f}')
                    path_parts.append(f'L {rect.x0:.2f} {rect.y1:.2f}')
                    path_parts.append('Z')

                elif item_type == 'qu':  # Quad (4-point polygon)
                    # ('qu', Quad)
                    quad = item[1]
                    path_parts.append(f'M {quad.ul.x:.2f} {quad.ul.y:.2f}')
                    path_parts.append(f'L {quad.ur.x:.2f} {quad.ur.y:.2f}')
                    path_parts.append(f'L {quad.lr.x:.2f} {quad.lr.y:.2f}')
                    path_parts.append(f'L {quad.ll.x:.2f} {quad.ll.y:.2f}')
                    path_parts.append('Z')

                elif item_type == 'c':  # Cubic bezier curve
                    # ('c', Point, Point, Point, Point) - start, ctrl1, ctrl2, end
                    p0, p1, p2, p3 = item[1], item[2], item[3], item[4]
                    if not path_parts or not path_parts[-1].startswith('M'):
                        path_parts.append(f'M {p0.x:.2f} {p0.y:.2f}')
                    path_parts.append(f'C {p1.x:.2f} {p1.y:.2f} {p2.x:.2f} {p2.y:.2f} {p3.x:.2f} {p3.y:.2f}')

            except (IndexError, AttributeError) as e:
                logger.debug(f"Failed to process drawing item {item_type}: {e}")
                continue

        return ' '.join(path_parts) if path_parts else None

    def _color_to_svg(self, color) -> Optional[str]:
        """Convert PyMuPDF color to SVG color string."""
        if color is None:
            return None

        if isinstance(color, (list, tuple)):
            if len(color) == 3:
                r, g, b = [int(c * 255) for c in color]
                return f'rgb({r},{g},{b})'
            elif len(color) == 1:
                # Grayscale
                gray = int(color[0] * 255)
                return f'rgb({gray},{gray},{gray})'
            elif len(color) == 4:
                # CMYK - convert to RGB (simplified)
                c, m, y, k = color
                r = int(255 * (1 - c) * (1 - k))
                g = int(255 * (1 - m) * (1 - k))
                b = int(255 * (1 - y) * (1 - k))
                return f'rgb({r},{g},{b})'

        return None

    def _simplify_svg_paths(self, svg_content: str) -> str:
        """
        Basic SVG path simplification.
        Reduces decimal precision to shrink file size.
        """
        import re

        # Reduce decimal precision in path data
        def reduce_precision(match):
            num = float(match.group())
            return f'{num:.1f}'

        # Match floating point numbers in SVG
        simplified = re.sub(r'-?\d+\.\d{3,}', reduce_precision, svg_content)

        return simplified

    @mcp_tool(
        name="markdown_to_pdf",
        description=(
            "Convert a Markdown file (or inline text) to PDF using pandoc. "
            "Auto-detects available PDF engines (xelatex, pdflatex, tectonic, "
            "weasyprint, wkhtmltopdf) and falls back through them in that order. "
            "Pass pdf_engine to override, or extra_args for custom pandoc options "
            "(e.g. ['-V', 'geometry:margin=1in']). Requires pandoc binary on host "
            "and at least one PDF engine. Install with: pip install mcp-pdf[markdown]"
        ),
        annotations={
            "readOnlyHint": False,       # writes the PDF at output_path
            "destructiveHint": True,     # overwrites output_path if it exists
            "idempotentHint": True,      # same markdown + engine produce the same document
            "openWorldHint": True,       # shells out to pandoc and an external PDF engine
        },
    )
    async def markdown_to_pdf(
        self,
        output_path: str,
        markdown_path: Optional[str] = None,
        markdown_text: Optional[str] = None,
        pdf_engine: Optional[
            Literal["xelatex", "pdflatex", "tectonic", "weasyprint", "wkhtmltopdf"]
        ] = None,
        toc: bool = False,
        title: Optional[str] = None,
        author: Optional[str] = None,
        date: Optional[str] = None,
        base_path: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Convert Markdown to PDF using pandoc as the parser and one of several
        available PDF rendering engines as the backend.

        Provide either `markdown_path` (a .md file) or `markdown_text` (an
        inline string), but not both. The output PDF is written to `output_path`.

        Engine selection:
            If `pdf_engine` is None, the first engine found on PATH is used,
            preferring quality: xelatex > pdflatex > tectonic > weasyprint > wkhtmltopdf.
            If a specific engine is requested but not on PATH, an error is returned
            listing what *is* available.

        Args:
            output_path: Where to write the resulting PDF (required).
            markdown_path: Path to a .md file. Mutually exclusive with markdown_text.
            markdown_text: Inline markdown content. Mutually exclusive with markdown_path.
            pdf_engine: Force a specific engine ("xelatex", "pdflatex", "tectonic",
                "weasyprint", "wkhtmltopdf"). None (default) auto-detects the
                first of those five found on PATH, in that preference order.
            toc: Generate a table of contents from headings.
            title: Document title (overrides any YAML frontmatter title).
            author: Document author (overrides any YAML frontmatter author).
            date: Document date string (overrides any YAML frontmatter date).
            base_path: Resource resolution base for relative image references.
                Defaults to the markdown file's directory when markdown_path is used.
            extra_args: Additional raw pandoc CLI arguments (advanced).

        Returns:
            Dict with output_path, file_size, file_size_kb, engine_used,
            detected_engines, toc and conversion_time. Unlike the other tools in
            this server there is NO "success" key: a failure returns a dict whose
            only substantive key is "error" (plus conversion_time, and
            detected_engines when a requested engine was missing). Branch on the
            presence of "error" / "output_path", not on "success".
        """
        import shutil

        start_time = time.time()

        ENGINE_PREFERENCE = ["xelatex", "pdflatex", "tectonic", "weasyprint", "wkhtmltopdf"]

        try:
            # Optional dep check — pypandoc is gated behind the [markdown] extra
            try:
                import pypandoc
            except ImportError:
                return {
                    "error": (
                        "pypandoc is not installed. Install with: "
                        "pip install mcp-pdf[markdown]   (also requires the pandoc "
                        "binary and a PDF engine on PATH)"
                    ),
                    "conversion_time": round(time.time() - start_time, 2),
                }

            # Verify pandoc binary is reachable — pypandoc raises OSError if missing
            try:
                pypandoc.get_pandoc_version()
            except OSError:
                return {
                    "error": (
                        "pandoc binary not found on PATH. Install pandoc: "
                        "https://pandoc.org/installing.html"
                    ),
                    "conversion_time": round(time.time() - start_time, 2),
                }

            # Validate input — exactly one of markdown_path or markdown_text
            if bool(markdown_path) == bool(markdown_text):
                return {
                    "error": "Provide exactly one of markdown_path or markdown_text",
                    "conversion_time": round(time.time() - start_time, 2),
                }

            # Validate output path
            output = validate_output_path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            # Detect available PDF engines on PATH
            available_engines = [e for e in ENGINE_PREFERENCE if shutil.which(e)]

            # Pick engine: explicit override or first available
            if pdf_engine:
                if not shutil.which(pdf_engine):
                    return {
                        "error": (
                            f"Requested PDF engine '{pdf_engine}' not found on PATH. "
                            f"Available engines: {available_engines or 'none'}"
                        ),
                        "detected_engines": available_engines,
                        "conversion_time": round(time.time() - start_time, 2),
                    }
                engine = pdf_engine
            else:
                if not available_engines:
                    return {
                        "error": (
                            "No PDF engine found on PATH. Install one of: "
                            + ", ".join(ENGINE_PREFERENCE)
                        ),
                        "conversion_time": round(time.time() - start_time, 2),
                    }
                engine = available_engines[0]

            # Build pandoc arguments
            args: List[str] = [f"--pdf-engine={engine}"]
            if toc:
                args.append("--toc")
            if title:
                args.extend(["-M", f"title={title}"])
            if author:
                args.extend(["-M", f"author={author}"])
            if date:
                args.extend(["-M", f"date={date}"])

            # Resource path for relative image refs — defaults to source dir
            if base_path:
                resource_dir = Path(base_path).resolve()
            elif markdown_path:
                resource_dir = Path(markdown_path).resolve().parent
            else:
                resource_dir = None

            if resource_dir:
                args.extend(["--resource-path", str(resource_dir)])

            if extra_args:
                args.extend(extra_args)

            # Convert — file path or inline text
            if markdown_path:
                source_path = Path(markdown_path).resolve()
                if not source_path.is_file():
                    return {
                        "error": f"Markdown file not found: {markdown_path}",
                        "conversion_time": round(time.time() - start_time, 2),
                    }
                pypandoc.convert_file(
                    str(source_path),
                    to="pdf",
                    outputfile=str(output),
                    extra_args=args,
                )
            else:
                pypandoc.convert_text(
                    markdown_text,
                    to="pdf",
                    format="md",
                    outputfile=str(output),
                    extra_args=args,
                )

            file_size = output.stat().st_size

            return {
                "output_path": str(output),
                "file_size": file_size,
                "file_size_kb": round(file_size / 1024, 2),
                "engine_used": engine,
                "detected_engines": available_engines,
                "toc": toc,
                "conversion_time": round(time.time() - start_time, 2),
            }

        except RuntimeError as e:
            # pypandoc raises RuntimeError for pandoc subprocess failures —
            # the message often contains the engine's stderr, which is the most
            # useful signal a user can get for typesetting errors
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Markdown to PDF conversion failed: {error_msg}")
            return {
                "error": f"Pandoc conversion failed: {error_msg}",
                "conversion_time": round(time.time() - start_time, 2),
            }
        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Markdown to PDF failed: {error_msg}")
            return {
                "error": error_msg,
                "conversion_time": round(time.time() - start_time, 2),
            }