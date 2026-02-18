"""
Image Processing Mixin - PDF image extraction and markdown conversion
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# PDF and image processing libraries
import fitz  # PyMuPDF

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
        self.max_file_size = 100 * 1024 * 1024  # 100MB

    @mcp_tool(
        name="extract_images",
        description="Extract images from PDF with custom output path"
    )
    async def extract_images(
        self,
        pdf_path: str,
        output_directory: Optional[str] = None,
        min_width: int = 100,
        min_height: int = 100,
        output_format: str = "png",
        pages: Optional[str] = None,
        include_context: bool = True,
        context_chars: int = 200
    ) -> Dict[str, Any]:
        """
        Extract images from PDF with custom output directory and clean summary.

        Args:
            pdf_path: Path to PDF file or HTTPS URL
            output_directory: Directory to save extracted images (default: temp directory)
            min_width: Minimum image width to extract
            min_height: Minimum image height to extract
            output_format: Output image format ("png", "jpg", "jpeg")
            pages: Page numbers to extract (comma-separated, 1-based), None for all
            include_context: Whether to include surrounding text context
            context_chars: Number of context characters around images

        Returns:
            Dictionary containing image extraction summary and paths
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
            doc = fitz.open(str(input_pdf_path))
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
                            pix = fitz.Pixmap(doc, xref)

                            # Check image dimensions
                            if pix.width < min_width or pix.height < min_height:
                                images_skipped += 1
                                pix = None
                                continue

                            # Convert CMYK to RGB if necessary
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                pass
                            else:  # CMYK: convert to RGB first
                                pix = fitz.Pixmap(fitz.csRGB, pix)

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
            "Convert PDF to markdown and write to a .md file. Images are extracted "
            "to {output_directory}/images/ with relative ./images/ paths. Returns "
            "the output file path and a short preview — full markdown is in the file. "
            "Set inline=True to get full markdown in the response instead."
        )
    )
    async def pdf_to_markdown(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        include_images: bool = True,
        include_metadata: bool = True,
        output_directory: Optional[str] = None,
        min_width: int = 100,
        min_height: int = 100,
        image_format: str = "png",
        inline: bool = False
    ) -> Dict[str, Any]:
        """
        Convert PDF to clean markdown format and write to file.

        By default, writes markdown to a file and extracts images to an images/
        subdirectory with relative paths. Returns file path + summary to avoid
        filling the MCP context window. Set inline=True for full markdown in response.

        Args:
            pdf_path: Path to PDF file or HTTPS URL
            pages: Page numbers to convert (comma-separated, 1-based), None for all
            include_images: Whether to include images in markdown
            include_metadata: Whether to include document metadata
            output_directory: Directory for output .md file and images/ subdirectory.
                Defaults to a temp directory if not specified.
            min_width: Minimum image width to extract (filters small decorative images)
            min_height: Minimum image height to extract (filters small decorative images)
            image_format: Image format - "png" or "jpg"
            inline: Return full markdown in response instead of writing to file

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
            doc = fitz.open(str(input_pdf_path))
            total_pages = len(doc)

            # Determine pages to process
            pages_to_process = parsed_pages if parsed_pages else list(range(total_pages))
            pages_to_process = [p for p in pages_to_process if 0 <= p < total_pages]

            # Setup output directory — always needed (file output is the default)
            images_extracted = 0
            images_skipped = 0
            extracted_image_info = []

            if output_directory:
                output_dir = validate_output_path(output_directory)
            else:
                output_dir = Path(tempfile.mkdtemp(prefix="pdf_markdown_"))
            output_dir.mkdir(parents=True, exist_ok=True)
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

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
                                pix = fitz.Pixmap(doc, xref)

                                if pix.width < min_width or pix.height < min_height:
                                    images_skipped += 1
                                    pix = None
                                    continue

                                # Convert CMYK to RGB if necessary
                                if pix.n - pix.alpha >= 4:
                                    pix = fitz.Pixmap(fitz.csRGB, pix)

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
                "images_skipped": images_skipped
            }

            # Inline mode: return full markdown in response
            if inline:
                return {
                    "success": True,
                    "markdown": full_markdown,
                    "conversion_summary": conversion_summary,
                    "image_output": {
                        "images_directory": str(images_dir),
                        "images": extracted_image_info
                    },
                    "file_info": {
                        "input_path": str(input_pdf_path),
                        "pages_processed": pages or "all"
                    },
                    "conversion_time": round(time.time() - start_time, 2)
                }

            # File output mode (default): write .md file, return path + summary
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

            return {
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
                        "image_format": image_format
                    },
                    "images": extracted_image_info
                },
                "file_info": {
                    "input_path": str(input_pdf_path),
                    "output_directory": str(output_dir),
                    "pages_processed": pages or "all"
                },
                "conversion_time": round(time.time() - start_time, 2)
            }

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

    @mcp_tool(
        name="extract_vector_graphics",
        description="Extract vector graphics from PDF to SVG format. Ideal for schematics, charts, and technical drawings."
    )
    async def extract_vector_graphics(
        self,
        pdf_path: str,
        output_directory: Optional[str] = None,
        pages: Optional[str] = None,
        mode: str = "full_page",
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
            pdf_path: Path to PDF file or HTTPS URL
            output_directory: Directory to save SVG files (default: temp directory)
            pages: Page numbers to extract (comma-separated, 1-based), None for all
            mode: Extraction mode:
                - "full_page": Complete page as SVG (default, best for general use)
                - "drawings_only": Extract individual vector paths as separate SVG
                - "both": Export both formats for flexibility
            include_text: Whether to include text in SVG output (default: True)
            simplify_paths: Reduce path complexity for smaller files (default: False)

        Returns:
            Dictionary containing extraction summary and SVG file paths
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
            doc = fitz.open(str(input_pdf_path))
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
            f'<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" ',
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