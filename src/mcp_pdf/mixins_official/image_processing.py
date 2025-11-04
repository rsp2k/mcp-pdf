"""
Image Processing Mixin - PDF image extraction and markdown conversion
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import asyncio
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# PDF and image processing libraries
import fitz  # PyMuPDF
from PIL import Image
import io
import base64

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
                output_dir = await validate_output_path(output_directory)
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
        description="Convert PDF to markdown with MCP resource URIs"
    )
    async def pdf_to_markdown(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        include_images: bool = True,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Convert PDF to clean markdown format with MCP resource URIs for images.

        Args:
            pdf_path: Path to PDF file or HTTPS URL
            pages: Page numbers to convert (comma-separated, 1-based), None for all
            include_images: Whether to include images in markdown
            include_metadata: Whether to include document metadata

        Returns:
            Dictionary containing markdown content and metadata
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
                                # Create MCP resource URI for the image
                                image_id = f"page_{page_num + 1}_img_{img_index + 1}"
                                mcp_uri = f"pdf-image://{image_id}"

                                # Add markdown image reference
                                alt_text = f"Image {img_index + 1} from page {page_num + 1}"
                                markdown_parts.append(f"![{alt_text}]({mcp_uri})\n\n")

                            except Exception as e:
                                logger.warning(f"Failed to process image {img_index + 1} on page {page_num + 1}: {e}")

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

            return {
                "success": True,
                "markdown": full_markdown,
                "conversion_summary": {
                    "pages_converted": len(pages_to_process),
                    "total_pages": total_pages,
                    "word_count": word_count,
                    "line_count": line_count,
                    "character_count": char_count,
                    "includes_images": include_images,
                    "includes_metadata": include_metadata
                },
                "mcp_integration": {
                    "image_uri_format": "pdf-image://{image_id}",
                    "description": "Images use MCP resource URIs for seamless client integration"
                },
                "file_info": {
                    "input_path": str(input_pdf_path),
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