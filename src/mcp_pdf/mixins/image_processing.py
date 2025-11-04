"""
Image Processing Mixin - PDF image extraction and conversion capabilities
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# PDF processing libraries
import fitz  # PyMuPDF

from .base import MCPMixin, mcp_tool
from ..security import validate_pdf_path, parse_pages_parameter, validate_output_path, sanitize_error_message

logger = logging.getLogger(__name__)

# Cache directory for temporary files
CACHE_DIR = Path(os.environ.get("PDF_TEMP_DIR", "/tmp/mcp-pdf-processing"))
CACHE_DIR.mkdir(exist_ok=True, parents=True, mode=0o700)


class ImageProcessingMixin(MCPMixin):
    """
    Handles all PDF image extraction and conversion operations.

    Tools provided:
    - extract_images: Extract images from PDF with custom output path
    - pdf_to_markdown: Convert PDF to markdown with MCP resource URIs
    """

    def get_mixin_name(self) -> str:
        return "ImageProcessing"

    def get_required_permissions(self) -> List[str]:
        return ["read_files", "write_files", "image_processing"]

    def _setup(self):
        """Initialize image processing specific configuration"""
        self.default_output_format = "png"
        self.min_image_size = 100

    @mcp_tool(
        name="extract_images",
        description="Extract images from PDF with custom output path and clean summary"
    )
    async def extract_images(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        min_width: int = 100,
        min_height: int = 100,
        output_format: str = "png",
        output_directory: Optional[str] = None,
        include_context: bool = True,
        context_chars: int = 200
    ) -> Dict[str, Any]:
        """
        Extract images from PDF with positioning context for text-image coordination.

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
            # Validate inputs using centralized security functions
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

                                # Generate filename
                                base_name = Path(pdf_path).stem
                                filename = f"{base_name}_page{page_num + 1}_img{img_index + 1}.{output_format}"
                                filepath = output_dir / filename

                                # Save image
                                if output_format.lower() == "png":
                                    pix.save(str(filepath))
                                else:
                                    pix.save(str(filepath), output=output_format.upper())

                                file_size = filepath.stat().st_size
                                total_size += file_size

                                image_info = {
                                    "filename": filename,
                                    "filepath": str(filepath),
                                    "page": page_num + 1,  # 1-based for user
                                    "index": img_index + 1,
                                    "width": pix.width,
                                    "height": pix.height,
                                    "size_bytes": file_size,
                                    "format": output_format.upper()
                                }

                                # Add context if requested
                                if include_context and page_text:
                                    # Simple context extraction around image position
                                    context_start = max(0, len(page_text) // 2 - context_chars // 2)
                                    context_end = min(len(page_text), context_start + context_chars)
                                    image_info["context"] = page_text[context_start:context_end].strip()

                                page_images.append(image_info)
                                extracted_files.append(image_info)

                        pix = None  # Free memory

                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue

                if page_images:
                    pages_with_images.append({
                        "page": page_num + 1,
                        "image_count": len(page_images),
                        "images": page_images
                    })

            doc.close()

            # Format file size for display
            def format_size(size_bytes):
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if size_bytes < 1024.0:
                        return f"{size_bytes:.1f} {unit}"
                    size_bytes /= 1024.0
                return f"{size_bytes:.1f} TB"

            return {
                "success": True,
                "images_extracted": len(extracted_files),
                "pages_with_images": [p["page"] for p in pages_with_images],
                "total_size": format_size(total_size),
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
                    "positioning_data": False  # Could be enhanced in future
                },
                "extracted_images": extracted_files
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Image extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "images_extracted": 0,
                "pages_with_images": [],
                "output_directory": str(output_directory) if output_directory else str(CACHE_DIR)
            }

    @mcp_tool(
        name="pdf_to_markdown",
        description="Convert PDF to markdown with MCP resource URIs for images"
    )
    async def pdf_to_markdown(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        include_images: bool = True,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Convert PDF to markdown format with MCP resource URIs for images.

        Args:
            pdf_path: Path to PDF file or URL
            pages: Specific pages to convert (e.g., "1-5,10" or "all")
            include_images: Whether to include image references
            include_metadata: Whether to include document metadata

        Returns:
            Markdown content with MCP resource URIs for images
        """
        try:
            path = await validate_pdf_path(pdf_path)
            parsed_pages = parse_pages_parameter(pages)
            doc = fitz.open(str(path))

            markdown_parts = []

            # Add metadata if requested
            if include_metadata:
                metadata = doc.metadata
                if metadata.get("title"):
                    markdown_parts.append(f"# {metadata['title']}")
                if metadata.get("author"):
                    markdown_parts.append(f"*Author: {metadata['author']}*")
                if metadata.get("subject"):
                    markdown_parts.append(f"*Subject: {metadata['subject']}*")
                markdown_parts.append("")  # Empty line

            page_range = parsed_pages if parsed_pages else range(len(doc))

            for page_num in page_range:
                page = doc[page_num]

                # Add page header
                markdown_parts.append(f"## Page {page_num + 1}")
                markdown_parts.append("")

                # Extract text
                text = page.get_text()
                if text.strip():
                    # Basic text formatting
                    lines = text.split('\n')
                    formatted_lines = []
                    for line in lines:
                        line = line.strip()
                        if line:
                            formatted_lines.append(line)

                    markdown_parts.append('\n'.join(formatted_lines))
                    markdown_parts.append("")

                # Add image references if requested
                if include_images:
                    image_list = page.get_images()
                    if image_list:
                        markdown_parts.append("### Images")
                        for img_index, img in enumerate(image_list):
                            # Create MCP resource URI for image
                            image_id = f"page{page_num + 1}_img{img_index + 1}"
                            markdown_parts.append(f"![Image {img_index + 1}](pdf-image://{image_id})")
                        markdown_parts.append("")

            doc.close()

            markdown_content = '\n'.join(markdown_parts)

            return {
                "success": True,
                "markdown": markdown_content,
                "pages_processed": len(page_range),
                "total_pages": len(doc),
                "include_images": include_images,
                "include_metadata": include_metadata,
                "character_count": len(markdown_content),
                "line_count": len(markdown_parts)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"PDF to markdown conversion failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "markdown": "",
                "pages_processed": 0
            }