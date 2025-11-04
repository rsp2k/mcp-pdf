"""
PDF Utilities Mixin - Additional PDF processing tools
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# PDF processing libraries
import fitz  # PyMuPDF
from PIL import Image
import io

# Official FastMCP mixin
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, validate_output_path, sanitize_error_message
from .utils import parse_pages_parameter

logger = logging.getLogger(__name__)


class PDFUtilitiesMixin(MCPMixin):
    """
    Handles additional PDF utility operations including comparison, optimization, and repair.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()
        self.max_file_size = 100 * 1024 * 1024  # 100MB

    @mcp_tool(
        name="compare_pdfs",
        description="Compare two PDFs for differences in text, structure, and metadata"
    )
    async def compare_pdfs(
        self,
        pdf_path1: str,
        pdf_path2: str,
        comparison_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Compare two PDF files for differences.

        Args:
            pdf_path1: Path to first PDF file
            pdf_path2: Path to second PDF file
            comparison_type: Type of comparison ("text", "structure", "metadata", "all")

        Returns:
            Dictionary containing comparison results
        """
        start_time = time.time()

        try:
            # Validate both PDF paths
            path1 = await validate_pdf_path(pdf_path1)
            path2 = await validate_pdf_path(pdf_path2)

            doc1 = fitz.open(str(path1))
            doc2 = fitz.open(str(path2))

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
        description="Optimize PDF file size and performance"
    )
    async def optimize_pdf(
        self,
        pdf_path: str,
        optimization_level: str = "balanced",
        preserve_quality: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize PDF file for smaller size and better performance.

        Args:
            pdf_path: Path to PDF file to optimize
            optimization_level: Level of optimization ("light", "balanced", "aggressive")
            preserve_quality: Whether to preserve visual quality

        Returns:
            Dictionary containing optimization results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)

            # Generate optimized filename
            optimized_path = path.parent / f"{path.stem}_optimized.pdf"

            doc = fitz.open(str(path))
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
        description="Attempt to repair corrupted or damaged PDF files"
    )
    async def repair_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Attempt to repair a corrupted or damaged PDF file.

        Args:
            pdf_path: Path to PDF file to repair

        Returns:
            Dictionary containing repair results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)

            # Generate repaired filename
            repaired_path = path.parent / f"{path.stem}_repaired.pdf"

            # Attempt to open and repair the PDF
            try:
                doc = fitz.open(str(path))

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
                    except Exception as e:
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
        description="Rotate specific pages by 90, 180, or 270 degrees"
    )
    async def rotate_pages(
        self,
        pdf_path: str,
        rotation: int = 90,
        pages: Optional[str] = None,
        output_filename: str = "rotated_document.pdf"
    ) -> Dict[str, Any]:
        """
        Rotate specific pages in a PDF document.

        Args:
            pdf_path: Path to input PDF file
            rotation: Rotation angle (90, 180, 270 degrees)
            pages: Page numbers to rotate (comma-separated, 1-based), None for all
            output_filename: Name for the output file

        Returns:
            Dictionary containing rotation results
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

            doc = fitz.open(str(path))
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
        description="Convert PDF pages to image files"
    )
    async def convert_to_images(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        dpi: int = 300,
        format: str = "png",
        output_prefix: str = "page"
    ) -> Dict[str, Any]:
        """
        Convert PDF pages to image files.

        Args:
            pdf_path: Path to PDF file
            pages: Page numbers to convert (comma-separated, 1-based), None for all
            dpi: DPI for image rendering
            format: Output image format ("png", "jpg", "jpeg")
            output_prefix: Prefix for output image files

        Returns:
            Dictionary containing conversion results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))
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
                    mat = fitz.Matrix(dpi/72, dpi/72)
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

            return {
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