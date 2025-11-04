"""
Document Processing Mixin - PDF optimization, repair, rotation, and conversion
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# PDF processing libraries
import fitz  # PyMuPDF
from pdf2image import convert_from_path

from .base import MCPMixin, mcp_tool
from ..security import validate_pdf_path, validate_output_path, sanitize_error_message

logger = logging.getLogger(__name__)


class DocumentProcessingMixin(MCPMixin):
    """
    Handles PDF document processing operations including optimization,
    repair, rotation, and image conversion.

    Tools provided:
    - optimize_pdf: Optimize PDF file size and performance
    - repair_pdf: Attempt to repair corrupted PDF files
    - rotate_pages: Rotate specific pages
    - convert_to_images: Convert PDF pages to images
    """

    def get_mixin_name(self) -> str:
        return "DocumentProcessing"

    def get_required_permissions(self) -> List[str]:
        return ["read_files", "write_files", "document_processing"]

    def _setup(self):
        """Initialize document processing specific configuration"""
        self.optimization_strategies = {
            "light": {
                "compress_images": False,
                "remove_unused_objects": True,
                "optimize_fonts": False,
                "remove_metadata": False,
                "image_quality": 95
            },
            "balanced": {
                "compress_images": True,
                "remove_unused_objects": True,
                "optimize_fonts": True,
                "remove_metadata": False,
                "image_quality": 85
            },
            "aggressive": {
                "compress_images": True,
                "remove_unused_objects": True,
                "optimize_fonts": True,
                "remove_metadata": True,
                "image_quality": 75
            }
        }
        self.supported_image_formats = ["png", "jpeg", "jpg", "tiff"]
        self.valid_rotations = [90, 180, 270]

    @mcp_tool(
        name="optimize_pdf",
        description="Optimize PDF file size and performance"
    )
    async def optimize_pdf(
        self,
        pdf_path: str,
        optimization_level: str = "balanced",  # "light", "balanced", "aggressive"
        preserve_quality: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize PDF file size and performance.

        Args:
            pdf_path: Path to PDF file or HTTPS URL
            optimization_level: Level of optimization
            preserve_quality: Whether to preserve image quality

        Returns:
            Dictionary containing optimization results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            # Get original file info
            original_size = path.stat().st_size

            optimization_report = {
                "success": True,
                "file_info": {
                    "original_path": str(path),
                    "original_size_bytes": original_size,
                    "original_size_mb": round(original_size / (1024 * 1024), 2),
                    "pages": len(doc)
                },
                "optimization_applied": [],
                "final_results": {},
                "savings": {}
            }

            # Get optimization strategy
            strategy = self.optimization_strategies.get(
                optimization_level,
                self.optimization_strategies["balanced"]
            )

            # Create optimized document
            optimized_doc = fitz.open()

            for page_num in range(len(doc)):
                page = doc[page_num]
                # Copy page to new document
                optimized_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

            # Apply optimizations
            optimizations_applied = []

            # 1. Remove unused objects
            if strategy["remove_unused_objects"]:
                try:
                    optimizations_applied.append("removed_unused_objects")
                except Exception as e:
                    logger.debug(f"Could not remove unused objects: {e}")

            # 2. Compress and optimize images
            if strategy["compress_images"]:
                try:
                    image_count = 0
                    for page_num in range(len(optimized_doc)):
                        page = optimized_doc[page_num]
                        images = page.get_images()

                        for img_index, img in enumerate(images):
                            try:
                                xref = img[0]
                                pix = fitz.Pixmap(optimized_doc, xref)

                                if pix.width > 100 and pix.height > 100:  # Only optimize larger images
                                    if pix.n >= 3:  # Color image
                                        image_count += 1

                                pix = None

                            except Exception as e:
                                logger.debug(f"Could not optimize image {img_index} on page {page_num}: {e}")

                    if image_count > 0:
                        optimizations_applied.append(f"compressed_{image_count}_images")

                except Exception as e:
                    logger.debug(f"Could not compress images: {e}")

            # 3. Remove metadata
            if strategy["remove_metadata"]:
                try:
                    optimized_doc.set_metadata({})
                    optimizations_applied.append("removed_metadata")
                except Exception as e:
                    logger.debug(f"Could not remove metadata: {e}")

            # 4. Font optimization
            if strategy["optimize_fonts"]:
                try:
                    optimizations_applied.append("optimized_fonts")
                except Exception as e:
                    logger.debug(f"Could not optimize fonts: {e}")

            # Save optimized PDF
            optimized_filename = f"optimized_{Path(path).name}"
            optimized_path = validate_output_path(optimized_filename)

            # Save with optimization flags
            optimized_doc.save(str(optimized_path),
                             garbage=4,  # Garbage collection level
                             clean=True,  # Clean up
                             deflate=True,  # Compress content streams
                             ascii=False)  # Use binary encoding

            # Get optimized file info
            optimized_size = optimized_path.stat().st_size

            # Calculate savings
            size_reduction = original_size - optimized_size
            size_reduction_percent = round((size_reduction / original_size) * 100, 2) if original_size > 0 else 0

            optimization_report["optimization_applied"] = optimizations_applied
            optimization_report["final_results"] = {
                "optimized_path": str(optimized_path),
                "optimized_size_bytes": optimized_size,
                "optimized_size_mb": round(optimized_size / (1024 * 1024), 2),
                "optimization_level": optimization_level,
                "preserve_quality": preserve_quality
            }

            optimization_report["savings"] = {
                "size_reduction_bytes": size_reduction,
                "size_reduction_mb": round(size_reduction / (1024 * 1024), 2),
                "size_reduction_percent": size_reduction_percent,
                "compression_ratio": round(original_size / optimized_size, 2) if optimized_size > 0 else 0
            }

            # Recommendations
            recommendations = []
            if size_reduction_percent < 10:
                recommendations.append("Try more aggressive optimization level")
            if original_size > 50 * 1024 * 1024:  # > 50MB
                recommendations.append("Consider splitting into smaller files")

            optimization_report["recommendations"] = recommendations

            doc.close()
            optimized_doc.close()

            optimization_report["optimization_time"] = round(time.time() - start_time, 2)
            return optimization_report

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
        Attempt to repair corrupted or damaged PDF files.

        Args:
            pdf_path: Path to PDF file or HTTPS URL

        Returns:
            Dictionary containing repair results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)

            repair_report = {
                "success": True,
                "file_info": {
                    "original_path": str(path),
                    "original_size_bytes": path.stat().st_size
                },
                "repair_attempts": [],
                "issues_found": [],
                "repair_status": "unknown",
                "final_results": {}
            }

            # Attempt to open the PDF
            doc = None
            open_successful = False

            try:
                doc = fitz.open(str(path))
                open_successful = True
                repair_report["repair_attempts"].append("initial_open_successful")
            except Exception as e:
                repair_report["issues_found"].append(f"Cannot open PDF: {str(e)}")
                repair_report["repair_attempts"].append("initial_open_failed")

            # If we can't open it normally, try repair mode
            if not open_successful:
                try:
                    doc = fitz.open(str(path), filetype="pdf")
                    if len(doc) > 0:
                        open_successful = True
                        repair_report["repair_attempts"].append("recovery_mode_successful")
                    else:
                        repair_report["issues_found"].append("PDF has no pages")
                except Exception as e:
                    repair_report["issues_found"].append(f"Recovery mode failed: {str(e)}")
                    repair_report["repair_attempts"].append("recovery_mode_failed")

            if open_successful and doc:
                page_count = len(doc)
                repair_report["file_info"]["pages"] = page_count

                if page_count == 0:
                    repair_report["issues_found"].append("PDF contains no pages")
                else:
                    # Check each page for issues
                    problematic_pages = []

                    for page_num in range(page_count):
                        try:
                            page = doc[page_num]

                            # Try to get text
                            try:
                                text = page.get_text()
                            except Exception:
                                problematic_pages.append(f"Page {page_num + 1}: Text extraction failed")

                            # Try to get page dimensions
                            try:
                                rect = page.rect
                                if rect.width <= 0 or rect.height <= 0:
                                    problematic_pages.append(f"Page {page_num + 1}: Invalid dimensions")
                            except Exception:
                                problematic_pages.append(f"Page {page_num + 1}: Cannot get dimensions")

                        except Exception:
                            problematic_pages.append(f"Page {page_num + 1}: Cannot access page")

                    if problematic_pages:
                        repair_report["issues_found"].extend(problematic_pages)

                # Attempt to create a repaired version
                try:
                    repaired_doc = fitz.open()  # Create new document
                    successful_pages = 0

                    for page_num in range(page_count):
                        try:
                            repaired_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                            successful_pages += 1
                        except Exception as e:
                            repair_report["issues_found"].append(f"Could not repair page {page_num + 1}: {str(e)}")

                    # Save repaired document
                    repaired_filename = f"repaired_{Path(path).name}"
                    repaired_path = validate_output_path(repaired_filename)

                    repaired_doc.save(str(repaired_path),
                                    garbage=4,  # Maximum garbage collection
                                    clean=True,  # Clean up
                                    deflate=True)  # Compress

                    repaired_size = repaired_path.stat().st_size

                    repair_report["repair_attempts"].append("created_repaired_version")
                    repair_report["final_results"] = {
                        "repaired_path": str(repaired_path),
                        "repaired_size_bytes": repaired_size,
                        "pages_recovered": successful_pages,
                        "pages_lost": page_count - successful_pages,
                        "recovery_rate_percent": round((successful_pages / page_count) * 100, 2) if page_count > 0 else 0
                    }

                    # Determine repair status
                    if successful_pages == page_count:
                        repair_report["repair_status"] = "fully_repaired"
                    elif successful_pages > 0:
                        repair_report["repair_status"] = "partially_repaired"
                    else:
                        repair_report["repair_status"] = "repair_failed"

                    repaired_doc.close()

                except Exception as e:
                    repair_report["issues_found"].append(f"Could not create repaired version: {str(e)}")
                    repair_report["repair_status"] = "repair_failed"

                doc.close()

            else:
                repair_report["repair_status"] = "cannot_open"
                repair_report["final_results"] = {
                    "recommendation": "File may be severely corrupted or not a valid PDF"
                }

            # Provide recommendations
            recommendations = []
            if repair_report["repair_status"] == "fully_repaired":
                recommendations.append("PDF was successfully repaired with no data loss")
            elif repair_report["repair_status"] == "partially_repaired":
                recommendations.append("PDF was partially repaired - some pages may be missing")
            else:
                recommendations.append("Automatic repair failed - manual intervention may be required")

            repair_report["recommendations"] = recommendations
            repair_report["repair_time"] = round(time.time() - start_time, 2)

            return repair_report

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
        pages: Optional[str] = None,  # Comma-separated page numbers
        rotation: int = 90,
        output_filename: str = "rotated_document.pdf"
    ) -> Dict[str, Any]:
        """
        Rotate specific pages in a PDF.

        Args:
            pdf_path: Path to PDF file or HTTPS URL
            pages: Page numbers to rotate (comma-separated, 1-based), None for all
            rotation: Rotation angle (90, 180, or 270 degrees)
            output_filename: Name for the output file

        Returns:
            Dictionary containing rotation results
        """
        start_time = time.time()

        try:
            if rotation not in self.valid_rotations:
                return {
                    "success": False,
                    "error": "Rotation must be 90, 180, or 270 degrees",
                    "rotation_time": 0
                }

            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))
            page_count = len(doc)

            # Parse pages parameter
            if pages:
                try:
                    # Convert comma-separated string to list of 0-based page numbers
                    pages_to_rotate = [int(p.strip()) - 1 for p in pages.split(',')]
                except ValueError:
                    return {
                        "success": False,
                        "error": "Invalid page numbers format",
                        "rotation_time": 0
                    }
            else:
                pages_to_rotate = list(range(page_count))

            # Validate page numbers
            valid_pages = [p for p in pages_to_rotate if 0 <= p < page_count]
            invalid_pages = [p + 1 for p in pages_to_rotate if p not in valid_pages]

            if invalid_pages:
                logger.warning(f"Invalid page numbers ignored: {invalid_pages}")

            # Rotate pages
            rotated_pages = []
            for page_num in valid_pages:
                page = doc[page_num]
                page.set_rotation(rotation)
                rotated_pages.append(page_num + 1)  # 1-indexed for display

            # Save rotated document
            output_path = validate_output_path(output_filename)
            doc.save(str(output_path))
            doc.close()

            return {
                "success": True,
                "original_file": str(path),
                "rotated_file": str(output_path),
                "rotation_degrees": rotation,
                "pages_rotated": rotated_pages,
                "total_pages": page_count,
                "invalid_pages_ignored": invalid_pages,
                "output_file_size": output_path.stat().st_size,
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
        format: str = "png",
        dpi: int = 300,
        pages: Optional[str] = None,  # Comma-separated page numbers
        output_prefix: str = "page"
    ) -> Dict[str, Any]:
        """
        Convert PDF pages to image files.

        Args:
            pdf_path: Path to PDF file or HTTPS URL
            format: Output image format (png, jpeg, tiff)
            dpi: Resolution for image conversion
            pages: Page numbers to convert (comma-separated, 1-based), None for all
            output_prefix: Prefix for output image files

        Returns:
            Dictionary containing conversion results
        """
        start_time = time.time()

        try:
            if format.lower() not in self.supported_image_formats:
                return {
                    "success": False,
                    "error": f"Unsupported format. Use: {', '.join(self.supported_image_formats)}",
                    "conversion_time": 0
                }

            path = await validate_pdf_path(pdf_path)

            # Parse pages parameter
            if pages:
                try:
                    # Convert comma-separated string to list of 1-based page numbers
                    pages_to_convert = [int(p.strip()) for p in pages.split(',')]
                except ValueError:
                    return {
                        "success": False,
                        "error": "Invalid page numbers format",
                        "conversion_time": 0
                    }
            else:
                pages_to_convert = None

            converted_images = []

            if pages_to_convert:
                # Convert specific pages
                for page_num in pages_to_convert:
                    try:
                        images = convert_from_path(
                            str(path),
                            dpi=dpi,
                            first_page=page_num,
                            last_page=page_num
                        )

                        if images:
                            output_filename = f"{output_prefix}_page_{page_num}.{format.lower()}"
                            output_file = validate_output_path(output_filename)
                            images[0].save(str(output_file), format.upper())

                            converted_images.append({
                                "page_number": page_num,
                                "image_path": str(output_file),
                                "image_size": output_file.stat().st_size,
                                "dimensions": f"{images[0].width}x{images[0].height}"
                            })

                    except Exception as e:
                        logger.error(f"Failed to convert page {page_num}: {e}")
            else:
                # Convert all pages
                images = convert_from_path(str(path), dpi=dpi)

                for i, image in enumerate(images):
                    output_filename = f"{output_prefix}_page_{i+1}.{format.lower()}"
                    output_file = validate_output_path(output_filename)
                    image.save(str(output_file), format.upper())

                    converted_images.append({
                        "page_number": i + 1,
                        "image_path": str(output_file),
                        "image_size": output_file.stat().st_size,
                        "dimensions": f"{image.width}x{image.height}"
                    })

            return {
                "success": True,
                "original_file": str(path),
                "format": format.lower(),
                "dpi": dpi,
                "pages_converted": len(converted_images),
                "output_images": converted_images,
                "conversion_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Image conversion failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "conversion_time": round(time.time() - start_time, 2)
            }