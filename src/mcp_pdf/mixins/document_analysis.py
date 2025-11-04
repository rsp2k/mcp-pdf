"""
Document Analysis Mixin - PDF metadata extraction and structure analysis
"""

import time
from pathlib import Path
from typing import Dict, Any, List
import logging

# PDF processing libraries
import fitz  # PyMuPDF

from .base import MCPMixin, mcp_tool
from ..security import validate_pdf_path, sanitize_error_message

logger = logging.getLogger(__name__)


class DocumentAnalysisMixin(MCPMixin):
    """
    Handles all PDF document analysis and metadata operations.

    Tools provided:
    - extract_metadata: Comprehensive metadata extraction
    - get_document_structure: Document structure and outline analysis
    - analyze_pdf_health: PDF health and quality analysis
    """

    def get_mixin_name(self) -> str:
        return "DocumentAnalysis"

    def get_required_permissions(self) -> List[str]:
        return ["read_files", "metadata_access"]

    def _setup(self):
        """Initialize document analysis specific configuration"""
        self.max_pages_analyze = 100  # Limit for detailed analysis

    @mcp_tool(
        name="extract_metadata",
        description="Extract comprehensive PDF metadata"
    )
    async def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from PDF.

        Args:
            pdf_path: Path to PDF file or URL

        Returns:
            Dictionary containing all available metadata
        """
        try:
            # Validate inputs using centralized security functions
            path = await validate_pdf_path(pdf_path)

            # Get file stats
            file_stats = path.stat()

            # PyMuPDF metadata
            doc = fitz.open(str(path))
            fitz_metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": str(doc.metadata.get("creationDate", "")),
                "modification_date": str(doc.metadata.get("modDate", "")),
                "trapped": doc.metadata.get("trapped", ""),
            }

            # Document statistics
            has_annotations = False
            has_links = False

            try:
                for page in doc:
                    if hasattr(page, 'annots') and page.annots() is not None:
                        annots_list = list(page.annots())
                        if len(annots_list) > 0:
                            has_annotations = True
                            break
            except Exception:
                pass

            try:
                for page in doc:
                    if page.get_links():
                        has_links = True
                        break
            except Exception:
                pass

            # Additional document properties
            document_stats = {
                "page_count": len(doc),
                "file_size_bytes": file_stats.st_size,
                "file_size_mb": round(file_stats.st_size / 1024 / 1024, 2),
                "has_annotations": has_annotations,
                "has_links": has_links,
                "is_encrypted": doc.is_encrypted,
                "needs_password": doc.needs_pass,
                "pdf_version": getattr(doc, 'pdf_version', 'unknown'),
            }

            doc.close()

            return {
                "success": True,
                "metadata": fitz_metadata,
                "document_stats": document_stats,
                "file_info": {
                    "path": str(path),
                    "name": path.name,
                    "extension": path.suffix,
                    "created": file_stats.st_ctime,
                    "modified": file_stats.st_mtime,
                    "size_bytes": file_stats.st_size
                }
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Metadata extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    @mcp_tool(
        name="get_document_structure",
        description="Extract document structure including headers, sections, and metadata"
    )
    async def get_document_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract document structure including headers, sections, and metadata.

        Args:
            pdf_path: Path to PDF file or URL

        Returns:
            Dictionary containing document structure information
        """
        try:
            # Validate inputs using centralized security functions
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            structure = {
                "metadata": {
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "keywords": doc.metadata.get("keywords", ""),
                    "creator": doc.metadata.get("creator", ""),
                    "producer": doc.metadata.get("producer", ""),
                    "creation_date": str(doc.metadata.get("creationDate", "")),
                    "modification_date": str(doc.metadata.get("modDate", "")),
                },
                "pages": len(doc),
                "outline": []
            }

            # Extract table of contents / bookmarks
            toc = doc.get_toc()
            for level, title, page in toc:
                structure["outline"].append({
                    "level": level,
                    "title": title,
                    "page": page
                })

            # Extract page-level information (sample first few pages)
            page_info = []
            sample_pages = min(5, len(doc))

            for i in range(sample_pages):
                page = doc[i]
                page_data = {
                    "page_number": i + 1,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "rotation": page.rotation,
                    "text_length": len(page.get_text()),
                    "image_count": len(page.get_images()),
                    "link_count": len(page.get_links())
                }
                page_info.append(page_data)

            structure["page_samples"] = page_info
            structure["total_pages_analyzed"] = sample_pages

            doc.close()

            return {
                "success": True,
                "structure": structure
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Document structure extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    @mcp_tool(
        name="analyze_pdf_health",
        description="Comprehensive PDF health and quality analysis"
    )
    async def analyze_pdf_health(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze PDF health, quality, and potential issues.

        Args:
            pdf_path: Path to PDF file or URL

        Returns:
            Dictionary containing health analysis results
        """
        start_time = time.time()

        try:
            # Validate inputs using centralized security functions
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            health_report = {
                "file_info": {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "size_mb": round(path.stat().st_size / 1024 / 1024, 2)
                },
                "document_health": {},
                "quality_metrics": {},
                "optimization_suggestions": [],
                "warnings": [],
                "errors": []
            }

            # Basic document health
            page_count = len(doc)
            health_report["document_health"]["page_count"] = page_count
            health_report["document_health"]["is_valid"] = page_count > 0

            # Check for corruption by trying to access each page
            corrupted_pages = []
            total_text_length = 0
            total_images = 0

            for i, page in enumerate(doc):
                try:
                    text = page.get_text()
                    total_text_length += len(text)
                    total_images += len(page.get_images())
                except Exception as e:
                    corrupted_pages.append({"page": i + 1, "error": str(e)})

            health_report["document_health"]["corrupted_pages"] = corrupted_pages
            health_report["document_health"]["corruption_detected"] = len(corrupted_pages) > 0

            # Quality metrics
            health_report["quality_metrics"]["average_text_per_page"] = total_text_length / page_count if page_count > 0 else 0
            health_report["quality_metrics"]["total_images"] = total_images
            health_report["quality_metrics"]["images_per_page"] = total_images / page_count if page_count > 0 else 0

            # Font analysis
            fonts_used = set()
            embedded_fonts = 0

            for page in doc:
                try:
                    for font_info in page.get_fonts():
                        font_name = font_info[3]
                        fonts_used.add(font_name)
                        if font_info[1] != "n/a":  # Embedded font
                            embedded_fonts += 1
                except Exception:
                    pass

            health_report["quality_metrics"]["fonts_used"] = len(fonts_used)
            health_report["quality_metrics"]["fonts_list"] = list(fonts_used)
            health_report["quality_metrics"]["embedded_fonts"] = embedded_fonts

            # Security and protection
            health_report["document_health"]["is_encrypted"] = doc.is_encrypted
            health_report["document_health"]["needs_password"] = doc.needs_pass

            # Optimization suggestions
            file_size_mb = health_report["file_info"]["size_mb"]

            if file_size_mb > 10:
                health_report["optimization_suggestions"].append(
                    "Large file size detected. Consider optimizing images or using compression."
                )

            if total_images > page_count * 5:
                health_report["optimization_suggestions"].append(
                    "High image density detected. Consider image compression or resolution reduction."
                )

            if len(fonts_used) > 20:
                health_report["optimization_suggestions"].append(
                    f"Many fonts in use ({len(fonts_used)}). Consider font subset embedding to reduce file size."
                )

            if embedded_fonts < len(fonts_used) / 2:
                health_report["warnings"].append(
                    "Many non-embedded fonts detected. Document may not display correctly on other systems."
                )

            # Calculate overall health score
            health_score = 100
            if len(corrupted_pages) > 0:
                health_score -= 30
            if file_size_mb > 20:
                health_score -= 10
            if not health_report["document_health"]["is_valid"]:
                health_score -= 50
            if embedded_fonts < len(fonts_used) / 2:
                health_score -= 5

            health_report["overall_health_score"] = max(0, health_score)
            health_report["processing_time"] = round(time.time() - start_time, 2)

            doc.close()

            return {
                "success": True,
                **health_report
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"PDF health analysis failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2)
            }