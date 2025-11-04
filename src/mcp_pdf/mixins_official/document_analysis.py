"""
Document Analysis Mixin - PDF metadata, structure, and health analysis
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# PDF processing libraries
import fitz  # PyMuPDF
from PIL import Image
import io

# Official FastMCP mixin
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, sanitize_error_message

logger = logging.getLogger(__name__)


class DocumentAnalysisMixin(MCPMixin):
    """
    Handles PDF document analysis operations including metadata, structure, and health checks.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()
        self.max_file_size = 100 * 1024 * 1024  # 100MB

    @mcp_tool(
        name="extract_metadata",
        description="Extract comprehensive PDF metadata"
    )
    async def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from PDF document.

        Args:
            pdf_path: Path to PDF file or HTTPS URL

        Returns:
            Dictionary containing document metadata
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            # Extract basic metadata
            metadata = doc.metadata

            # Get document structure information
            page_count = len(doc)
            total_text_length = 0
            total_images = 0
            total_links = 0

            # Sample first few pages for analysis
            sample_size = min(5, page_count)

            for page_num in range(sample_size):
                page = doc[page_num]
                page_text = page.get_text()
                total_text_length += len(page_text)
                total_images += len(page.get_images())
                total_links += len(page.get_links())

            # Estimate total document statistics
            if sample_size > 0:
                avg_text_per_page = total_text_length / sample_size
                avg_images_per_page = total_images / sample_size
                avg_links_per_page = total_links / sample_size

                estimated_total_text = int(avg_text_per_page * page_count)
                estimated_total_images = int(avg_images_per_page * page_count)
                estimated_total_links = int(avg_links_per_page * page_count)
            else:
                estimated_total_text = 0
                estimated_total_images = 0
                estimated_total_links = 0

            # Get document permissions
            permissions = {
                "printing": doc.permissions & fitz.PDF_PERM_PRINT != 0,
                "copying": doc.permissions & fitz.PDF_PERM_COPY != 0,
                "modification": doc.permissions & fitz.PDF_PERM_MODIFY != 0,
                "annotation": doc.permissions & fitz.PDF_PERM_ANNOTATE != 0
            }

            # Check for encryption
            is_encrypted = doc.needs_pass
            is_linearized = doc.is_pdf and hasattr(doc, 'is_fast_web_view') and doc.is_fast_web_view

            doc.close()

            # File size information
            file_size = path.stat().st_size
            file_size_mb = round(file_size / (1024 * 1024), 2)

            return {
                "success": True,
                "metadata": {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "keywords": metadata.get("keywords", ""),
                    "creator": metadata.get("creator", ""),
                    "producer": metadata.get("producer", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "modification_date": metadata.get("modDate", ""),
                    "trapped": metadata.get("trapped", "")
                },
                "document_info": {
                    "page_count": page_count,
                    "file_size_bytes": file_size,
                    "file_size_mb": file_size_mb,
                    "is_encrypted": is_encrypted,
                    "is_linearized": is_linearized,
                    "pdf_version": getattr(doc, 'pdf_version', 'Unknown')
                },
                "content_analysis": {
                    "estimated_text_characters": estimated_total_text,
                    "estimated_total_images": estimated_total_images,
                    "estimated_total_links": estimated_total_links,
                    "sample_pages_analyzed": sample_size
                },
                "permissions": permissions,
                "file_info": {
                    "path": str(path)
                },
                "extraction_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Metadata extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "extraction_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="get_document_structure",
        description="Extract document structure and outline"
    )
    async def get_document_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract document structure including bookmarks, outline, and page organization.

        Args:
            pdf_path: Path to PDF file or HTTPS URL

        Returns:
            Dictionary containing document structure information
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            # Extract table of contents/bookmarks
            toc = doc.get_toc()
            bookmarks = []

            for item in toc:
                level, title, page = item
                bookmarks.append({
                    "level": level,
                    "title": title.strip(),
                    "page": page,
                    "indent": "  " * (level - 1) + title.strip()
                })

            # Analyze page sizes and orientations
            page_analysis = []
            unique_page_sizes = set()

            for page_num in range(len(doc)):
                page = doc[page_num]
                rect = page.rect
                width, height = rect.width, rect.height

                # Determine orientation
                if width > height:
                    orientation = "landscape"
                elif height > width:
                    orientation = "portrait"
                else:
                    orientation = "square"

                page_info = {
                    "page": page_num + 1,
                    "width": round(width, 2),
                    "height": round(height, 2),
                    "orientation": orientation,
                    "rotation": page.rotation
                }
                page_analysis.append(page_info)
                unique_page_sizes.add((round(width, 2), round(height, 2)))

            # Document structure analysis
            has_bookmarks = len(bookmarks) > 0
            has_uniform_pages = len(unique_page_sizes) == 1
            total_pages = len(doc)

            # Check for forms
            has_forms = False
            try:
                # Simple check for form fields
                for page_num in range(min(5, total_pages)):  # Check first 5 pages
                    page = doc[page_num]
                    widgets = page.widgets()
                    if widgets:
                        has_forms = True
                        break
            except:
                pass

            doc.close()

            return {
                "success": True,
                "structure_summary": {
                    "total_pages": total_pages,
                    "has_bookmarks": has_bookmarks,
                    "bookmark_count": len(bookmarks),
                    "has_uniform_page_sizes": has_uniform_pages,
                    "unique_page_sizes": len(unique_page_sizes),
                    "has_forms": has_forms
                },
                "bookmarks": bookmarks,
                "page_analysis": {
                    "total_pages": total_pages,
                    "unique_page_sizes": list(unique_page_sizes),
                    "pages": page_analysis[:10]  # Limit to first 10 pages for context
                },
                "document_organization": {
                    "bookmark_hierarchy_depth": max([b["level"] for b in bookmarks]) if bookmarks else 0,
                    "estimated_sections": len([b for b in bookmarks if b["level"] <= 2]),
                    "page_size_consistency": has_uniform_pages
                },
                "file_info": {
                    "path": str(path)
                },
                "analysis_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Document structure analysis failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "analysis_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="analyze_pdf_health",
        description="Comprehensive PDF health analysis"
    )
    async def analyze_pdf_health(self, pdf_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive health analysis of PDF document.

        Args:
            pdf_path: Path to PDF file or HTTPS URL

        Returns:
            Dictionary containing health analysis results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            health_issues = []
            warnings = []
            recommendations = []

            # Check basic document properties
            total_pages = len(doc)
            file_size = path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            # File size analysis
            if file_size_mb > 50:
                warnings.append(f"Large file size: {file_size_mb:.1f}MB")
                recommendations.append("Consider optimizing or compressing the PDF")

            # Page count analysis
            if total_pages > 500:
                warnings.append(f"Large document: {total_pages} pages")
                recommendations.append("Consider splitting into smaller documents")

            # Check for corruption or structural issues
            try:
                # Test if we can read all pages
                problematic_pages = []
                for page_num in range(min(10, total_pages)):  # Check first 10 pages
                    try:
                        page = doc[page_num]
                        page.get_text()  # Try to extract text
                        page.get_images()  # Try to get images
                    except Exception as e:
                        problematic_pages.append(page_num + 1)
                        health_issues.append(f"Page {page_num + 1} has reading issues: {str(e)[:100]}")

                if problematic_pages:
                    recommendations.append("Some pages may be corrupted - verify document integrity")

            except Exception as e:
                health_issues.append(f"Document structure issues: {str(e)[:100]}")

            # Check encryption and security
            is_encrypted = doc.needs_pass
            if is_encrypted:
                health_issues.append("Document is password protected")

            # Check permissions
            permissions = doc.permissions
            if permissions == 0:
                warnings.append("Document has restricted permissions")

            # Analyze content quality
            sample_pages = min(5, total_pages)
            total_text = 0
            total_images = 0
            blank_pages = 0

            for page_num in range(sample_pages):
                page = doc[page_num]
                text = page.get_text().strip()
                images = page.get_images()

                total_text += len(text)
                total_images += len(images)

                if len(text) < 10 and len(images) == 0:
                    blank_pages += 1

            # Content quality analysis
            if blank_pages > 0:
                warnings.append(f"Found {blank_pages} potentially blank pages in sample")

            avg_text_per_page = total_text / sample_pages if sample_pages > 0 else 0
            if avg_text_per_page < 100:
                warnings.append("Low text content - may be image-based PDF")
                recommendations.append("Consider OCR for text extraction")

            # Check PDF version
            pdf_version = getattr(doc, 'pdf_version', 'Unknown')
            if pdf_version and isinstance(pdf_version, (int, float)):
                if pdf_version < 1.4:
                    warnings.append(f"Old PDF version: {pdf_version}")
                    recommendations.append("Consider updating to newer PDF version")

            doc.close()

            # Determine overall health score
            health_score = 100
            health_score -= len(health_issues) * 20  # Major issues
            health_score -= len(warnings) * 5       # Minor issues
            health_score = max(0, health_score)

            # Determine health status
            if health_score >= 90:
                health_status = "Excellent"
            elif health_score >= 70:
                health_status = "Good"
            elif health_score >= 50:
                health_status = "Fair"
            else:
                health_status = "Poor"

            return {
                "success": True,
                "health_score": health_score,
                "health_status": health_status,
                "summary": {
                    "total_issues": len(health_issues),
                    "total_warnings": len(warnings),
                    "total_recommendations": len(recommendations)
                },
                "issues": health_issues,
                "warnings": warnings,
                "recommendations": recommendations,
                "document_stats": {
                    "total_pages": total_pages,
                    "file_size_mb": round(file_size_mb, 2),
                    "pdf_version": pdf_version,
                    "is_encrypted": is_encrypted,
                    "sample_pages_analyzed": sample_pages,
                    "estimated_text_density": round(avg_text_per_page, 1)
                },
                "file_info": {
                    "path": str(path)
                },
                "analysis_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"PDF health analysis failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "analysis_time": round(time.time() - start_time, 2)
            }