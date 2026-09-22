"""
Document Analysis Mixin - PDF metadata, structure, and health analysis
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import time
from typing import Dict, Any
import logging

# PDF processing libraries
import pymupdf

# Official FastMCP mixin
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, sanitize_error_message
from ..xfa import is_xfa_pdf as _detect_xfa

logger = logging.getLogger(__name__)


class DocumentAnalysisMixin(MCPMixin):
    """
    Handles PDF document analysis operations including metadata, structure, and health checks.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()

    @mcp_tool(
        name="extract_metadata",
        description=(
            "Read the document information dictionary (title, author, subject, "
            "keywords, creator, producer, creation/modification dates, trapped) "
            "plus page count, file size, permission flags and rough content "
            "estimates. Read-only: writes nothing.\n"
            "\n"
            "Pick between the three document-level analysers by what you want:\n"
            "  extract_metadata     — the factual header fields and page count. "
            "Start here when you need the title/author/date or just how many "
            "pages the file has.\n"
            "  analyze_pdf_health   — is this file usable? Returns a 0-100 score, "
            "issues, warnings, recommendations, plus XFA detection.\n"
            "  analyze_pdf_security — how locked-down is it? Encryption, the full "
            "permission bitmask, JavaScript and metadata-disclosure warnings.\n"
            "\n"
            "Caveats worth knowing before you trust the numbers: everything under "
            "`content_analysis` (estimated_text_characters, estimated_total_images, "
            "estimated_total_links) is EXTRAPOLATED from the first 5 pages, not "
            "counted — use extract_images or extract_links for real counts. "
            "`is_encrypted` is really \"needs a password to open\", so an "
            "owner-password-only file reads as False. `is_linearized` is always "
            "False and `pdf_version` is always the string \"Unknown\" in the "
            "current build; ignore both."
        ),
        annotations={
            "readOnlyHint": True,        # opens the PDF, writes nothing
            "idempotentHint": True,
            "openWorldHint": True,       # pdf_path may be an http(s) URL
        },
    )
    async def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from PDF document.

        Args:
            pdf_path: Path to the PDF, or an http(s) URL to fetch.

        Returns:
            Dict with success plus:
              - metadata: title, author, subject, keywords, creator, producer,
                creation_date, modification_date, trapped. Absent fields come
                back as "" rather than being omitted.
              - document_info: page_count, file_size_bytes, file_size_mb,
                is_encrypted (needs a password to open), is_linearized
                (always False, see description), pdf_version (always
                "Unknown", see description).
              - content_analysis: estimated_text_characters,
                estimated_total_images, estimated_total_links — all
                extrapolated from sample_pages_analyzed (first 5 pages).
              - permissions: printing, copying, modification, annotation booleans.
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = pymupdf.open(str(path))

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
                "printing": doc.permissions & pymupdf.PDF_PERM_PRINT != 0,
                "copying": doc.permissions & pymupdf.PDF_PERM_COPY != 0,
                "modification": doc.permissions & pymupdf.PDF_PERM_MODIFY != 0,
                "annotation": doc.permissions & pymupdf.PDF_PERM_ANNOTATE != 0
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
        description=(
            "Cheap structural overview of a PDF: does it have a bookmark "
            "outline, how deep, are all pages the same size, which way are they "
            "rotated, does it contain form fields. Read-only, no file written, "
            "one pass over the pages.\n"
            "\n"
            "This is the shallow one. Choose between the three structure tools:\n"
            "  get_document_structure — the PDF's OWN embedded outline plus page "
            "geometry. Nothing is inferred. Fast, and returns only a TRUNCATED "
            "TEXT PREVIEW of the bookmarks (first 20 indented titles); the "
            "per-bookmark page numbers are NOT in the response.\n"
            "  detect_structure       — infers chapters/sections even when there "
            "are no bookmarks, using font-size analysis and numbering patterns, "
            "and writes the full hierarchy to JSON. Use it when you need real "
            "section boundaries and page ranges to act on.\n"
            "  analyze_layout         — within-page geometry (text blocks, "
            "columns, coverage), not document-level sectioning.\n"
            "\n"
            "Caveat: has_forms is decided by looking at the FIRST 5 PAGES ONLY, "
            "so a form whose fields start on page 6 reports has_forms=false. "
            "unique_page_sizes comes back as a list of [width, height] pairs in "
            "PDF points."
        ),
        annotations={
            "readOnlyHint": True,        # opens the PDF, writes nothing
            "idempotentHint": True,
            "openWorldHint": True,       # pdf_path may be an http(s) URL
        },
    )
    async def get_document_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract document structure including bookmarks, outline, and page organization.

        Args:
            pdf_path: Path to the PDF, or an http(s) URL to fetch.

        Returns:
            Dict with success plus:
              - structure_summary: total_pages, has_bookmarks, bookmark_count,
                bookmark_hierarchy_depth (deepest outline level, 0 if none),
                estimated_sections (bookmarks at level 1 or 2),
                has_uniform_page_sizes, unique_page_sizes (list of
                [width, height] point pairs), has_forms (first 5 pages only).
              - bookmark_preview: up to 20 indented title strings, then a
                "... and N more bookmarks" line. Titles only — no page
                numbers, and no way to page past the first 20. Use
                detect_structure when you need the whole outline.
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = pymupdf.open(str(path))

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

            # Cap bookmark preview to avoid flooding MCP context
            max_bookmark_preview = 20
            bookmark_preview = [
                b["indent"] for b in bookmarks[:max_bookmark_preview]
            ]
            if len(bookmarks) > max_bookmark_preview:
                bookmark_preview.append(
                    f"... and {len(bookmarks) - max_bookmark_preview} more bookmarks"
                )

            return {
                "success": True,
                "structure_summary": {
                    "total_pages": total_pages,
                    "has_bookmarks": has_bookmarks,
                    "bookmark_count": len(bookmarks),
                    "bookmark_hierarchy_depth": max(b["level"] for b in bookmarks) if bookmarks else 0,
                    "estimated_sections": len([b for b in bookmarks if b["level"] <= 2]),
                    "has_uniform_page_sizes": has_uniform_pages,
                    "unique_page_sizes": list(unique_page_sizes),
                    "has_forms": has_forms,
                },
                "bookmark_preview": bookmark_preview,
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
        description=(
            "Triage a PDF before you spend time on it: can its pages be read, "
            "is it password protected, is it a dynamic XFA form, is it so "
            "text-poor that it probably needs OCR. Returns a 0-100 "
            "health_score with separate issues / warnings / recommendations "
            "lists. Read-only, writes nothing.\n"
            "\n"
            "Run this FIRST when a PDF misbehaves; the XFA warning in "
            "particular explains why extract_form_data or ocr_pdf would "
            "otherwise return an almost-empty Adobe placeholder page. Use "
            "extract_metadata instead for plain header fields, and "
            "analyze_pdf_security for encryption and permission detail — "
            "health is about usability, security is about lockdown.\n"
            "\n"
            "How to read health_score: it starts at 100 and loses 20 per entry "
            "in `issues` and 5 per entry in `warnings`, floored at 0. "
            "health_status is Excellent >=90, Good >=70, Fair >=50, else Poor. "
            "The score is therefore a count of complaints, not a measurement — "
            "read the `issues` and `warnings` strings themselves.\n"
            "\n"
            "Sampling limits: page-read errors are probed on the first 10 pages "
            "only, and blank-page / text-density figures come from the first 5. "
            "estimated_text_density is mean characters per sampled page; below "
            "100 it recommends OCR. pdf_version is always \"Unknown\" in the "
            "current build, so the old-version warning never fires."
        ),
        annotations={
            "readOnlyHint": True,        # opens the PDF, writes nothing
            "idempotentHint": True,
            "openWorldHint": True,       # pdf_path may be an http(s) URL
        },
    )
    async def analyze_pdf_health(self, pdf_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive health analysis of PDF document.

        Args:
            pdf_path: Path to the PDF, or an http(s) URL to fetch.

        Returns:
            Dict with success plus:
              - health_score (0-100) and health_status
                (Excellent/Good/Fair/Poor)
              - summary: total_issues, total_warnings, total_recommendations
              - issues: human-readable strings for hard problems (unreadable
                pages, password protection)
              - warnings: softer findings (large file, >500 pages, blank pages
                in the sample, low text density, XFA)
              - recommendations: suggested next actions, e.g. "Consider OCR
                for text extraction"
              - document_stats: total_pages, file_size_mb, pdf_version
                (always "Unknown"), is_encrypted, is_xfa, xfa_type
                ("dynamic"/"static"/None), xfa_detection_failed,
                sample_pages_analyzed, estimated_text_density
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = pymupdf.open(str(path))

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

            # Detect XFA, because dynamic XFA changes what other tools
            # (extract_form_data, convert_to_images, ocr_pdf) can deliver.
            #
            # Guarded locally and on purpose. This is an incidental probe
            # bolted onto an analysis that has already completed by this
            # point, so it must not be able to throw away a successful result:
            # pypdf can fail on files MuPDF reads fine (encryption revisions,
            # for one), and without this guard that failure would turn a good
            # analysis into success: False.
            try:
                xfa_info = _detect_xfa(str(path))
            except Exception as e:
                logger.warning(f"XFA probe failed, continuing: {e}")
                xfa_info = {
                    "is_xfa": None, "xfa_type": None, "detection_failed": True,
                }

            if xfa_info.get("detection_failed"):
                warnings.append(
                    "Could not determine whether this is an XFA form; pypdf "
                    "could not read the file structure even though MuPDF "
                    "could. That asymmetry often means the file is damaged."
                )
            elif xfa_info.get("is_xfa") and xfa_info.get("xfa_type") == "dynamic":
                warnings.append(
                    "Dynamic XFA form detected. Most tools will only see the "
                    "Adobe placeholder page; use extract_xfa_fields for the "
                    "form schema."
                )

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
                    "is_xfa": xfa_info.get("is_xfa"),
                    "xfa_type": xfa_info.get("xfa_type"),
                    "xfa_detection_failed": xfa_info.get("detection_failed", False),
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