"""
Security Analysis Mixin - PDF security analysis and watermark detection
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


class SecurityAnalysisMixin(MCPMixin):
    """
    Handles PDF security analysis including permissions, encryption, and watermark detection.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()
        self.max_file_size = 100 * 1024 * 1024  # 100MB

    @mcp_tool(
        name="analyze_pdf_security",
        description="Analyze PDF security features and potential issues"
    )
    async def analyze_pdf_security(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze PDF security features including encryption, permissions, and vulnerabilities.

        Args:
            pdf_path: Path to PDF file or HTTPS URL

        Returns:
            Dictionary containing security analysis results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            # Basic security information
            is_encrypted = doc.needs_pass
            is_linearized = getattr(doc, 'is_linearized', False)
            pdf_version = getattr(doc, 'pdf_version', 'Unknown')

            # Permission analysis
            permissions = doc.permissions
            permission_details = {
                "print_allowed": bool(permissions & fitz.PDF_PERM_PRINT),
                "copy_allowed": bool(permissions & fitz.PDF_PERM_COPY),
                "modify_allowed": bool(permissions & fitz.PDF_PERM_MODIFY),
                "annotate_allowed": bool(permissions & fitz.PDF_PERM_ANNOTATE),
                "form_fill_allowed": bool(permissions & fitz.PDF_PERM_FORM),
                "extract_allowed": bool(permissions & fitz.PDF_PERM_ACCESSIBILITY),
                "assemble_allowed": bool(permissions & fitz.PDF_PERM_ASSEMBLE),
                "print_high_quality_allowed": bool(permissions & fitz.PDF_PERM_PRINT_HQ)
            }

            # Security warnings and recommendations
            security_warnings = []
            security_recommendations = []

            # Check for common security issues
            if not is_encrypted:
                security_warnings.append("Document is not password protected")
                security_recommendations.append("Consider adding password protection for sensitive documents")

            if permission_details["copy_allowed"] and permission_details["extract_allowed"]:
                security_warnings.append("Text extraction and copying is unrestricted")

            if permission_details["modify_allowed"]:
                security_warnings.append("Document modification is allowed")
                security_recommendations.append("Consider restricting modification permissions")

            # Check PDF version for security considerations
            if isinstance(pdf_version, (int, float)) and pdf_version < 1.4:
                security_warnings.append(f"Old PDF version ({pdf_version}) may have security vulnerabilities")
                security_recommendations.append("Consider updating to PDF version 1.7 or newer")

            # Analyze metadata for potential information disclosure
            metadata = doc.metadata
            metadata_warnings = []

            potentially_sensitive_fields = ["creator", "producer", "title", "author", "subject"]
            for field in potentially_sensitive_fields:
                if metadata.get(field):
                    metadata_warnings.append(f"Metadata contains {field}: {metadata[field][:50]}...")

            if metadata_warnings:
                security_warnings.append("Document metadata may contain sensitive information")
                security_recommendations.append("Review and sanitize metadata before distribution")

            # Check for JavaScript (potential security risk)
            has_javascript = False
            javascript_count = 0

            for page_num in range(min(10, len(doc))):  # Check first 10 pages
                page = doc[page_num]
                try:
                    # Look for JavaScript annotations
                    annotations = page.annots()
                    for annot in annotations:
                        annot_dict = annot.info
                        if 'javascript' in str(annot_dict).lower():
                            has_javascript = True
                            javascript_count += 1
                except:
                    pass

            if has_javascript:
                security_warnings.append(f"Document contains JavaScript ({javascript_count} instances)")
                security_recommendations.append("JavaScript in PDFs can pose security risks - review content")

            # Check for embedded files
            embedded_files = []
            try:
                for i in range(doc.embedded_file_count()):
                    file_info = doc.embedded_file_info(i)
                    embedded_files.append({
                        "name": file_info.get("name", f"embedded_file_{i}"),
                        "size": file_info.get("size", 0),
                        "type": file_info.get("type", "unknown")
                    })
            except:
                pass

            if embedded_files:
                security_warnings.append(f"Document contains {len(embedded_files)} embedded files")
                security_recommendations.append("Embedded files should be scanned for malware")

            # Calculate security score
            security_score = 100
            security_score -= len(security_warnings) * 10
            if not is_encrypted:
                security_score -= 20
            if has_javascript:
                security_score -= 15
            if embedded_files:
                security_score -= 10

            security_score = max(0, security_score)

            # Determine security level
            if security_score >= 80:
                security_level = "High"
            elif security_score >= 60:
                security_level = "Medium"
            elif security_score >= 40:
                security_level = "Low"
            else:
                security_level = "Critical"

            doc.close()

            return {
                "success": True,
                "security_score": security_score,
                "security_level": security_level,
                "encryption_info": {
                    "is_encrypted": is_encrypted,
                    "is_linearized": is_linearized,
                    "pdf_version": pdf_version
                },
                "permissions": permission_details,
                "security_features": {
                    "has_javascript": has_javascript,
                    "javascript_instances": javascript_count,
                    "embedded_files_count": len(embedded_files),
                    "embedded_files": embedded_files
                },
                "metadata_analysis": {
                    "has_metadata": bool(any(metadata.values())),
                    "metadata_warnings": metadata_warnings
                },
                "security_assessment": {
                    "warnings": security_warnings,
                    "recommendations": security_recommendations,
                    "total_issues": len(security_warnings)
                },
                "file_info": {
                    "path": str(path),
                    "file_size": path.stat().st_size
                },
                "analysis_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Security analysis failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "analysis_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="detect_watermarks",
        description="Detect and analyze watermarks in PDF"
    )
    async def detect_watermarks(self, pdf_path: str) -> Dict[str, Any]:
        """
        Detect and analyze watermarks in PDF document.

        Args:
            pdf_path: Path to PDF file or HTTPS URL

        Returns:
            Dictionary containing watermark detection results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            watermark_analysis = []
            total_watermarks = 0
            watermark_types = {"text": 0, "image": 0, "shape": 0}

            # Analyze each page for watermarks
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_watermarks = []

                try:
                    # Check for text watermarks (often low opacity or behind content)
                    text_dict = page.get_text("dict")

                    for block in text_dict.get("blocks", []):
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    text = span.get("text", "").strip()
                                    # Common watermark indicators
                                    if (len(text) > 0 and
                                        (text.upper() in ["DRAFT", "CONFIDENTIAL", "COPY", "SAMPLE", "WATERMARK"] or
                                         "watermark" in text.lower() or
                                         "confidential" in text.lower() or
                                         "draft" in text.lower())):

                                        page_watermarks.append({
                                            "type": "text",
                                            "content": text,
                                            "font_size": span.get("size", 0),
                                            "coordinates": {
                                                "x": round(span.get("bbox", [0, 0, 0, 0])[0], 2),
                                                "y": round(span.get("bbox", [0, 0, 0, 0])[1], 2)
                                            }
                                        })
                                        watermark_types["text"] += 1

                    # Check for image watermarks (semi-transparent images)
                    images = page.get_images()
                    for img_index, img in enumerate(images):
                        try:
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)

                            # Check if image is likely a watermark (small or semi-transparent)
                            if pix.width < 200 or pix.height < 200:
                                page_watermarks.append({
                                    "type": "image",
                                    "size": f"{pix.width}x{pix.height}",
                                    "image_index": img_index + 1,
                                    "coordinates": "analysis_required"
                                })
                                watermark_types["image"] += 1

                            pix = None
                        except:
                            pass

                    # Check for drawing watermarks (shapes, lines)
                    drawings = page.get_drawings()
                    for drawing in drawings:
                        # Simple heuristic: large shapes that might be watermarks
                        if len(drawing.get("items", [])) > 5:  # Complex shape
                            page_watermarks.append({
                                "type": "shape",
                                "complexity": len(drawing.get("items", [])),
                                "coordinates": "shape_detected"
                            })
                            watermark_types["shape"] += 1

                except Exception as e:
                    logger.warning(f"Failed to analyze page {page_num + 1} for watermarks: {e}")

                if page_watermarks:
                    watermark_analysis.append({
                        "page": page_num + 1,
                        "watermarks_found": len(page_watermarks),
                        "watermarks": page_watermarks
                    })
                    total_watermarks += len(page_watermarks)

            doc.close()

            # Watermark assessment
            has_watermarks = total_watermarks > 0
            watermark_density = total_watermarks / len(doc) if len(doc) > 0 else 0

            # Determine watermark pattern
            if watermark_density > 0.8:
                pattern = "comprehensive"  # Most pages have watermarks
            elif watermark_density > 0.3:
                pattern = "selective"      # Some pages have watermarks
            elif watermark_density > 0:
                pattern = "minimal"        # Few pages have watermarks
            else:
                pattern = "none"

            return {
                "success": True,
                "watermark_summary": {
                    "has_watermarks": has_watermarks,
                    "total_watermarks": total_watermarks,
                    "watermark_density": round(watermark_density, 2),
                    "pattern": pattern,
                    "types_found": watermark_types
                },
                "page_analysis": watermark_analysis,
                "watermark_insights": {
                    "pages_with_watermarks": len(watermark_analysis),
                    "pages_without_watermarks": len(doc) - len(watermark_analysis),
                    "most_common_type": max(watermark_types, key=watermark_types.get) if any(watermark_types.values()) else "none"
                },
                "recommendations": [
                    "Check text watermarks for sensitive information disclosure",
                    "Verify image watermarks don't contain hidden data",
                    "Consider watermark removal if document is for public distribution"
                ] if has_watermarks else ["No watermarks detected"],
                "file_info": {
                    "path": str(path),
                    "total_pages": len(doc)
                },
                "analysis_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Watermark detection failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "analysis_time": round(time.time() - start_time, 2)
            }