"""
Security Analysis Mixin - PDF security analysis and watermark detection
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


class SecurityAnalysisMixin(MCPMixin):
    """
    Handles PDF security analysis including encryption, permissions,
    JavaScript detection, and watermark identification.

    Tools provided:
    - analyze_pdf_security: Comprehensive security analysis
    - detect_watermarks: Detect and analyze watermarks
    """

    def get_mixin_name(self) -> str:
        return "SecurityAnalysis"

    def get_required_permissions(self) -> List[str]:
        return ["read_files", "security_analysis"]

    def _setup(self):
        """Initialize security analysis specific configuration"""
        self.sensitive_keywords = ['password', 'ssn', 'credit', 'bank', 'account']
        self.watermark_keywords = [
            'confidential', 'draft', 'copy', 'watermark', 'sample',
            'preview', 'demo', 'trial', 'protected'
        ]

    @mcp_tool(
        name="analyze_pdf_security",
        description="Analyze PDF security features and potential issues"
    )
    async def analyze_pdf_security(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze PDF security features and potential issues.

        Args:
            pdf_path: Path to PDF file or HTTPS URL

        Returns:
            Dictionary containing security analysis results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            security_report = {
                "success": True,
                "file_info": {
                    "path": str(path),
                    "size_bytes": path.stat().st_size
                },
                "encryption": {},
                "permissions": {},
                "signatures": {},
                "javascript": {},
                "security_warnings": [],
                "security_score": 0
            }

            # Encryption analysis
            security_report["encryption"]["is_encrypted"] = doc.is_encrypted
            security_report["encryption"]["needs_password"] = doc.needs_pass
            security_report["encryption"]["can_open"] = not doc.needs_pass

            # Check for password protection
            if doc.is_encrypted and not doc.needs_pass:
                security_report["encryption"]["encryption_type"] = "owner_password_only"
            elif doc.needs_pass:
                security_report["encryption"]["encryption_type"] = "user_password_required"
            else:
                security_report["encryption"]["encryption_type"] = "none"

            # Permission analysis
            if hasattr(doc, 'permissions'):
                perms = doc.permissions
                security_report["permissions"] = {
                    "can_print": bool(perms & 4),
                    "can_modify": bool(perms & 8),
                    "can_copy": bool(perms & 16),
                    "can_annotate": bool(perms & 32),
                    "can_form_fill": bool(perms & 256),
                    "can_extract_for_accessibility": bool(perms & 512),
                    "can_assemble": bool(perms & 1024),
                    "can_print_high_quality": bool(perms & 2048)
                }

            # JavaScript detection
            has_js = False
            js_count = 0

            for page_num in range(min(len(doc), 10)):  # Check first 10 pages for performance
                page = doc[page_num]
                text = page.get_text()

                # Simple JavaScript detection
                if any(keyword in text.lower() for keyword in ['javascript:', '/js', 'app.alert', 'this.print']):
                    has_js = True
                    js_count += 1

            security_report["javascript"]["detected"] = has_js
            security_report["javascript"]["pages_with_js"] = js_count

            if has_js:
                security_report["security_warnings"].append("JavaScript detected - potential security risk")

            # Digital signature detection (basic)
            security_report["signatures"]["has_signatures"] = doc.signature_count() > 0 if hasattr(doc, 'signature_count') else False
            security_report["signatures"]["signature_count"] = doc.signature_count() if hasattr(doc, 'signature_count') else 0

            # File size anomalies
            if security_report["file_info"]["size_bytes"] > 100 * 1024 * 1024:  # > 100MB
                security_report["security_warnings"].append("Large file size - review for embedded content")

            # Metadata analysis for privacy
            metadata = doc.metadata
            sensitive_metadata = []

            for key, value in metadata.items():
                if value and len(str(value)) > 0:
                    if any(word in str(value).lower() for word in ['user', 'author', 'creator']):
                        sensitive_metadata.append(key)

            if sensitive_metadata:
                security_report["security_warnings"].append(f"Potentially sensitive metadata found: {', '.join(sensitive_metadata)}")

            # Form analysis for security
            if doc.is_form_pdf:
                # Check for potentially dangerous form actions
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    widgets = page.widgets()

                    for widget in widgets:
                        if hasattr(widget, 'field_name') and widget.field_name:
                            if any(dangerous in widget.field_name.lower() for dangerous in self.sensitive_keywords):
                                security_report["security_warnings"].append("Form contains potentially sensitive field names")
                                break

            # Calculate security score
            score = 100

            if not doc.is_encrypted:
                score -= 20
            if has_js:
                score -= 30
            if len(security_report["security_warnings"]) > 0:
                score -= len(security_report["security_warnings"]) * 10
            if sensitive_metadata:
                score -= 10

            security_report["security_score"] = max(0, min(100, score))

            # Security level assessment
            if score >= 80:
                security_level = "high"
            elif score >= 60:
                security_level = "medium"
            elif score >= 40:
                security_level = "low"
            else:
                security_level = "critical"

            security_report["security_level"] = security_level

            doc.close()
            security_report["analysis_time"] = round(time.time() - start_time, 2)

            return security_report

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
        Detect and analyze watermarks in PDF.

        Args:
            pdf_path: Path to PDF file or HTTPS URL

        Returns:
            Dictionary containing watermark detection results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            watermark_report = {
                "success": True,
                "has_watermarks": False,
                "watermarks_detected": [],
                "detection_summary": {},
                "analysis_time": 0
            }

            text_watermarks = []
            image_watermarks = []

            # Check each page for potential watermarks
            for page_num, page in enumerate(doc):
                # Text-based watermark detection
                # Look for text with unusual properties (transparency, large size, repetitive)
                text_blocks = page.get_text("dict")["blocks"]

                for block in text_blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                font_size = span["size"]

                                # Heuristics for watermark detection
                                is_potential_watermark = (
                                    len(text) > 3 and
                                    (font_size > 40 or  # Large text
                                     any(keyword in text.lower() for keyword in self.watermark_keywords) or
                                     text.count(' ') == 0 and len(text) > 8)  # Long single word
                                )

                                if is_potential_watermark:
                                    text_watermarks.append({
                                        "page": page_num + 1,
                                        "text": text,
                                        "font_size": font_size,
                                        "coordinates": {
                                            "x": span["bbox"][0],
                                            "y": span["bbox"][1]
                                        },
                                        "type": "text"
                                    })

                # Image-based watermark detection (basic)
                # Look for images that might be watermarks
                images = page.get_images()

                for img_index, img in enumerate(images):
                    try:
                        # Get image properties
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)

                        # Small or very large images might be watermarks
                        if pix.width < 200 and pix.height < 200:  # Small logos
                            image_watermarks.append({
                                "page": page_num + 1,
                                "size": f"{pix.width}x{pix.height}",
                                "type": "small_image",
                                "potential_logo": True
                            })
                        elif pix.width > 1000 or pix.height > 1000:  # Large background
                            image_watermarks.append({
                                "page": page_num + 1,
                                "size": f"{pix.width}x{pix.height}",
                                "type": "large_background",
                                "potential_background": True
                            })

                        pix = None  # Clean up

                    except Exception as e:
                        logger.debug(f"Could not analyze image on page {page_num + 1}: {e}")

            # Combine results
            all_watermarks = text_watermarks + image_watermarks

            watermark_report["has_watermarks"] = len(all_watermarks) > 0
            watermark_report["watermarks_detected"] = all_watermarks

            # Summary
            watermark_report["detection_summary"] = {
                "total_detected": len(all_watermarks),
                "text_watermarks": len(text_watermarks),
                "image_watermarks": len(image_watermarks),
                "pages_with_watermarks": len(set(w["page"] for w in all_watermarks)),
                "total_pages": len(doc)
            }

            doc.close()
            watermark_report["analysis_time"] = round(time.time() - start_time, 2)

            return watermark_report

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Watermark detection failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "analysis_time": round(time.time() - start_time, 2)
            }