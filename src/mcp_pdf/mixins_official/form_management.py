"""
Form Management Mixin - PDF form creation, filling, and field extraction
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import asyncio
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# PDF processing libraries
import fitz  # PyMuPDF
# Note: reportlab is imported lazily in create_form_pdf (optional dependency)

# Official FastMCP mixin
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, validate_output_path, sanitize_error_message
from ..xfa import extract_xfa_schema, is_xfa_pdf as _detect_xfa

logger = logging.getLogger(__name__)


class FormManagementMixin(MCPMixin):
    """
    Handles PDF form operations including creation, filling, and field extraction.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()

    @mcp_tool(
        name="extract_form_data",
        description="Extract form fields and values"
    )
    async def extract_form_data(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract all form fields and their current values from PDF.

        Args:
            pdf_path: Path to PDF file or HTTPS URL

        Returns:
            Dictionary containing form fields and their values
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)

            # XFA early-detect — dynamic XFA forms have no AcroForm widgets,
            # so we'd return total_fields=0 and the user would think the form
            # is empty. Surface the real diagnosis + a pointer to the right tool.
            xfa_info = _detect_xfa(str(path))
            if xfa_info["is_xfa"] and xfa_info["xfa_type"] == "dynamic":
                return {
                    "success": False,
                    "is_xfa": True,
                    "xfa_type": "dynamic",
                    "error": (
                        "Dynamic XFA form — fields are not in AcroForm and "
                        "cannot be extracted by this tool."
                    ),
                    "hint": (
                        "Use extract_xfa_fields to get the XFA field schema "
                        "(names, types, captions, shared/positional categories)."
                    ),
                    "extraction_time": round(time.time() - start_time, 2),
                }

            doc = fitz.open(str(path))

            form_fields = []
            total_fields = 0

            for page_num in range(len(doc)):
                page = doc[page_num]

                try:
                    # Get form widgets (interactive fields)
                    widgets = page.widgets()

                    for widget in widgets:
                        field_info = {
                            "page": page_num + 1,
                            "field_name": widget.field_name or f"field_{total_fields + 1}",
                            "field_type": self._get_field_type(widget),
                            "field_value": widget.field_value or "",
                            "field_label": widget.field_label or "",
                            "is_required": getattr(widget, 'field_flags', 0) & 2 != 0,  # Required flag
                            "is_readonly": getattr(widget, 'field_flags', 0) & 1 != 0,  # Readonly flag
                            "coordinates": {
                                "x": round(widget.rect.x0, 2),
                                "y": round(widget.rect.y0, 2),
                                "width": round(widget.rect.width, 2),
                                "height": round(widget.rect.height, 2)
                            }
                        }

                        # Add field-specific properties
                        if hasattr(widget, 'choice_values') and widget.choice_values:
                            field_info["choices"] = widget.choice_values

                        if hasattr(widget, 'text_maxlen') and widget.text_maxlen:
                            field_info["max_length"] = widget.text_maxlen

                        form_fields.append(field_info)
                        total_fields += 1

                except Exception as e:
                    logger.warning(f"Failed to extract widgets from page {page_num + 1}: {e}")

            doc.close()

            # Analyze form structure
            field_types = {}
            required_fields = 0
            readonly_fields = 0

            for field in form_fields:
                field_type = field["field_type"]
                field_types[field_type] = field_types.get(field_type, 0) + 1

                if field["is_required"]:
                    required_fields += 1
                if field["is_readonly"]:
                    readonly_fields += 1

            return {
                "success": True,
                "form_summary": {
                    "total_fields": total_fields,
                    "required_fields": required_fields,
                    "readonly_fields": readonly_fields,
                    "field_types": field_types,
                    "has_form": total_fields > 0
                },
                "form_fields": form_fields,
                "file_info": {
                    "path": str(path),
                    "total_pages": len(doc) if 'doc' in locals() else 0
                },
                "extraction_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Form data extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "extraction_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="fill_form_pdf",
        description="Fill PDF form with provided data"
    )
    async def fill_form_pdf(
        self,
        input_path: str,
        output_path: str,
        form_data: str,
        flatten: bool = False
    ) -> Dict[str, Any]:
        """
        Fill an existing PDF form with provided data.

        Args:
            input_path: Path to input PDF file or HTTPS URL
            output_path: Path where filled PDF will be saved
            form_data: JSON string containing field names and values
            flatten: Whether to flatten the form (make fields non-editable)

        Returns:
            Dictionary containing operation results
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_pdf_path = validate_output_path(output_path)

            # Parse form data
            try:
                data = json.loads(form_data)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON in form_data: {e}",
                    "fill_time": round(time.time() - start_time, 2)
                }

            # Open and process the PDF
            doc = fitz.open(str(input_pdf_path))
            fields_filled = 0
            fields_failed = 0
            failed_fields = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                try:
                    widgets = page.widgets()

                    for widget in widgets:
                        field_name = widget.field_name
                        if field_name and field_name in data:
                            try:
                                # Set field value
                                widget.field_value = str(data[field_name])
                                widget.update()
                                fields_filled += 1
                            except Exception as e:
                                fields_failed += 1
                                failed_fields.append({
                                    "field_name": field_name,
                                    "error": str(e)
                                })

                except Exception as e:
                    logger.warning(f"Failed to process widgets on page {page_num + 1}: {e}")

            # Save the filled PDF
            if flatten:
                # Create a flattened version by rendering to new PDF
                flattened_doc = fitz.open()
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap()
                    new_page = flattened_doc.new_page(width=page.rect.width, height=page.rect.height)
                    new_page.insert_image(new_page.rect, pixmap=pix)

                flattened_doc.save(str(output_pdf_path))
                flattened_doc.close()
            else:
                doc.save(str(output_pdf_path), incremental=False, encryption=fitz.PDF_ENCRYPT_NONE)

            doc.close()

            return {
                "success": True,
                "fill_summary": {
                    "fields_filled": fields_filled,
                    "fields_failed": fields_failed,
                    "total_data_provided": len(data),
                    "form_flattened": flatten
                },
                "failed_fields": failed_fields,
                "output_info": {
                    "output_path": str(output_pdf_path),
                    "output_size_bytes": output_pdf_path.stat().st_size
                },
                "fill_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Form filling failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "fill_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="create_form_pdf",
        description="Create new PDF form with interactive fields"
    )
    async def create_form_pdf(
        self,
        output_path: str,
        fields: str,
        title: str = "Form Document",
        page_size: str = "A4"
    ) -> Dict[str, Any]:
        """
        Create a new PDF form with interactive fields.

        Args:
            output_path: Path where new PDF form will be saved
            fields: JSON string describing form fields
            title: Document title
            page_size: Page size ("A4", "Letter", "Legal")

        Returns:
            Dictionary containing creation results
        """
        start_time = time.time()

        try:
            # Lazy import reportlab (optional dependency)
            try:
                from reportlab.pdfgen import canvas
                from reportlab.lib.pagesizes import letter, A4, legal
                from reportlab.lib.colors import black, blue, red
            except ImportError:
                return {
                    "success": False,
                    "error": "reportlab is required for create_form_pdf. Install with: pip install mcp-pdf[forms]",
                    "creation_time": round(time.time() - start_time, 2)
                }

            # Validate output path
            output_pdf_path = validate_output_path(output_path)

            # Parse fields data
            try:
                field_definitions = json.loads(fields)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON in fields: {e}",
                    "creation_time": round(time.time() - start_time, 2)
                }

            # Set page size
            page_sizes = {
                "A4": A4,
                "Letter": letter,
                "Legal": legal
            }
            page_size_tuple = page_sizes.get(page_size, A4)

            # Create PDF using ReportLab
            def create_form():
                c = canvas.Canvas(str(output_pdf_path), pagesize=page_size_tuple)
                c.setTitle(title)

                fields_created = 0

                for field_def in field_definitions:
                    try:
                        field_name = field_def.get("name", f"field_{fields_created + 1}")
                        field_type = field_def.get("type", "text")
                        x = field_def.get("x", 50)
                        y = field_def.get("y", 700 - (fields_created * 40))
                        width = field_def.get("width", 200)
                        height = field_def.get("height", 20)
                        label = field_def.get("label", field_name)

                        # Draw field label
                        c.drawString(x, y + height + 5, label)

                        # Create field based on type
                        if field_type == "text":
                            c.acroForm.textfield(
                                name=field_name,
                                tooltip=field_def.get("tooltip", ""),
                                x=x, y=y, width=width, height=height,
                                borderWidth=1,
                                forceBorder=True
                            )

                        elif field_type == "checkbox":
                            c.acroForm.checkbox(
                                name=field_name,
                                tooltip=field_def.get("tooltip", ""),
                                x=x, y=y, size=height,
                                checked=field_def.get("checked", False),
                                buttonStyle='check'
                            )

                        elif field_type == "dropdown":
                            options = field_def.get("options", ["Option 1", "Option 2"])
                            c.acroForm.choice(
                                name=field_name,
                                tooltip=field_def.get("tooltip", ""),
                                x=x, y=y, width=width, height=height,
                                options=options,
                                forceBorder=True
                            )

                        elif field_type == "signature":
                            c.acroForm.textfield(
                                name=field_name,
                                tooltip="Digital signature field",
                                x=x, y=y, width=width, height=height,
                                borderWidth=2,
                                forceBorder=True
                            )
                            # Draw signature indicator
                            c.setFillColor(blue)
                            c.drawString(x + 5, y + 5, "SIGNATURE")
                            c.setFillColor(black)

                        fields_created += 1

                    except Exception as e:
                        logger.warning(f"Failed to create field {field_def}: {e}")

                c.save()
                return fields_created

            # Run in executor to avoid blocking
            fields_created = await asyncio.get_event_loop().run_in_executor(None, create_form)

            return {
                "success": True,
                "form_info": {
                    "fields_created": fields_created,
                    "total_fields_requested": len(field_definitions),
                    "page_size": page_size,
                    "title": title
                },
                "output_info": {
                    "output_path": str(output_pdf_path),
                    "output_size_bytes": output_pdf_path.stat().st_size
                },
                "creation_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Form creation failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "creation_time": round(time.time() - start_time, 2)
            }

    # Helper methods
    def _get_field_type(self, widget) -> str:
        """Map PyMuPDF widget type to the portable cross-tool vocabulary.

        Aligned with the XFA tool (extract_xfa_fields) so callers see the same
        six core terms regardless of which form system the PDF uses:
        text / checkbox / radio / dropdown / date / signature, plus
        button / unknown for edge cases. Notably, listbox + combobox both
        collapse to "dropdown" — the distinction is widget-hover behavior, not
        semantic field type, and callers asking "is this a dropdown?" don't
        care which one it is.
        """
        field_type = getattr(widget, 'field_type', 0)

        # Field type constants from PyMuPDF
        if field_type == fitz.PDF_WIDGET_TYPE_BUTTON:
            return "button"
        elif field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
            return "checkbox"
        elif field_type == fitz.PDF_WIDGET_TYPE_RADIOBUTTON:
            return "radio"
        elif field_type == fitz.PDF_WIDGET_TYPE_TEXT:
            return "text"
        elif field_type == fitz.PDF_WIDGET_TYPE_LISTBOX:
            return "dropdown"
        elif field_type == fitz.PDF_WIDGET_TYPE_COMBOBOX:
            return "dropdown"
        elif field_type == fitz.PDF_WIDGET_TYPE_SIGNATURE:
            return "signature"
        else:
            return "unknown"

    @mcp_tool(
        name="is_xfa_pdf",
        description=(
            "Detect whether a PDF is XFA (Adobe LiveCycle / dynamic forms) "
            "and whether it's dynamic or static. Dynamic XFA forms can't be "
            "rendered by any open-source PDF library — only Adobe's runtime "
            "executes them. Use this to branch BEFORE calling extract_form_data "
            "or convert_to_images on a PDF that might be dynamic XFA. Returns "
            "{is_xfa, xfa_type: 'dynamic'|'static'|None, has_acroform}."
        )
    )
    async def is_xfa_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Detect XFA presence and type (dynamic vs static).

        Args:
            pdf_path: Path to PDF file or HTTPS URL.

        Returns:
            Dict with is_xfa (bool), xfa_type ("dynamic" / "static" / None),
            and has_acroform (bool).
        """
        start_time = time.time()
        try:
            path = await validate_pdf_path(pdf_path)
            result = _detect_xfa(str(path))
            result["detection_time"] = round(time.time() - start_time, 2)
            return result
        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"XFA detection failed: {error_msg}")
            return {
                "is_xfa": False,
                "xfa_type": None,
                "has_acroform": False,
                "error": error_msg,
                "detection_time": round(time.time() - start_time, 2),
            }

    @mcp_tool(
        name="extract_xfa_fields",
        description=(
            "Extract the XFA template field schema (names, captions, UI types) "
            "from a dynamic XFA PDF. Recovers form structure that extract_form_data "
            "can't reach because the fields aren't in AcroForm. Uses the zipForm "
            "producer profile by default; pass profile='generic' for forms from "
            "other producers and supply extra_plumbing_patterns / "
            "extra_positional_patterns. Returns shared (canonical) + positional "
            "+ plumbing-dropped field breakdown. The `original` XFA name is on "
            "every field — that's the round-trip key for actually filling the form."
        )
    )
    async def extract_xfa_fields(
        self,
        pdf_path: str,
        profile: str = "zipform",
        extra_plumbing_exact: Optional[List[str]] = None,
        extra_plumbing_patterns: Optional[List[str]] = None,
        extra_positional_patterns: Optional[List[str]] = None,
        canonical_separator: str = "_",
        include_design_time_bbox: bool = False,
    ) -> Dict[str, Any]:
        """Extract the XFA field schema from a dynamic-XFA PDF.

        Args:
            pdf_path: Path to PDF file or HTTPS URL.
            profile: Producer profile — "zipform" (Lone Wolf / zipForm Plus
                conventions) or "generic" (only the Global_Info- shared-prefix
                convention; callers add producer-specific patterns themselves).
            extra_plumbing_exact: Additional exact field names to drop as plumbing.
            extra_plumbing_patterns: Regex patterns (strings) for additional
                plumbing fields. Matched case-insensitively.
            extra_positional_patterns: Regex patterns (strings) identifying
                additional opaque positional codes.
            canonical_separator: Separator for canonical names — "_" (snake,
                default), "." (dotted), "-" (kebab).
            include_design_time_bbox: Include best-effort design-time geometry
                on every field. NOT authoritative for dynamic XFA — coordinates
                reflow at Adobe render time. Useful as a hint, not a source of
                truth.

        Returns:
            Dict matching the response shape in
            docs/agent-threads/xfa-form-support/004-*.md — is_xfa, xfa_type,
            xfa_parts, field_count, fields (with original on every entry,
            canonical_name only on shared), shared_fields, plumbing_fields_dropped,
            profile_used, warnings.
        """
        start_time = time.time()
        try:
            path = await validate_pdf_path(pdf_path)
            result = extract_xfa_schema(
                str(path),
                profile=profile,
                extra_plumbing_exact=extra_plumbing_exact,
                extra_plumbing_patterns=extra_plumbing_patterns,
                extra_positional_patterns=extra_positional_patterns,
                canonical_separator=canonical_separator,
                include_design_time_bbox=include_design_time_bbox,
            )
            result["extraction_time"] = round(time.time() - start_time, 2)
            return result
        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"XFA field extraction failed: {error_msg}")
            return {
                "error": error_msg,
                "extraction_time": round(time.time() - start_time, 2),
            }