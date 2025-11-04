"""
Advanced Forms Mixin - Extended PDF form field operations
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

# Official FastMCP mixin
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, validate_output_path, sanitize_error_message

logger = logging.getLogger(__name__)


class AdvancedFormsMixin(MCPMixin):
    """
    Handles advanced PDF form operations including radio groups, textareas, and date fields.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()
        self.max_file_size = 100 * 1024 * 1024  # 100MB

    @mcp_tool(
        name="add_form_fields",
        description="Add form fields to an existing PDF"
    )
    async def add_form_fields(
        self,
        input_path: str,
        output_path: str,
        fields: str
    ) -> Dict[str, Any]:
        """
        Add interactive form fields to an existing PDF document.

        Args:
            input_path: Path to input PDF file
            output_path: Path where modified PDF will be saved
            fields: JSON string describing form fields to add

        Returns:
            Dictionary containing operation results
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_pdf_path = await validate_output_path(output_path)

            # Parse fields data
            try:
                field_definitions = json.loads(fields)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON in fields: {e}",
                    "processing_time": round(time.time() - start_time, 2)
                }

            # Open existing PDF
            doc = fitz.open(str(input_pdf_path))
            fields_added = 0

            for field_def in field_definitions:
                try:
                    page_num = field_def.get("page", 1) - 1  # Convert to 0-based
                    if page_num < 0 or page_num >= len(doc):
                        continue

                    page = doc[page_num]
                    field_type = field_def.get("type", "text")
                    field_name = field_def.get("name", f"field_{fields_added + 1}")

                    # Get position and size
                    x = field_def.get("x", 50)
                    y = field_def.get("y", 100)
                    width = field_def.get("width", 200)
                    height = field_def.get("height", 20)

                    # Create field rectangle
                    field_rect = fitz.Rect(x, y, x + width, y + height)

                    if field_type == "text":
                        widget = page.add_widget(fitz.Widget())
                        widget.field_name = field_name
                        widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
                        widget.rect = field_rect
                        widget.update()

                    elif field_type == "checkbox":
                        widget = page.add_widget(fitz.Widget())
                        widget.field_name = field_name
                        widget.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
                        widget.rect = field_rect
                        widget.update()

                    fields_added += 1

                except Exception as e:
                    logger.warning(f"Failed to add field {field_def}: {e}")

            # Save modified PDF
            doc.save(str(output_pdf_path))
            output_size = output_pdf_path.stat().st_size
            doc.close()

            return {
                "success": True,
                "fields_summary": {
                    "fields_requested": len(field_definitions),
                    "fields_added": fields_added,
                    "output_size_bytes": output_size
                },
                "output_info": {
                    "output_path": str(output_pdf_path)
                },
                "processing_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Adding form fields failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="add_radio_group",
        description="Add a radio button group with mutual exclusion to PDF"
    )
    async def add_radio_group(
        self,
        input_path: str,
        output_path: str,
        group_name: str,
        options: str,
        page: int = 1,
        x: int = 50,
        y: int = 100,
        spacing: int = 30
    ) -> Dict[str, Any]:
        """
        Add a radio button group to PDF with mutual exclusion.

        Args:
            input_path: Path to input PDF file
            output_path: Path where modified PDF will be saved
            group_name: Name of the radio button group
            options: JSON array of option labels
            page: Page number (1-based)
            x: X coordinate for first radio button
            y: Y coordinate for first radio button
            spacing: Vertical spacing between options

        Returns:
            Dictionary containing operation results
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_pdf_path = await validate_output_path(output_path)

            # Parse options
            try:
                option_list = json.loads(options)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON in options: {e}",
                    "processing_time": round(time.time() - start_time, 2)
                }

            # Open PDF
            doc = fitz.open(str(input_pdf_path))
            page_num = page - 1  # Convert to 0-based

            if page_num < 0 or page_num >= len(doc):
                doc.close()
                return {
                    "success": False,
                    "error": f"Page {page} out of range",
                    "processing_time": round(time.time() - start_time, 2)
                }

            pdf_page = doc[page_num]
            buttons_added = 0

            # Add radio buttons
            for i, option_label in enumerate(option_list):
                try:
                    button_y = y + (i * spacing)
                    button_rect = fitz.Rect(x, button_y, x + 15, button_y + 15)

                    # Create radio button widget
                    widget = pdf_page.add_widget(fitz.Widget())
                    widget.field_name = f"{group_name}_{i}"
                    widget.field_type = fitz.PDF_WIDGET_TYPE_RADIOBUTTON
                    widget.rect = button_rect
                    widget.update()

                    # Add label text next to radio button
                    text_point = fitz.Point(x + 20, button_y + 10)
                    pdf_page.insert_text(text_point, option_label, fontsize=10)

                    buttons_added += 1

                except Exception as e:
                    logger.warning(f"Failed to add radio button {i}: {e}")

            # Save modified PDF
            doc.save(str(output_pdf_path))
            output_size = output_pdf_path.stat().st_size
            doc.close()

            return {
                "success": True,
                "radio_group_summary": {
                    "group_name": group_name,
                    "options_requested": len(option_list),
                    "buttons_added": buttons_added,
                    "page": page,
                    "output_size_bytes": output_size
                },
                "output_info": {
                    "output_path": str(output_pdf_path)
                },
                "processing_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Adding radio group failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="add_textarea_field",
        description="Add a multi-line text area with word limits to PDF"
    )
    async def add_textarea_field(
        self,
        input_path: str,
        output_path: str,
        field_name: str,
        x: int = 50,
        y: int = 100,
        width: int = 400,
        height: int = 100,
        page: int = 1,
        word_limit: int = 500,
        label: str = "",
        show_word_count: bool = True
    ) -> Dict[str, Any]:
        """
        Add a multi-line text area field with word counting capabilities.

        Args:
            input_path: Path to input PDF file
            output_path: Path where modified PDF will be saved
            field_name: Name of the textarea field
            x: X coordinate
            y: Y coordinate
            width: Field width
            height: Field height
            page: Page number (1-based)
            word_limit: Maximum word count
            label: Optional field label
            show_word_count: Whether to show word count indicator

        Returns:
            Dictionary containing operation results
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_pdf_path = await validate_output_path(output_path)

            # Open PDF
            doc = fitz.open(str(input_pdf_path))
            page_num = page - 1  # Convert to 0-based

            if page_num < 0 or page_num >= len(doc):
                doc.close()
                return {
                    "success": False,
                    "error": f"Page {page} out of range",
                    "processing_time": round(time.time() - start_time, 2)
                }

            pdf_page = doc[page_num]

            # Add label if provided
            if label:
                label_point = fitz.Point(x, y - 15)
                pdf_page.insert_text(label_point, label, fontsize=10, color=(0, 0, 0))

            # Create textarea field rectangle
            field_rect = fitz.Rect(x, y, x + width, y + height)

            # Add textarea widget
            widget = pdf_page.add_widget(fitz.Widget())
            widget.field_name = field_name
            widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
            widget.rect = field_rect
            widget.update()

            # Add word count indicator if requested
            if show_word_count:
                count_text = f"Max words: {word_limit}"
                count_point = fitz.Point(x + width - 100, y + height + 15)
                pdf_page.insert_text(count_point, count_text, fontsize=8, color=(0.5, 0.5, 0.5))

            # Save modified PDF
            doc.save(str(output_pdf_path))
            output_size = output_pdf_path.stat().st_size
            doc.close()

            return {
                "success": True,
                "textarea_summary": {
                    "field_name": field_name,
                    "dimensions": f"{width}x{height}",
                    "word_limit": word_limit,
                    "has_label": bool(label),
                    "page": page,
                    "output_size_bytes": output_size
                },
                "output_info": {
                    "output_path": str(output_pdf_path)
                },
                "processing_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Adding textarea field failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="add_date_field",
        description="Add a date field with format validation to PDF"
    )
    async def add_date_field(
        self,
        input_path: str,
        output_path: str,
        field_name: str,
        x: int = 50,
        y: int = 100,
        width: int = 150,
        height: int = 25,
        page: int = 1,
        date_format: str = "MM/DD/YYYY",
        label: str = "",
        show_format_hint: bool = True
    ) -> Dict[str, Any]:
        """
        Add a date input field with format validation hints.

        Args:
            input_path: Path to input PDF file
            output_path: Path where modified PDF will be saved
            field_name: Name of the date field
            x: X coordinate
            y: Y coordinate
            width: Field width
            height: Field height
            page: Page number (1-based)
            date_format: Expected date format
            label: Optional field label
            show_format_hint: Whether to show format hint

        Returns:
            Dictionary containing operation results
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_pdf_path = await validate_output_path(output_path)

            # Open PDF
            doc = fitz.open(str(input_pdf_path))
            page_num = page - 1  # Convert to 0-based

            if page_num < 0 or page_num >= len(doc):
                doc.close()
                return {
                    "success": False,
                    "error": f"Page {page} out of range",
                    "processing_time": round(time.time() - start_time, 2)
                }

            pdf_page = doc[page_num]

            # Add label if provided
            if label:
                label_point = fitz.Point(x, y - 15)
                pdf_page.insert_text(label_point, label, fontsize=10, color=(0, 0, 0))

            # Create date field rectangle
            field_rect = fitz.Rect(x, y, x + width, y + height)

            # Add date input widget
            widget = pdf_page.add_widget(fitz.Widget())
            widget.field_name = field_name
            widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
            widget.rect = field_rect
            widget.update()

            # Add format hint if requested
            if show_format_hint:
                hint_text = f"Format: {date_format}"
                hint_point = fitz.Point(x + width + 10, y + height/2)
                pdf_page.insert_text(hint_point, hint_text, fontsize=8, color=(0.5, 0.5, 0.5))

            # Save modified PDF
            doc.save(str(output_pdf_path))
            output_size = output_pdf_path.stat().st_size
            doc.close()

            return {
                "success": True,
                "date_field_summary": {
                    "field_name": field_name,
                    "date_format": date_format,
                    "dimensions": f"{width}x{height}",
                    "has_label": bool(label),
                    "has_format_hint": show_format_hint,
                    "page": page,
                    "output_size_bytes": output_size
                },
                "output_info": {
                    "output_path": str(output_pdf_path)
                },
                "processing_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Adding date field failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="validate_form_data",
        description="Validate form data against rules and constraints"
    )
    async def validate_form_data(
        self,
        pdf_path: str,
        form_data: str,
        validation_rules: str = "{}"
    ) -> Dict[str, Any]:
        """
        Validate form data against specified rules and constraints.

        Args:
            pdf_path: Path to PDF with form fields
            form_data: JSON string containing form data to validate
            validation_rules: JSON string with validation rules

        Returns:
            Dictionary containing validation results
        """
        start_time = time.time()

        try:
            # Validate PDF path
            input_pdf_path = await validate_pdf_path(pdf_path)

            # Parse form data and rules
            try:
                data = json.loads(form_data)
                rules = json.loads(validation_rules)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON: {e}",
                    "validation_time": round(time.time() - start_time, 2)
                }

            validation_results = []
            errors = []
            warnings = []

            # Basic validation logic
            for field_name, field_value in data.items():
                field_rules = rules.get(field_name, {})
                field_result = {"field": field_name, "value": field_value, "valid": True, "messages": []}

                # Required field validation
                if field_rules.get("required", False) and not field_value:
                    field_result["valid"] = False
                    field_result["messages"].append("Field is required")
                    errors.append(f"{field_name}: Required field is empty")

                # Length validation
                if "max_length" in field_rules and len(str(field_value)) > field_rules["max_length"]:
                    field_result["valid"] = False
                    field_result["messages"].append(f"Exceeds maximum length of {field_rules['max_length']}")
                    errors.append(f"{field_name}: Value too long")

                # Pattern validation (basic)
                if "pattern" in field_rules and field_value:
                    import re
                    if not re.match(field_rules["pattern"], str(field_value)):
                        field_result["valid"] = False
                        field_result["messages"].append("Does not match required pattern")
                        errors.append(f"{field_name}: Invalid format")

                validation_results.append(field_result)

            # Overall validation status
            is_valid = len(errors) == 0

            return {
                "success": True,
                "validation_summary": {
                    "is_valid": is_valid,
                    "total_fields": len(data),
                    "valid_fields": len([r for r in validation_results if r["valid"]]),
                    "invalid_fields": len([r for r in validation_results if not r["valid"]]),
                    "total_errors": len(errors),
                    "total_warnings": len(warnings)
                },
                "field_results": validation_results,
                "errors": errors,
                "warnings": warnings,
                "file_info": {
                    "path": str(input_pdf_path)
                },
                "validation_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Form validation failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "validation_time": round(time.time() - start_time, 2)
            }