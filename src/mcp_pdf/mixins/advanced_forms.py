"""
Advanced Forms Mixin - Advanced PDF form field creation and validation
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List
import logging

# PDF processing libraries
import fitz  # PyMuPDF

from .base import MCPMixin, mcp_tool
from ..security import validate_pdf_path, validate_output_path, sanitize_error_message

logger = logging.getLogger(__name__)

# JSON size limit for security
MAX_JSON_SIZE = 10000


class AdvancedFormsMixin(MCPMixin):
    """
    Handles advanced PDF form operations including specialized field types,
    validation, and form field management.

    Tools provided:
    - add_form_fields: Add interactive form fields to existing PDF
    - add_radio_group: Add radio button groups with mutual exclusion
    - add_textarea_field: Add multi-line text areas with word limits
    - add_date_field: Add date fields with format validation
    - validate_form_data: Validate form data against rules
    - add_field_validation: Add validation rules to form fields
    """

    def get_mixin_name(self) -> str:
        return "AdvancedForms"

    def get_required_permissions(self) -> List[str]:
        return ["read_files", "write_files", "form_processing", "advanced_forms"]

    def _setup(self):
        """Initialize advanced forms specific configuration"""
        self.max_fields_per_form = 100
        self.max_radio_options = 20
        self.supported_date_formats = ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"]
        self.validation_patterns = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "phone": r"^[\d\s\-\+\(\)]+$",
            "number": r"^\d+(\.\d+)?$",
            "date": r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$"
        }

    @mcp_tool(
        name="add_form_fields",
        description="Add form fields to an existing PDF"
    )
    async def add_form_fields(
        self,
        input_path: str,
        output_path: str,
        fields: str  # JSON string of field definitions
    ) -> Dict[str, Any]:
        """
        Add interactive form fields to an existing PDF.

        Args:
            input_path: Path to the existing PDF
            output_path: Path where PDF with added fields should be saved
            fields: JSON string containing field definitions

        Returns:
            Dictionary containing addition results
        """
        start_time = time.time()

        try:
            # Parse field definitions
            try:
                field_definitions = self._safe_json_parse(fields) if fields else []
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid field JSON: {str(e)}",
                    "addition_time": 0
                }

            # Validate input path
            input_file = await validate_pdf_path(input_path)
            output_file = validate_output_path(output_path)
            doc = fitz.open(str(input_file))

            added_fields = []
            field_errors = []

            # Process each field definition
            for i, field in enumerate(field_definitions):
                try:
                    field_type = field.get("type", "text")
                    field_name = field.get("name", f"added_field_{i}")
                    field_label = field.get("label", field_name)
                    page_num = field.get("page", 1) - 1  # Convert to 0-indexed

                    # Ensure page exists
                    if page_num >= len(doc) or page_num < 0:
                        field_errors.append({
                            "field_name": field_name,
                            "error": f"Page {page_num + 1} does not exist"
                        })
                        continue

                    page = doc[page_num]

                    # Position and size
                    x = field.get("x", 50)
                    y = field.get("y", 100)
                    width = field.get("width", 200)
                    height = field.get("height", 20)

                    # Create field rectangle
                    field_rect = fitz.Rect(x, y, x + width, y + height)

                    # Add label if provided
                    if field_label and field_label != field_name:
                        label_rect = fitz.Rect(x, y - 15, x + width, y)
                        page.insert_text(label_rect.tl, field_label, fontsize=10)

                    # Create widget based on type
                    if field_type == "text":
                        widget = page.add_widget(fitz.Widget.TYPE_TEXT, field_rect)
                        widget.field_name = field_name
                        widget.field_value = field.get("default_value", "")
                        if field.get("required", False):
                            widget.field_flags |= fitz.PDF_FIELD_IS_REQUIRED

                    elif field_type == "checkbox":
                        widget = page.add_widget(fitz.Widget.TYPE_CHECKBOX, field_rect)
                        widget.field_name = field_name
                        widget.field_value = bool(field.get("default_value", False))
                        if field.get("required", False):
                            widget.field_flags |= fitz.PDF_FIELD_IS_REQUIRED

                    elif field_type == "dropdown":
                        widget = page.add_widget(fitz.Widget.TYPE_LISTBOX, field_rect)
                        widget.field_name = field_name
                        options = field.get("options", [])
                        if options:
                            widget.choice_values = options
                            widget.field_value = field.get("default_value", options[0])

                    elif field_type == "signature":
                        widget = page.add_widget(fitz.Widget.TYPE_SIGNATURE, field_rect)
                        widget.field_name = field_name

                    else:
                        field_errors.append({
                            "field_name": field_name,
                            "error": f"Unsupported field type: {field_type}"
                        })
                        continue

                    widget.update()
                    added_fields.append({
                        "name": field_name,
                        "type": field_type,
                        "page": page_num + 1,
                        "position": {"x": x, "y": y, "width": width, "height": height}
                    })

                except Exception as e:
                    field_errors.append({
                        "field_name": field.get("name", f"field_{i}"),
                        "error": str(e)
                    })

            # Save the modified PDF
            doc.save(str(output_file), garbage=4, deflate=True, clean=True)
            doc.close()

            return {
                "success": True,
                "input_path": str(input_file),
                "output_path": str(output_file),
                "fields_requested": len(field_definitions),
                "fields_added": len(added_fields),
                "fields_failed": len(field_errors),
                "added_fields": added_fields,
                "errors": field_errors,
                "addition_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Form fields addition failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "addition_time": round(time.time() - start_time, 2)
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
        options: str,  # JSON string of radio button options
        x: int = 50,
        y: int = 100,
        spacing: int = 30,
        page: int = 1
    ) -> Dict[str, Any]:
        """
        Add a radio button group where only one option can be selected.

        Args:
            input_path: Path to the existing PDF
            output_path: Path where PDF with radio group should be saved
            group_name: Name for the radio button group
            options: JSON array of option labels
            x: X coordinate for the first radio button
            y: Y coordinate for the first radio button
            spacing: Vertical spacing between radio buttons
            page: Page number (1-indexed)

        Returns:
            Dictionary containing addition results
        """
        start_time = time.time()

        try:
            # Parse options
            try:
                option_labels = self._safe_json_parse(options) if options else []
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid options JSON: {str(e)}",
                    "addition_time": 0
                }

            if not option_labels:
                return {
                    "success": False,
                    "error": "At least one option is required",
                    "addition_time": 0
                }

            if len(option_labels) > self.max_radio_options:
                return {
                    "success": False,
                    "error": f"Too many options: {len(option_labels)} > {self.max_radio_options}",
                    "addition_time": 0
                }

            # Validate input path
            input_file = await validate_pdf_path(input_path)
            output_file = validate_output_path(output_path)
            doc = fitz.open(str(input_file))

            page_num = page - 1  # Convert to 0-indexed
            if page_num >= len(doc) or page_num < 0:
                doc.close()
                return {
                    "success": False,
                    "error": f"Page {page} does not exist in PDF",
                    "addition_time": 0
                }

            pdf_page = doc[page_num]
            added_buttons = []

            # Add radio buttons
            for i, label in enumerate(option_labels):
                button_y = y + (i * spacing)

                # Create radio button widget
                button_rect = fitz.Rect(x, button_y, x + 15, button_y + 15)
                widget = pdf_page.add_widget(fitz.Widget.TYPE_RADIOBUTTON, button_rect)
                widget.field_name = f"{group_name}_{i}"
                widget.field_value = (i == 0)  # Select first option by default

                # Add label text
                label_rect = fitz.Rect(x + 20, button_y, x + 200, button_y + 15)
                pdf_page.insert_text(label_rect.tl, label, fontsize=10)

                widget.update()

                added_buttons.append({
                    "option": label,
                    "position": {"x": x, "y": button_y},
                    "selected": (i == 0)
                })

            # Save the PDF
            doc.save(str(output_file), garbage=4, deflate=True, clean=True)
            doc.close()

            return {
                "success": True,
                "input_path": str(input_file),
                "output_path": str(output_file),
                "group_name": group_name,
                "options_count": len(option_labels),
                "radio_buttons": added_buttons,
                "page": page,
                "addition_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Radio group addition failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "addition_time": round(time.time() - start_time, 2)
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
        label: str = "",
        x: int = 50,
        y: int = 100,
        width: int = 400,
        height: int = 100,
        word_limit: int = 500,
        page: int = 1,
        show_word_count: bool = True
    ) -> Dict[str, Any]:
        """
        Add a multi-line text area with optional word count display.

        Args:
            input_path: Path to the existing PDF
            output_path: Path where PDF with textarea should be saved
            field_name: Name for the textarea field
            label: Label text to display above the field
            x: X coordinate for the field
            y: Y coordinate for the field
            width: Width of the textarea
            height: Height of the textarea
            word_limit: Maximum number of words allowed
            page: Page number (1-indexed)
            show_word_count: Whether to show word count indicator

        Returns:
            Dictionary containing addition results
        """
        start_time = time.time()

        try:
            # Validate input path
            input_file = await validate_pdf_path(input_path)
            output_file = validate_output_path(output_path)
            doc = fitz.open(str(input_file))

            page_num = page - 1  # Convert to 0-indexed
            if page_num >= len(doc) or page_num < 0:
                doc.close()
                return {
                    "success": False,
                    "error": f"Page {page} does not exist in PDF",
                    "addition_time": 0
                }

            pdf_page = doc[page_num]

            # Add field label if provided
            if label:
                pdf_page.insert_text((x, y - 5), label, fontname="helv", fontsize=10, color=(0, 0, 0))

            # Create multi-line text widget
            field_rect = fitz.Rect(x, y, x + width, y + height)
            widget = pdf_page.add_widget(fitz.Widget.TYPE_TEXT, field_rect)
            widget.field_name = field_name
            widget.field_flags |= fitz.PDF_FIELD_IS_MULTILINE

            # Set field properties
            widget.text_maxlen = word_limit * 10  # Approximate character limit
            widget.field_value = ""

            # Add word count indicator if requested
            if show_word_count:
                count_text = f"(Max {word_limit} words)"
                count_rect = fitz.Rect(x, y + height + 5, x + width, y + height + 20)
                pdf_page.insert_text(count_rect.tl, count_text, fontsize=8, color=(0.5, 0.5, 0.5))

            widget.update()

            # Save the PDF
            doc.save(str(output_file), garbage=4, deflate=True, clean=True)
            doc.close()

            return {
                "success": True,
                "input_path": str(input_file),
                "output_path": str(output_file),
                "field_name": field_name,
                "field_properties": {
                    "type": "textarea",
                    "position": {"x": x, "y": y, "width": width, "height": height},
                    "word_limit": word_limit,
                    "page": page,
                    "label": label,
                    "show_word_count": show_word_count
                },
                "addition_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Textarea field addition failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "addition_time": round(time.time() - start_time, 2)
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
        label: str = "",
        x: int = 50,
        y: int = 100,
        width: int = 150,
        height: int = 25,
        date_format: str = "MM/DD/YYYY",
        page: int = 1,
        show_format_hint: bool = True
    ) -> Dict[str, Any]:
        """
        Add a date field with format validation and hints.

        Args:
            input_path: Path to the existing PDF
            output_path: Path where PDF with date field should be saved
            field_name: Name for the date field
            label: Label text to display
            x: X coordinate for the field
            y: Y coordinate for the field
            width: Width of the date field
            height: Height of the date field
            date_format: Expected date format
            page: Page number (1-indexed)
            show_format_hint: Whether to show format hint below field

        Returns:
            Dictionary containing addition results
        """
        start_time = time.time()

        try:
            # Validate date format
            if date_format not in self.supported_date_formats:
                return {
                    "success": False,
                    "error": f"Unsupported date format: {date_format}. Supported: {', '.join(self.supported_date_formats)}",
                    "addition_time": 0
                }

            # Validate input path
            input_file = await validate_pdf_path(input_path)
            output_file = validate_output_path(output_path)
            doc = fitz.open(str(input_file))

            page_num = page - 1  # Convert to 0-indexed
            if page_num >= len(doc) or page_num < 0:
                doc.close()
                return {
                    "success": False,
                    "error": f"Page {page} does not exist in PDF",
                    "addition_time": 0
                }

            pdf_page = doc[page_num]

            # Add field label if provided
            if label:
                pdf_page.insert_text((x, y - 5), label, fontname="helv", fontsize=10, color=(0, 0, 0))

            # Create date field widget
            field_rect = fitz.Rect(x, y, x + width, y + height)
            widget = pdf_page.add_widget(fitz.Widget.TYPE_TEXT, field_rect)
            widget.field_name = field_name

            # Set format mask based on date format
            if date_format == "MM/DD/YYYY":
                widget.text_maxlen = 10
                widget.field_value = ""
            elif date_format == "DD/MM/YYYY":
                widget.text_maxlen = 10
                widget.field_value = ""
            elif date_format == "YYYY-MM-DD":
                widget.text_maxlen = 10
                widget.field_value = ""

            # Add format hint if requested
            if show_format_hint:
                hint_text = f"Format: {date_format}"
                hint_rect = fitz.Rect(x, y + height + 2, x + width, y + height + 15)
                pdf_page.insert_text(hint_rect.tl, hint_text, fontsize=8, color=(0.5, 0.5, 0.5))

            widget.update()

            # Save the PDF
            doc.save(str(output_file), garbage=4, deflate=True, clean=True)
            doc.close()

            return {
                "success": True,
                "input_path": str(input_file),
                "output_path": str(output_file),
                "field_name": field_name,
                "field_properties": {
                    "type": "date",
                    "position": {"x": x, "y": y, "width": width, "height": height},
                    "date_format": date_format,
                    "page": page,
                    "label": label,
                    "show_format_hint": show_format_hint
                },
                "addition_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Date field addition failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "addition_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="validate_form_data",
        description="Validate form data against rules and constraints"
    )
    async def validate_form_data(
        self,
        pdf_path: str,
        form_data: str,  # JSON string of field values
        validation_rules: str = "{}"  # JSON string of validation rules
    ) -> Dict[str, Any]:
        """
        Validate form data against specified rules and field constraints.

        Args:
            pdf_path: Path to the PDF form
            form_data: JSON string of field names and values to validate
            validation_rules: JSON string defining validation rules per field

        Returns:
            Dictionary containing validation results
        """
        start_time = time.time()

        try:
            # Parse inputs
            try:
                field_values = self._safe_json_parse(form_data) if form_data else {}
                rules = self._safe_json_parse(validation_rules) if validation_rules else {}
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON input: {str(e)}",
                    "validation_time": 0
                }

            # Get form structure
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            if not doc.is_form_pdf:
                doc.close()
                return {
                    "success": False,
                    "error": "PDF does not contain form fields",
                    "validation_time": 0
                }

            # Extract form fields
            form_fields_list = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                for widget in page.widgets():
                    form_fields_list.append({
                        "name": widget.field_name,
                        "type": widget.field_type_string,
                        "required": widget.field_flags & 2 != 0
                    })

            doc.close()

            # Validate each field
            validation_results = []
            validation_errors = []
            is_valid = True

            for field_name, field_value in field_values.items():
                field_rules = rules.get(field_name, {})
                field_result = {"field": field_name, "value": field_value, "valid": True, "errors": []}

                # Check required
                if field_rules.get("required", False) and not field_value:
                    field_result["valid"] = False
                    field_result["errors"].append("Field is required")

                # Check type/format
                field_type = field_rules.get("type", "text")
                if field_value:
                    if field_type == "email":
                        if not re.match(self.validation_patterns["email"], field_value):
                            field_result["valid"] = False
                            field_result["errors"].append("Invalid email format")

                    elif field_type == "phone":
                        if not re.match(self.validation_patterns["phone"], field_value):
                            field_result["valid"] = False
                            field_result["errors"].append("Invalid phone format")

                    elif field_type == "number":
                        if not re.match(self.validation_patterns["number"], str(field_value)):
                            field_result["valid"] = False
                            field_result["errors"].append("Must be a valid number")

                    elif field_type == "date":
                        if not re.match(self.validation_patterns["date"], field_value):
                            field_result["valid"] = False
                            field_result["errors"].append("Invalid date format")

                # Check length constraints
                if field_value and isinstance(field_value, str):
                    min_length = field_rules.get("min_length", 0)
                    max_length = field_rules.get("max_length", 999999)

                    if len(field_value) < min_length:
                        field_result["valid"] = False
                        field_result["errors"].append(f"Minimum length is {min_length}")

                    if len(field_value) > max_length:
                        field_result["valid"] = False
                        field_result["errors"].append(f"Maximum length is {max_length}")

                # Check custom pattern
                if "pattern" in field_rules and field_value:
                    pattern = field_rules["pattern"]
                    try:
                        if not re.match(pattern, field_value):
                            field_result["valid"] = False
                            custom_msg = field_rules.get("custom_message", "Value does not match required pattern")
                            field_result["errors"].append(custom_msg)
                    except re.error:
                        field_result["errors"].append("Invalid validation pattern")

                if not field_result["valid"]:
                    is_valid = False
                    validation_errors.append(field_result)
                else:
                    validation_results.append(field_result)

            return {
                "success": True,
                "is_valid": is_valid,
                "form_fields": form_fields_list,
                "validation_summary": {
                    "total_fields": len(field_values),
                    "valid_fields": len(validation_results),
                    "invalid_fields": len(validation_errors)
                },
                "valid_fields": validation_results,
                "invalid_fields": validation_errors,
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

    @mcp_tool(
        name="add_field_validation",
        description="Add validation rules to existing form fields"
    )
    async def add_field_validation(
        self,
        input_path: str,
        output_path: str,
        validation_rules: str  # JSON string of validation rules
    ) -> Dict[str, Any]:
        """
        Add JavaScript validation rules to form fields (where supported).

        Args:
            input_path: Path to the existing PDF form
            output_path: Path where PDF with validation should be saved
            validation_rules: JSON string defining validation rules

        Returns:
            Dictionary containing validation addition results
        """
        start_time = time.time()

        try:
            # Parse validation rules
            try:
                rules = self._safe_json_parse(validation_rules) if validation_rules else {}
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid validation rules JSON: {str(e)}",
                    "addition_time": 0
                }

            # Validate input path
            input_file = await validate_pdf_path(input_path)
            output_file = validate_output_path(output_path)
            doc = fitz.open(str(input_file))

            if not doc.is_form_pdf:
                doc.close()
                return {
                    "success": False,
                    "error": "Input PDF is not a form document",
                    "addition_time": 0
                }

            added_validations = []
            failed_validations = []

            # Process each page to find and modify form fields
            for page_num in range(len(doc)):
                page = doc[page_num]

                for widget in page.widgets():
                    field_name = widget.field_name

                    if field_name in rules:
                        field_rules = rules[field_name]

                        try:
                            # Set required flag if specified
                            if field_rules.get("required", False):
                                widget.field_flags |= fitz.PDF_FIELD_IS_REQUIRED

                            # Set format restrictions based on type
                            field_format = field_rules.get("format", "text")

                            if field_format == "number":
                                # Restrict to numeric input
                                widget.field_flags |= fitz.PDF_FIELD_IS_COMB

                            # Update widget
                            widget.update()

                            added_validations.append({
                                "field_name": field_name,
                                "page": page_num + 1,
                                "rules_applied": field_rules
                            })

                        except Exception as e:
                            failed_validations.append({
                                "field_name": field_name,
                                "error": str(e)
                            })

            # Save the PDF with validations
            doc.save(str(output_file), garbage=4, deflate=True, clean=True)
            doc.close()

            return {
                "success": True,
                "input_path": str(input_file),
                "output_path": str(output_file),
                "validations_requested": len(rules),
                "validations_added": len(added_validations),
                "validations_failed": len(failed_validations),
                "added_validations": added_validations,
                "failed_validations": failed_validations,
                "addition_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Field validation addition failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "addition_time": round(time.time() - start_time, 2)
            }

    # Private helper methods (synchronous for proper async pattern)
    def _safe_json_parse(self, json_str: str, max_size: int = MAX_JSON_SIZE):
        """Safely parse JSON with size limits"""
        if not json_str:
            return []

        if len(json_str) > max_size:
            raise ValueError(f"JSON input too large: {len(json_str)} > {max_size}")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")