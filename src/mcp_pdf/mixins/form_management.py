"""
Form Management Mixin - PDF form creation, filling, and data extraction
"""

import json
import time
from collections import defaultdict
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


class FormManagementMixin(MCPMixin):
    """
    Handles all PDF form creation, filling, and management operations.

    Tools provided:
    - extract_form_data: Extract form fields and their values
    - fill_form_pdf: Fill existing PDF forms with data
    - create_form_pdf: Create new interactive PDF forms
    """

    def get_mixin_name(self) -> str:
        return "FormManagement"

    def get_required_permissions(self) -> List[str]:
        return ["read_files", "write_files", "form_processing"]

    def _setup(self):
        """Initialize form management specific configuration"""
        self.supported_page_sizes = ["A4", "Letter", "Legal"]
        self.max_fields_per_form = 100

    @mcp_tool(
        name="extract_form_data",
        description="Extract form fields and their values from PDF forms"
    )
    async def extract_form_data(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract form fields and their values from PDF forms.

        Args:
            pdf_path: Path to PDF file or URL

        Returns:
            Dictionary containing form data
        """
        start_time = time.time()

        try:
            # Validate inputs using centralized security functions
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            form_data = {
                "has_forms": False,
                "form_fields": [],
                "form_summary": {},
                "extraction_time": 0
            }

            # Check if document has forms
            if doc.is_form_pdf:
                form_data["has_forms"] = True

                # Extract form fields
                fields_by_type = defaultdict(int)

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    widgets = page.widgets()

                    for widget in widgets:
                        field_info = {
                            "page": page_num + 1,
                            "field_name": widget.field_name or f"unnamed_field_{len(form_data['form_fields'])}",
                            "field_type": widget.field_type_string,
                            "field_value": widget.field_value,
                            "is_required": widget.field_flags & 2 != 0,
                            "is_readonly": widget.field_flags & 1 != 0,
                            "coordinates": {
                                "x0": widget.rect.x0,
                                "y0": widget.rect.y0,
                                "x1": widget.rect.x1,
                                "y1": widget.rect.y1
                            }
                        }

                        # Count field types
                        fields_by_type[widget.field_type_string] += 1
                        form_data["form_fields"].append(field_info)

                # Create summary
                form_data["form_summary"] = {
                    "total_fields": len(form_data["form_fields"]),
                    "fields_by_type": dict(fields_by_type),
                    "pages_with_forms": len(set(field["page"] for field in form_data["form_fields"]))
                }

            form_data["extraction_time"] = round(time.time() - start_time, 2)
            doc.close()

            return {
                "success": True,
                **form_data
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
        description="Fill an existing PDF form with provided data"
    )
    async def fill_form_pdf(
        self,
        input_path: str,
        output_path: str,
        form_data: str,  # JSON string of field values
        flatten: bool = False  # Whether to flatten form (make non-editable)
    ) -> Dict[str, Any]:
        """
        Fill an existing PDF form with provided data.

        Args:
            input_path: Path to the PDF form to fill
            output_path: Path where filled PDF should be saved
            form_data: JSON string of field names and values {"field_name": "value"}
            flatten: Whether to flatten the form (make fields non-editable)

        Returns:
            Dictionary containing filling results
        """
        start_time = time.time()

        try:
            # Parse form data
            try:
                field_values = self._safe_json_parse(form_data) if form_data else {}
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid form data JSON: {str(e)}",
                    "fill_time": 0
                }

            # Validate paths
            input_file = await validate_pdf_path(input_path)
            output_file = validate_output_path(output_path)

            doc = fitz.open(str(input_file))

            if not doc.is_form_pdf:
                doc.close()
                return {
                    "success": False,
                    "error": "Input PDF is not a form document",
                    "fill_time": 0
                }

            filled_fields = []
            failed_fields = []

            # Fill form fields
            for field_name, field_value in field_values.items():
                try:
                    # Find the field and set its value
                    field_found = False
                    for page_num in range(len(doc)):
                        page = doc[page_num]

                        for widget in page.widgets():
                            if widget.field_name == field_name:
                                field_found = True

                                # Handle different field types
                                if widget.field_type == fitz.PDF_WIDGET_TYPE_TEXT:
                                    widget.field_value = str(field_value)
                                    widget.update()
                                elif widget.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                                    widget.field_value = bool(field_value)
                                    widget.update()
                                elif widget.field_type == fitz.PDF_WIDGET_TYPE_RADIOBUTTON:
                                    widget.field_value = str(field_value)
                                    widget.update()
                                elif widget.field_type == fitz.PDF_WIDGET_TYPE_LISTBOX:
                                    widget.field_value = str(field_value)
                                    widget.update()

                                filled_fields.append({
                                    "field_name": field_name,
                                    "field_value": field_value,
                                    "field_type": widget.field_type_string,
                                    "page": page_num + 1
                                })
                                break

                    if not field_found:
                        failed_fields.append({
                            "field_name": field_name,
                            "reason": "Field not found in document"
                        })

                except Exception as e:
                    failed_fields.append({
                        "field_name": field_name,
                        "reason": f"Error setting value: {str(e)}"
                    })

            # Flatten form if requested
            if flatten:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    widgets = page.widgets()
                    for widget in widgets:
                        widget.field_flags |= fitz.PDF_FIELD_IS_READ_ONLY

            # Save the filled form
            doc.save(str(output_file))
            doc.close()

            return {
                "success": True,
                "output_path": str(output_file),
                "fields_filled": len(filled_fields),
                "fields_failed": len(failed_fields),
                "filled_fields": filled_fields,
                "failed_fields": failed_fields,
                "form_flattened": flatten,
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
        description="Create a new PDF form with interactive fields"
    )
    async def create_form_pdf(
        self,
        output_path: str,
        title: str = "Form Document",
        page_size: str = "A4",  # A4, Letter, Legal
        fields: str = "[]"  # JSON string of field definitions
    ) -> Dict[str, Any]:
        """
        Create a new PDF form with interactive fields.

        Args:
            output_path: Path where the PDF form should be saved
            title: Title of the form document
            page_size: Page size (A4, Letter, Legal)
            fields: JSON string containing field definitions

        Field format:
        [
            {
                "type": "text|checkbox|radio|dropdown|signature",
                "name": "field_name",
                "label": "Field Label",
                "x": 100, "y": 700, "width": 200, "height": 20,
                "required": true,
                "default_value": "",
                "options": ["opt1", "opt2"]  // for dropdown/radio
            }
        ]

        Returns:
            Dictionary containing creation results
        """
        start_time = time.time()

        try:
            # Parse field definitions
            try:
                field_definitions = self._safe_json_parse(fields) if fields != "[]" else []
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid field JSON: {str(e)}",
                    "creation_time": 0
                }

            # Validate output path
            output_file = validate_output_path(output_path)

            # Page size mapping
            page_sizes = {
                "A4": fitz.paper_rect("A4"),
                "Letter": fitz.paper_rect("letter"),
                "Legal": fitz.paper_rect("legal")
            }

            if page_size not in page_sizes:
                return {
                    "success": False,
                    "error": f"Unsupported page size: {page_size}. Use A4, Letter, or Legal",
                    "creation_time": 0
                }

            # Create new document
            doc = fitz.open()
            page = doc.new_page(width=page_sizes[page_size].width, height=page_sizes[page_size].height)

            # Set document metadata
            doc.set_metadata({
                "title": title,
                "creator": "MCP PDF Tools",
                "producer": "FastMCP Server"
            })

            created_fields = []
            field_errors = []

            # Add fields to the form
            for i, field_def in enumerate(field_definitions):
                try:
                    field_type = field_def.get("type", "text")
                    field_name = field_def.get("name", f"field_{i}")
                    field_label = field_def.get("label", field_name)
                    x = field_def.get("x", 100)
                    y = field_def.get("y", 700 - i * 30)
                    width = field_def.get("width", 200)
                    height = field_def.get("height", 20)
                    required = field_def.get("required", False)
                    default_value = field_def.get("default_value", "")

                    # Create field rectangle
                    field_rect = fitz.Rect(x, y, x + width, y + height)

                    # Add label text
                    label_rect = fitz.Rect(x, y - 15, x + width, y)
                    page.insert_text(label_rect.tl, field_label, fontsize=10)

                    # Create widget based on type
                    if field_type == "text":
                        widget = page.add_widget(fitz.Widget.TYPE_TEXT, field_rect)
                        widget.field_name = field_name
                        widget.field_value = default_value
                        if required:
                            widget.field_flags |= fitz.PDF_FIELD_IS_REQUIRED

                    elif field_type == "checkbox":
                        widget = page.add_widget(fitz.Widget.TYPE_CHECKBOX, field_rect)
                        widget.field_name = field_name
                        widget.field_value = bool(default_value)
                        if required:
                            widget.field_flags |= fitz.PDF_FIELD_IS_REQUIRED

                    else:
                        field_errors.append({
                            "field_name": field_name,
                            "error": f"Unsupported field type: {field_type}"
                        })
                        continue

                    widget.update()
                    created_fields.append({
                        "name": field_name,
                        "type": field_type,
                        "position": {"x": x, "y": y, "width": width, "height": height}
                    })

                except Exception as e:
                    field_errors.append({
                        "field_name": field_def.get("name", f"field_{i}"),
                        "error": str(e)
                    })

            # Save the form
            doc.save(str(output_file))
            doc.close()

            return {
                "success": True,
                "output_path": str(output_file),
                "form_title": title,
                "page_size": page_size,
                "fields_created": len(created_fields),
                "field_errors": len(field_errors),
                "created_fields": created_fields,
                "errors": field_errors,
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

    # Private helper methods (synchronous for proper async pattern)
    def _safe_json_parse(self, json_str: str, max_size: int = MAX_JSON_SIZE) -> dict:
        """Safely parse JSON with size limits"""
        if not json_str:
            return {}

        if len(json_str) > max_size:
            raise ValueError(f"JSON input too large: {len(json_str)} > {max_size}")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")