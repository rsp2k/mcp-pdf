"""
Advanced Forms Mixin - Extended PDF form field operations
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import time
import json
from typing import Dict, Any
import logging

# PDF processing libraries
import pymupdf

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

    @mcp_tool(
        name="add_form_fields",
        description=(
            "Add interactive AcroForm widgets to an EXISTING PDF, writing a "
            "NEW PDF to output_path. Use create_form_pdf to build a form on a "
            "blank page instead, and add_radio_group / add_textarea_field / "
            "add_date_field for those single specialised widgets.\n"
            "\n"
            "IMPORTANT: only type \"text\" and type \"checkbox\" actually "
            "create a widget. Any other value (dropdown, radio, signature) "
            "still counts toward fields_added but produces NOTHING in the "
            "output, so verify with extract_form_data before trusting the "
            "count.\n"
            "\n"
            "`fields` is a JSON array of objects:\n"
            '  [{"page": 1, "type": "text", "name": "applicant", "x": 72, '
            '"y": 700, "width": 220, "height": 18}]\n'
            "\n"
            "page is 1-based (default 1); a page outside the document is "
            "skipped SILENTLY, with no entry in any error list. type is text "
            "| checkbox (default text). name defaults to field_<n> derived "
            "from the running added-count, so omitting it on several fields "
            "gives them generated names. x/y are the widget rect's LOWER-LEFT "
            "corner in PDF points from the page's BOTTOM-left origin "
            "(defaults 50/100); width defaults to 200, height to 20, and the "
            "rect spans (x, y) to (x+width, y+height). Note this is the "
            "opposite convention to fill_permit_form, which measures y from "
            "the TOP of the page."
        ),
        annotations={
            "readOnlyHint": False,       # writes a new PDF
            "destructiveHint": True,     # overwrites output_path if it exists
            "idempotentHint": True,      # same args produce the same output file
            "openWorldHint": True,       # input_path may be an HTTPS URL
        },
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
            input_path: Path to the source PDF, or an HTTPS URL to fetch.
            output_path: Where to write the modified PDF. Overwritten if it
                already exists; the source is never modified in place.
            fields: JSON array of field objects. Each may carry:
                - "page":   1-based page number (default 1). Out-of-range
                            pages are skipped without an error.
                - "type":   "text" or "checkbox" (default "text"). Any other
                            value creates no widget but is still counted.
                - "name":   AcroForm field name (default "field_<n>")
                - "x", "y": lower-left corner in PDF points from the page's
                            bottom-left origin (defaults 50, 100)
                - "width", "height": widget size in points (defaults 200, 20)
                Example: [{"page": 1, "type": "checkbox", "name": "agree",
                           "x": 72, "y": 120, "width": 12, "height": 12}]

        Returns:
            Dict with success, fields_requested / fields_added (which can
            differ silently), output size and the output path.
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_pdf_path = validate_output_path(output_path)

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
            doc = pymupdf.open(str(input_pdf_path))
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
                    field_rect = pymupdf.Rect(x, y, x + width, y + height)

                    # A Widget must be fully populated BEFORE add_widget:
                    # a bare Widget() has rect=None and add_widget raises
                    # AttributeError on it (PyMuPDF 1.28). Assigning the
                    # attributes to add_widget's return value, as this did
                    # until 2026-09-21, meant every call threw, got swallowed
                    # by the handler below, and produced a PDF with no fields
                    # while still reporting success.
                    if field_type == "text":
                        widget = pymupdf.Widget()
                        widget.field_name = field_name
                        widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
                        widget.rect = field_rect
                        page.add_widget(widget)

                    elif field_type == "checkbox":
                        widget = pymupdf.Widget()
                        widget.field_name = field_name
                        widget.field_type = pymupdf.PDF_WIDGET_TYPE_CHECKBOX
                        widget.rect = field_rect
                        page.add_widget(widget)

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
        description=(
            "Draw a vertical column of radio-button widgets with a text label "
            "beside each, writing a NEW PDF to output_path.\n"
            "\n"
            "The buttons ARE mutually exclusive: every widget in the group "
            "shares the one field name you pass as group_name, which is what "
            "makes a PDF reader allow only one selection, and each carries a "
            "distinct on-state named after its option label. So the field's "
            "value is the chosen label (\"Commercial\"), not a generic "
            "\"Yes\", and \"Off\" means nothing is selected yet.\n"
            "\n"
            "extract_form_data therefore reports one entry PER BUTTON, all "
            "sharing group_name; that is the correct PDF representation of a "
            "radio group, not a duplicate. To preselect an option, pass the "
            "label as the value for group_name in fill_form_pdf.\n"
            "\n"
            "`options` is a JSON array of label strings:\n"
            '  ["Residential", "Commercial", "Industrial"]\n'
            "\n"
            "Each button is a 15x15 point box whose lower-left corner is at "
            "(x, y + index * spacing). Because y grows UPWARD from the page's "
            "bottom-left origin, later options sit HIGHER on the page; pass a "
            "negative spacing to run top-to-bottom instead. The label is "
            "drawn 20 points to the right of each box in 10pt text. Buttons "
            "that fail are logged server-side and simply missing from "
            "buttons_added, so compare it against options_requested."
        ),
        annotations={
            "readOnlyHint": False,       # writes a new PDF
            "destructiveHint": True,     # overwrites output_path if it exists
            "idempotentHint": True,
            "openWorldHint": True,       # input_path may be an HTTPS URL
        },
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
        Add a column of radio-button widgets with labels.

        Args:
            input_path: Path to the source PDF, or an HTTPS URL to fetch.
            output_path: Where to write the modified PDF. Overwritten if it
                already exists; the source is never modified in place.
            group_name: The AcroForm field name shared by every button in the
                group. Sharing one name is what makes the buttons mutually
                exclusive, and the field's value is whichever option label is
                currently selected ("Off" when none is).
            options: JSON array of label strings, e.g.
                ["Residential", "Commercial"]. One 15x15 point button is
                drawn per entry, in array order.
            page: 1-based page number (default 1). Out of range is an error.
            x: Left edge of every button, in PDF points from the page's
                bottom-left origin (default 50).
            y: Bottom edge of the FIRST button, in PDF points from the page's
                bottom-left origin (default 100).
            spacing: Points added to y per option (default 30). Positive
                values stack options upward; pass a negative value to run
                down the page.

        Returns:
            Dict with success, group_name, options_requested, buttons_added,
            page, output size and the output path.
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_pdf_path = validate_output_path(output_path)

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
            doc = pymupdf.open(str(input_pdf_path))
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
            button_xrefs = []

            # Add radio buttons
            for i, option_label in enumerate(option_list):
                try:
                    button_y = y + (i * spacing)
                    button_rect = pymupdf.Rect(x, button_y, x + 15, button_y + 15)

                    # Populate the Widget before adding it; see the note in
                    # add_form_fields. A radio button additionally needs a
                    # string field_value set up front, or MuPDF raises
                    # "bad xref" while building its appearance stream.
                    #
                    # Every button in the group SHARES one field name. PDF
                    # enforces mutual exclusion only among widgets that share
                    # a name, so the previous f"{group_name}_{i}" made each
                    # button an independent checkbox and the group was not a
                    # group at all, despite the tool's name.
                    widget = pymupdf.Widget()
                    widget.field_name = group_name
                    widget.field_type = pymupdf.PDF_WIDGET_TYPE_RADIOBUTTON
                    widget.field_value = "Off"
                    widget.rect = button_rect
                    added = pdf_page.add_widget(widget)
                    # Remember the xref so the on-state can be renamed after
                    # the whole group exists (see below).
                    button_xrefs.append((added.xref, option_label))

                    # Add label text next to radio button
                    text_point = pymupdf.Point(x + 20, button_y + 10)
                    pdf_page.insert_text(text_point, option_label, fontsize=10)

                    buttons_added += 1

                except Exception as e:
                    logger.warning(f"Failed to add radio button {i}: {e}")

            # Give each button a DISTINCT on-state named after its option, so
            # the field's value says which one is selected. PyMuPDF's Widget
            # API gives every radio kid the same "/Yes" on-state and offers no
            # way to change it (setting field_value to the label up front
            # raises "bad xref"), so rewrite the appearance dictionary key
            # directly. Without this, sharing the field name would make the
            # buttons mutually exclusive but indistinguishable: selecting any
            # of them would just set the field to "Yes".
            for xref, label in button_xrefs:
                try:
                    kind, ap_n = doc.xref_get_key(xref, "AP/N")
                    if kind == "dict" and "/Yes" in ap_n:
                        doc.xref_set_key(xref, "AP/N", ap_n.replace("/Yes", f"/{label}"))
                    doc.xref_set_key(xref, "AS", "/Off")   # start unselected
                except Exception as exc:
                    logger.warning(
                        "Could not set on-state for radio option %r: %s", label, exc
                    )

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
        description=(
            "Add ONE large text widget to an existing PDF, writing a NEW PDF "
            "to output_path. Use add_form_fields to add several ordinary-sized "
            "fields in a single call.\n"
            "\n"
            "What it really produces is a plain AcroForm text widget of the "
            "given size. word_limit is NOT enforced anywhere: it only supplies "
            "the number in an optional grey \"Max words: N\" caption, and a "
            "reader will happily accept more. The widget is also not flagged "
            "multiline, so wrapping is up to the reader.\n"
            "\n"
            "Positioning, all in PDF points from the page's BOTTOM-left "
            "origin: the box spans (x, y) to (x+width, y+height). `label`, if "
            "non-empty, is drawn 15 points BELOW y, so it lands under the box "
            "rather than above it. The word-count caption is drawn 15 points "
            "ABOVE the box, near its right edge."
        ),
        annotations={
            "readOnlyHint": False,       # writes a new PDF
            "destructiveHint": True,     # overwrites output_path if it exists
            "idempotentHint": True,
            "openWorldHint": True,       # input_path may be an HTTPS URL
        },
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
        Add one large text widget, optionally captioned with a word limit.

        Args:
            input_path: Path to the source PDF, or an HTTPS URL to fetch.
            output_path: Where to write the modified PDF. Overwritten if it
                already exists; the source is never modified in place.
            field_name: AcroForm field name for the widget. Used verbatim.
            x: Left edge in PDF points from the page's bottom-left origin
                (default 50).
            y: Bottom edge in PDF points from the page's bottom-left origin
                (default 100).
            width: Box width in points (default 400).
            height: Box height in points (default 100).
            page: 1-based page number (default 1). Out of range is an error.
            word_limit: Number shown in the "Max words: N" caption (default
                500). Advisory only; nothing enforces it.
            label: Caption text drawn 15 points below y in 10pt black. Empty
                string (the default) draws nothing.
            show_word_count: Draw the grey "Max words: N" caption above the
                box (default True).

        Returns:
            Dict with success, field_name, dimensions, word_limit, has_label,
            page, output size and the output path.
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_pdf_path = validate_output_path(output_path)

            # Open PDF
            doc = pymupdf.open(str(input_pdf_path))
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
                label_point = pymupdf.Point(x, y - 15)
                pdf_page.insert_text(label_point, label, fontsize=10, color=(0, 0, 0))

            # Create textarea field rectangle
            field_rect = pymupdf.Rect(x, y, x + width, y + height)

            # Populate the Widget before adding it; see add_form_fields.
            widget = pymupdf.Widget()
            widget.field_name = field_name
            widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
            widget.rect = field_rect
            pdf_page.add_widget(widget)

            # Add word count indicator if requested
            if show_word_count:
                count_text = f"Max words: {word_limit}"
                count_point = pymupdf.Point(x + width - 100, y + height + 15)
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
        description=(
            "Add ONE text widget intended to hold a date, writing a NEW PDF "
            "to output_path.\n"
            "\n"
            "NO validation is applied, despite the name. date_format is free "
            "text interpolated verbatim into an optional grey \"Format: ...\" "
            "caption drawn to the RIGHT of the box; the widget itself is an "
            "ordinary AcroForm text field with no format action, so a reader "
            "accepts anything typed into it and fill_form_pdf accepts any "
            "string. If you need the value checked, run validate_form_data "
            "with a `pattern` rule separately.\n"
            "\n"
            "Positioning is in PDF points from the page's BOTTOM-left origin: "
            "the box spans (x, y) to (x+width, y+height), `label` is drawn 15 "
            "points BELOW y (under the box), and the format caption sits 10 "
            "points right of the box at half its height."
        ),
        annotations={
            "readOnlyHint": False,       # writes a new PDF
            "destructiveHint": True,     # overwrites output_path if it exists
            "idempotentHint": True,
            "openWorldHint": True,       # input_path may be an HTTPS URL
        },
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
        Add one text widget with an optional printed format hint.

        Args:
            input_path: Path to the source PDF, or an HTTPS URL to fetch.
            output_path: Where to write the modified PDF. Overwritten if it
                already exists; the source is never modified in place.
            field_name: AcroForm field name for the widget. Used verbatim.
            x: Left edge in PDF points from the page's bottom-left origin
                (default 50).
            y: Bottom edge in PDF points from the page's bottom-left origin
                (default 100).
            width: Box width in points (default 150).
            height: Box height in points (default 25).
            page: 1-based page number (default 1). Out of range is an error.
            date_format: Free-text pattern shown in the caption, default
                "MM/DD/YYYY". Purely cosmetic; it is not parsed and not
                enforced, so any string is accepted here and in the field.
            label: Caption drawn 15 points below y in 10pt black. Empty
                string (the default) draws nothing.
            show_format_hint: Draw the grey "Format: <date_format>" caption to
                the right of the box (default True).

        Returns:
            Dict with success, field_name, date_format, dimensions, has_label,
            has_format_hint, page, output size and the output path.
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_pdf_path = validate_output_path(output_path)

            # Open PDF
            doc = pymupdf.open(str(input_pdf_path))
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
                label_point = pymupdf.Point(x, y - 15)
                pdf_page.insert_text(label_point, label, fontsize=10, color=(0, 0, 0))

            # Create date field rectangle
            field_rect = pymupdf.Rect(x, y, x + width, y + height)

            # Populate the Widget before adding it; see add_form_fields.
            widget = pymupdf.Widget()
            widget.field_name = field_name
            widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
            widget.rect = field_rect
            pdf_page.add_widget(widget)

            # Add format hint if requested
            if show_format_hint:
                hint_text = f"Format: {date_format}"
                hint_point = pymupdf.Point(x + width + 10, y + height/2)
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
        description=(
            "Check a JSON payload of field values against a JSON payload of "
            "rules. Read-only: nothing is written.\n"
            "\n"
            "This is PURE DATA validation. pdf_path is opened only far enough "
            "to validate the path; the document's form fields are NEVER read, "
            "so nothing here confirms that a field name exists in the PDF or "
            "that its type matches. Call extract_form_data first to learn the "
            "real names. For the coordinate-overlay (permit) field-definition "
            "format, use validate_permit_form_data instead.\n"
            "\n"
            "`form_data` is a JSON object of field name -> value:\n"
            '  {"applicant": "Jane Roe", "zip": "83702"}\n'
            "`validation_rules` is a JSON object keyed by the SAME field "
            "names, each holding any of exactly three rules:\n"
            '  {"applicant": {"required": true, "max_length": 60},\n'
            '   "zip": {"pattern": "^[0-9]{5}$"}}\n'
            "  - \"required\" (bool): fails on a falsy value, so 0, false and "
            '""  all count as empty.\n'
            "  - \"max_length\" (int): compares len(str(value)).\n"
            "  - \"pattern\" (regex string): applied with re.match, so it is "
            "anchored at the START only and needs a trailing $ to pin the "
            "end.\n"
            "Any other rule key is ignored.\n"
            "\n"
            "Only fields PRESENT in form_data are examined. A \"required\" "
            "rule for a field the caller omitted entirely never fires, so "
            "this tool CANNOT detect a missing field. Rules naming fields "
            "absent from form_data are ignored. The returned warnings list is "
            "always empty. The default validation_rules of \"{}\" checks "
            "nothing and reports every field valid."
        ),
        annotations={
            "readOnlyHint": True,        # reads only, writes nothing
            "idempotentHint": True,
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
    )
    async def validate_form_data(
        self,
        pdf_path: str,
        form_data: str,
        validation_rules: str = "{}"
    ) -> Dict[str, Any]:
        """
        Validate a form-data payload against caller-supplied rules.

        Args:
            pdf_path: Path to a PDF, or an HTTPS URL to fetch. Only used to
                validate the path and echo it back; its form fields are not
                inspected and play no part in the result.
            form_data: JSON object mapping field name to value, e.g.
                {"applicant": "Jane Roe", "zip": "83702"}. Only the keys
                present here are validated.
            validation_rules: JSON object keyed by field name. Each value may
                carry "required" (bool, fails on any falsy value),
                "max_length" (int, against len(str(value))) and "pattern"
                (regex applied with re.match, start-anchored only). Defaults
                to "{}", which validates nothing.

        Returns:
            Dict with success, validation_summary (is_valid, total_fields,
            valid_fields, invalid_fields, total_errors, total_warnings),
            per-field results, the errors list, an always-empty warnings list
            and the echoed file path.
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