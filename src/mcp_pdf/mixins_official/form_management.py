"""
Form Management Mixin - PDF form creation, filling, and field extraction
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import asyncio
import functools
import time
import json
from typing import Dict, Any, Literal, Optional, List
import logging

# PDF processing libraries
import pymupdf
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
        description=(
            "List every interactive AcroForm field in a PDF with its name, "
            "type, current value, page, coordinates and required/readonly "
            "flags. Read-only: nothing is written. Run this FIRST to learn "
            "the exact field names before calling fill_form_pdf, whose data "
            "keys must match them byte for byte.\n"
            "\n"
            "Each field carries both `field_type`, from the portable "
            "six-term vocabulary shared with extract_xfa_fields (text, "
            "checkbox, radio, dropdown, date, signature, plus button and "
            "unknown), and `field_type_raw`, the unmerged AcroForm type that "
            "still distinguishes listbox from combobox. `coordinates` is "
            "{x, y, width, height} where x/y are the field's lower-left "
            "corner in PDF points from the page's BOTTOM-left origin, the "
            "opposite convention to extract_xfa_fields' design-time boxes. "
            "`choices` and `max_length` appear only when the widget defines "
            "them.\n"
            "\n"
            "DYNAMIC XFA forms return success=false with is_xfa=true, "
            "xfa_type=\"dynamic\" and a hint pointing at extract_xfa_fields, "
            "because their fields live in the XFA template and not in "
            "AcroForm at all. A file neither reader can parse returns "
            "success=false with xfa_detection_failed=true, which means "
            "damaged rather than form-less."
        ),
        annotations={
            "readOnlyHint": True,        # reads only, writes nothing
            "idempotentHint": True,
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
    )
    async def extract_form_data(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract all form fields and their current values from PDF.

        Args:
            pdf_path: Path to the PDF, or an HTTPS URL to fetch.

        Returns:
            Dict with success, form_summary (total_fields, required_fields,
            readonly_fields, field_types histogram, has_form) and form_fields,
            each entry carrying page, field_name, field_type, field_type_raw,
            field_value, field_label, is_required, is_readonly, coordinates
            and, when present, choices and max_length.

            On a dynamic XFA form: success=false plus is_xfa, xfa_type and a
            hint naming extract_xfa_fields.
        """
        start_time = time.time()

        # Bound before the try: the except handler reads it, and an exception
        # from validate_pdf_path would otherwise make this an UnboundLocalError
        # that masks the real error.
        xfa_detection_failed = False

        try:
            path = await validate_pdf_path(pdf_path)

            # XFA early-detect. Dynamic XFA forms have no AcroForm widgets, so
            # without this we'd report total_fields=0 and the caller would
            # conclude the form is empty. Give the real diagnosis plus a
            # pointer to the tool that can actually read it.
            #
            # Detection has three outcomes, and the third one matters: if we
            # could not read the file at all, we must not treat that as "not
            # XFA". We still try PyMuPDF (MuPDF recovers from damage pypdf will
            # not), but we carry the detection failure forward so the error
            # path can say the file looks damaged rather than emitting a bare
            # "document closed".
            xfa_info = _detect_xfa(str(path))
            xfa_detection_failed = bool(xfa_info.get("detection_failed"))

            if xfa_info.get("is_xfa") and xfa_info.get("xfa_type") == "dynamic":
                return {
                    "success": False,
                    "is_xfa": True,
                    "xfa_type": "dynamic",
                    "error": (
                        "Dynamic XFA form: fields live in the XFA template, "
                        "not in AcroForm, so this tool cannot reach them."
                    ),
                    "hint": (
                        "Use extract_xfa_fields to get the XFA field schema "
                        "(names, types, captions, shared/positional categories)."
                    ),
                    "extraction_time": round(time.time() - start_time, 2),
                }

            doc = pymupdf.open(str(path))
            # Read the page count while the document is still open. The
            # success return below used to call len(doc) after doc.close(),
            # which raises "document closed" on PyMuPDF 1.28 — from inside
            # the return statement, so the handler turned every successful
            # extraction into success=false with that opaque message.
            total_pages = len(doc)

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
                            "field_type_raw": self._get_raw_field_type(widget),
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
                    "total_pages": total_pages
                },
                "extraction_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Form data extraction failed: {error_msg}")
            response = {
                "success": False,
                "error": error_msg,
                "extraction_time": round(time.time() - start_time, 2)
            }
            if xfa_detection_failed:
                # Both readers failed on this file. Saying so is far more
                # actionable than the bare "document closed" a damaged or
                # truncated form package used to produce.
                response["hint"] = (
                    "Neither pypdf nor MuPDF could read this file's structure. "
                    "It is likely truncated or corrupt rather than simply "
                    "form-less. Try repair_pdf, or re-download the original."
                )
                response["xfa_detection_failed"] = True
            return response

    @mcp_tool(
        name="fill_form_pdf",
        description=(
            "Fill the interactive AcroForm fields of an existing PDF, writing "
            "a NEW PDF to output_path. Requires the PDF to have real form "
            "widgets: for a scanned or flat form with no widgets use "
            "fill_permit_form, which draws text at coordinates instead, and "
            "for a dynamic XFA form neither tool can fill it (see "
            "extract_xfa_fields).\n"
            "\n"
            "`form_data` is a JSON OBJECT keyed by AcroForm field name:\n"
            '  {"applicant_name": "Jane Roe", "parcel": "R1234567", '
            '"agree": "Yes"}\n'
            "\n"
            "Names must match the document exactly; get them from "
            "extract_form_data. A key with no matching widget is SILENTLY "
            "IGNORED and appears in no error list, so compare fields_filled "
            "against total_data_provided rather than trusting success. Every "
            "value is coerced with str(), so a checkbox needs its literal "
            "on-state string such as \"Yes\" or \"Off\" (JSON true becomes "
            'the useless string "True"), and a number becomes its text form.\n'
            "\n"
            "A RADIO GROUP is several widgets sharing one field name, so "
            "give it ONE key whose value is the export value of the option "
            "to select; the matching button is turned on and the rest are "
            "forced off. Export values are sanitised from the option labels "
            "(\"Conventional Loan\" becomes \"Conventional_Loan\"), so take "
            "them from add_radio_group's option_values or from the "
            "on-state in extract_form_data rather than passing the label. A "
            "value matching no option selects nothing and leaves "
            "fields_filled short of total_data_provided. The group counts as "
            "ONE filled field, not one per button.\n"
            "\n"
            "flatten=True does NOT merely lock the fields: it rasterises "
            "every page to an image at 72 DPI and builds a new document from "
            "those pictures. The result has no selectable text, no "
            "searchability and no annotations at all. Leave it False unless "
            "you specifically want a picture of the filled form."
        ),
        annotations={
            "readOnlyHint": False,       # writes a new PDF
            "destructiveHint": True,     # overwrites output_path if it exists
            "idempotentHint": True,      # same args produce the same output file
            "openWorldHint": True,       # input_path may be an HTTPS URL
        },
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
            input_path: Path to the source PDF, or an HTTPS URL to fetch.
            output_path: Where to write the filled PDF. Overwritten if it
                already exists; the source is never modified in place.
            form_data: JSON object mapping AcroForm field name to value, e.g.
                {"applicant_name": "Jane Roe", "agree": "Yes"}. Names must
                match the document exactly (see extract_form_data); unmatched
                keys are ignored without error. Values are str()-coerced, so
                checkboxes need their on-state string, not JSON true.
            flatten: When True, rasterise each page to a 72 DPI image and
                rebuild the document from those images. Fields become
                non-editable because they no longer exist, and all text
                becomes unsearchable. Default False, which keeps a real PDF
                with live (still editable) fields.

        Returns:
            Dict with success, fill_summary (fields_filled, fields_failed,
            total_data_provided, form_flattened), failed_fields, the output
            path and its size.
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
            doc = pymupdf.open(str(input_pdf_path))
            fields_filled = 0
            fields_failed = 0
            # Radio groups whose requested option was found, so an unmatched
            # value can be reported rather than looking like a silent no-op.
            radio_groups_matched = set()
            failed_fields = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                try:
                    widgets = page.widgets()

                    for widget in widgets:
                        field_name = widget.field_name
                        if field_name and field_name in data:
                            try:
                                requested = str(data[field_name])

                                # A radio group is several widgets sharing one
                                # field name, each with its own on-state.
                                # Assigning the requested value to every one of
                                # them turns them ALL on, which is what this
                                # did until 2026-09-22: filling a 5-option
                                # group reported fields_filled=5 for one key
                                # and produced a form with every option
                                # selected. Select the matching button and push
                                # the rest to "Off".
                                if widget.field_type == pymupdf.PDF_WIDGET_TYPE_RADIOBUTTON:
                                    states = widget.button_states() or {}
                                    normal = states.get("normal") or []
                                    on_state = next(
                                        (s for s in normal if s != "Off"), None
                                    )
                                    # Write /AS and /V directly instead of
                                    # going through widget.update(). For a
                                    # radio button PyMuPDF's updater ignores an
                                    # "Off" assignment and switches the widget
                                    # ON with its own state, so the obvious
                                    # implementation leaves EVERY option in the
                                    # group selected. /AS chooses which
                                    # appearance is drawn; /V is the field
                                    # value and is the same across the group.
                                    doc.xref_set_key(
                                        widget.xref, "AS",
                                        f"/{on_state}" if on_state == requested else "/Off",
                                    )
                                    doc.xref_set_key(widget.xref, "V", f"/{requested}")
                                    # Count the GROUP once, on the button that
                                    # actually took the value, so fields_filled
                                    # stays comparable to total_data_provided.
                                    if on_state == requested:
                                        fields_filled += 1
                                        radio_groups_matched.add(field_name)
                                    continue

                                # Set field value
                                widget.field_value = requested
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
                flattened_doc = pymupdf.open()
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap()
                    new_page = flattened_doc.new_page(width=page.rect.width, height=page.rect.height)
                    new_page.insert_image(new_page.rect, pixmap=pix)

                flattened_doc.save(str(output_pdf_path))
                flattened_doc.close()
            else:
                doc.save(str(output_pdf_path), incremental=False, encryption=pymupdf.PDF_ENCRYPT_NONE)

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
        description=(
            "Build a BRAND-NEW single-page PDF form from scratch, writing it "
            "to output_path. There is no input PDF; to add widgets to a "
            "document you already have, use add_form_fields instead. "
            "Requires reportlab (pip install mcp-pdf[forms]), and returns a "
            "clear error naming it if absent.\n"
            "\n"
            "`fields` is a JSON array of objects:\n"
            '  [{"name": "applicant", "type": "text", "label": "Applicant '
            'name", "x": 50, "y": 700, "width": 220, "height": 20,\n'
            '    "tooltip": "Legal name as on the deed"},\n'
            '   {"name": "county", "type": "dropdown", "options": ["Ada", '
            '"Canyon"]},\n'
            '   {"name": "agree", "type": "checkbox", "checked": false}]\n'
            "\n"
            "type is text | checkbox | dropdown | signature (default text). "
            "Anything else draws only the label and still counts toward "
            "fields_created. name defaults to field_<n>, label defaults to "
            "the name and is drawn 5 points above the box. x defaults to 50 "
            "and y to 700 minus 40 per field already created, both in PDF "
            "points from the page's BOTTOM-left origin; width 200, height 20. "
            "tooltip defaults to \"\". `options` applies to dropdown only, a "
            'list of strings defaulting to ["Option 1", "Option 2"] (an '
            "empty list falls back to that too), and the widget opens with "
            "its FIRST option already selected rather than blank. `checked` "
            "applies to checkbox only (default false), and a checkbox "
            "ignores width, using height as its side length. A signature "
            "field is really a text field with a "
            "thicker border and the word SIGNATURE printed inside; it is not "
            "a cryptographic signature field.\n"
            "\n"
            "EVERYTHING LANDS ON ONE PAGE. No page break is ever emitted, so "
            "with the stacked default y more than about 17 fields run off the "
            "bottom of the sheet and are invisible though still reported as "
            "created. Pass explicit x/y for anything larger."
        ),
        annotations={
            "readOnlyHint": False,       # writes a new PDF
            "destructiveHint": True,     # overwrites output_path if it exists
            "idempotentHint": True,      # same args produce the same output file
            "openWorldHint": False,      # no input path; nothing is fetched
        },
    )
    async def create_form_pdf(
        self,
        output_path: str,
        fields: str,
        title: str = "Form Document",
        page_size: Literal["A4", "Letter", "Legal"] = "A4"
    ) -> Dict[str, Any]:
        """
        Create a new PDF form with interactive fields.

        Args:
            output_path: Where to write the new PDF. Overwritten if it
                already exists.
            fields: JSON array of field objects. Each may carry:
                - "name":   AcroForm field name (default "field_<n>")
                - "type":   "text" | "checkbox" | "dropdown" | "signature"
                            (default "text"); other values draw only a label
                - "label":  text drawn 5 points above the box (default: name)
                - "x", "y": lower-left corner in PDF points from the page's
                            bottom-left origin (defaults 50 and
                            700 - 40*fields_already_created)
                - "width", "height": size in points (defaults 200, 20);
                            checkbox ignores width and uses height as its side
                - "tooltip": hover text (default "")
                - "options": dropdown choices, list of strings (default
                            ["Option 1", "Option 2"]); the first one starts
                            selected
                - "checked": checkbox initial state (default False)
            title: Value written into the PDF's document Title metadata
                (default "Form Document"). Not drawn on the page.
            page_size: Sheet size, one of "A4", "Letter" or "Legal"
                (default "A4"). Case-sensitive.

        Returns:
            Dict with success, form_info (fields_created,
            total_fields_requested, page_size, title), the output path and its
            size. A missing reportlab yields success=false with an install
            hint.
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
                            # `or` rather than a get() default so an explicit
                            # empty list also falls back: reportlab cannot
                            # build a choice widget with no options.
                            options = field_def.get("options") or ["Option 1", "Option 2"]
                            # value must be non-empty and one of the options.
                            # reportlab 4.4.3's _textfield only binds its
                            # internal `lbextras` inside `if value:`, so
                            # choice(value='') dies with UnboundLocalError —
                            # which this handler swallowed, silently dropping
                            # every dropdown from the generated form.
                            c.acroForm.choice(
                                name=field_name,
                                value=options[0],
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

        Aligned with extract_xfa_fields so callers see the same terms
        regardless of which form system the PDF uses: text / checkbox / radio
        / dropdown / date / signature, plus button / unknown for the edges.

        listbox and combobox both map to "dropdown" because that is the term
        the XFA side produces for its single `choiceList` widget, so one
        vocabulary needs one word for "pick from a list". The two are NOT
        interchangeable though: a combobox may allow free-text entry and a
        listbox may allow multi-select, and both of those change how a caller
        has to fill the field. Use `_get_raw_field_type` when that matters;
        `extract_form_data` reports it as `field_type_raw` on every field.

        Note that `date` is currently unreachable from this side, because
        PyMuPDF exposes no date widget type. AcroForm date fields arrive as
        text with a format action attached.
        """
        field_type = getattr(widget, 'field_type', 0)

        # Field type constants from PyMuPDF
        if field_type == pymupdf.PDF_WIDGET_TYPE_BUTTON:
            return "button"
        elif field_type == pymupdf.PDF_WIDGET_TYPE_CHECKBOX:
            return "checkbox"
        elif field_type == pymupdf.PDF_WIDGET_TYPE_RADIOBUTTON:
            return "radio"
        elif field_type == pymupdf.PDF_WIDGET_TYPE_TEXT:
            return "text"
        elif field_type == pymupdf.PDF_WIDGET_TYPE_LISTBOX:
            return "dropdown"
        elif field_type == pymupdf.PDF_WIDGET_TYPE_COMBOBOX:
            return "dropdown"
        elif field_type == pymupdf.PDF_WIDGET_TYPE_SIGNATURE:
            return "signature"
        else:
            return "unknown"

    def _get_raw_field_type(self, widget) -> str:
        """The unmerged AcroForm widget type.

        Preserves the distinctions the portable vocabulary deliberately
        collapses, so nothing is destroyed by the alignment. Currently that
        means listbox vs combobox, which differ on free-text entry and
        multi-select.
        """
        field_type = getattr(widget, 'field_type', 0)

        if field_type == pymupdf.PDF_WIDGET_TYPE_BUTTON:
            return "button"
        elif field_type == pymupdf.PDF_WIDGET_TYPE_CHECKBOX:
            return "checkbox"
        elif field_type == pymupdf.PDF_WIDGET_TYPE_RADIOBUTTON:
            return "radio"
        elif field_type == pymupdf.PDF_WIDGET_TYPE_TEXT:
            return "text"
        elif field_type == pymupdf.PDF_WIDGET_TYPE_LISTBOX:
            return "listbox"
        elif field_type == pymupdf.PDF_WIDGET_TYPE_COMBOBOX:
            return "combobox"
        elif field_type == pymupdf.PDF_WIDGET_TYPE_SIGNATURE:
            return "signature"
        else:
            return "unknown"

    @mcp_tool(
        name="is_xfa_pdf",
        description=(
            "Detect whether a PDF is XFA (Adobe LiveCycle / dynamic forms) "
            "and whether it is dynamic or static. Dynamic XFA forms cannot be "
            "rendered by any open-source PDF library; only Adobe's runtime "
            "executes them. Use this to branch BEFORE calling extract_form_data "
            "or convert_to_images on a PDF that might be dynamic XFA. Returns "
            "{success, is_xfa, xfa_type: 'dynamic'|'static'|None, has_acroform, "
            "detection_failed}. IMPORTANT: check detection_failed first. When "
            "it is true, is_xfa is null (not false) because the file could not "
            "be read at all, which is a different fact from 'no XFA here'."
        ),
        annotations={
            "readOnlyHint": True,        # inspects the file, writes nothing
            "idempotentHint": True,
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
    )
    async def is_xfa_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Detect XFA presence and type (dynamic vs static).

        Args:
            pdf_path: Path to PDF file or HTTPS URL.

        Returns:
            Dict with success, is_xfa (True / False / None), xfa_type
            ("dynamic" / "static" / None), has_acroform, detection_failed and
            reason.

            ``is_xfa`` is None exactly when ``detection_failed`` is True. A
            truncated or corrupt PDF lands there rather than being reported as
            a confident "not XFA", which would send the caller down the
            AcroForm path to a cryptic failure.
        """
        start_time = time.time()
        try:
            path = await validate_pdf_path(pdf_path)
            # pypdf parses the whole file in-memory, so keep it off the loop.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, _detect_xfa, str(path))
            result["success"] = not result.get("detection_failed", False)
            result["detection_time"] = round(time.time() - start_time, 2)
            return result
        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"XFA detection failed: {error_msg}")
            return {
                "success": False,
                "is_xfa": None,
                "xfa_type": None,
                "has_acroform": None,
                "detection_failed": True,
                "reason": error_msg,
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
            "+ other + plumbing-dropped field breakdown. NOTE four categories, "
            "not three: 'other' is the default bucket for names matching "
            "neither the shared prefix nor a positional pattern, and on a "
            "non-zipForm producer it usually holds most of the fields. The "
            "`original` XFA name is on every field and is the round-trip key "
            "for actually filling the form."
        ),
        annotations={
            "readOnlyHint": True,        # parses the XFA template, writes nothing
            "idempotentHint": True,
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
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
        max_inline_fields: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Extract the XFA field schema from a dynamic-XFA PDF.

        Args:
            pdf_path: Path to PDF file or HTTPS URL.
            profile: Producer profile, matched case-insensitively. Either
                "zipform" (Lone Wolf / zipForm Plus conventions) or "generic"
                (only the Global_Info- shared-prefix convention; callers add
                producer-specific patterns themselves). An unrecognized value
                is an error, not a silent fallback to generic.
            extra_plumbing_exact: Additional exact field names to drop as
                plumbing. Compared case-insensitively.
            extra_plumbing_patterns: Regex patterns (strings) for additional
                plumbing fields. Matched case-insensitively as a SUBSTRING
                search, so they need no anchoring.
            extra_positional_patterns: Regex patterns (strings) identifying
                additional opaque positional codes. Matched case-insensitively
                and ANCHORED AT THE START, so `tf\\d+$` will not match
                `p01tf001` but `^p\\d+tf\\d+$` will.
            canonical_separator: Separator for canonical names. "_" (snake,
                default), "." (dotted), or "-" (kebab).
            include_design_time_bbox: Include best-effort design-time geometry
                on every field. NOT authoritative for dynamic XFA, whose
                subforms reflow at render time. Geometry is page-relative with
                a top-left origin, the opposite convention to
                extract_form_data's bottom-up coordinates.
            max_inline_fields: Cap on fields serialized into the response
                (default 5000, or MCP_PDF_MAX_XFA_INLINE_FIELDS). Counts in
                `categories` and `shared_fields` always cover every field.

        Returns:
            Dict always carrying success, is_xfa and xfa_type, so a caller can
            read those keys on any outcome including failure. On success also
            xfa_parts (in file order), field_count, fields, categories,
            shared_fields, canonical_collisions, plumbing_fields_dropped,
            profile_used and warnings. See extract_xfa_schema's docstring for
            the full shape.
        """
        start_time = time.time()
        try:
            path = await validate_pdf_path(pdf_path)
            # pypdf + ElementTree are both blocking and can run for seconds on
            # a large template. Offload so one big form cannot stall the
            # server for every other request, matching create_form_pdf.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                functools.partial(
                    extract_xfa_schema,
                    str(path),
                    profile=profile,
                    extra_plumbing_exact=extra_plumbing_exact,
                    extra_plumbing_patterns=extra_plumbing_patterns,
                    extra_positional_patterns=extra_positional_patterns,
                    canonical_separator=canonical_separator,
                    include_design_time_bbox=include_design_time_bbox,
                    max_inline_fields=max_inline_fields,
                ),
            )
            result["extraction_time"] = round(time.time() - start_time, 2)
            return result
        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"XFA field extraction failed: {error_msg}")
            # Carry the contract keys even here. Without them a caller doing
            # result["is_xfa"] gets a KeyError precisely when the call failed.
            return {
                "success": False,
                "is_xfa": None,
                "xfa_type": None,
                "error": error_msg,
                "extraction_time": round(time.time() - start_time, 2),
            }