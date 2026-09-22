"""
Form Management Mixin - PDF form creation, filling, and field extraction
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import asyncio
import functools
import time
import json
from typing import Dict, Any, Optional, List
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
                    "total_pages": len(doc) if 'doc' in locals() else 0
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
            doc = pymupdf.open(str(input_pdf_path))
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
        )
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