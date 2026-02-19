"""
Permit Form Mixin - Coordinate-based PDF form filling using overlay technique
Uses official fastmcp.contrib.mcp_mixin pattern

This mixin enables filling ANY PDF (scanned, flat, non-interactive) by drawing
text and checkboxes at specified (x, y) coordinates, then merging the overlay
with the original template. This is ideal for government forms that don't have
proper AcroForm fields.

Requires: pip install mcp-pdf[forms]
"""
from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, BinaryIO, TYPE_CHECKING
import logging

# PDF processing libraries (always available)
from pypdf import PdfReader, PdfWriter
from PIL import Image

# Official FastMCP mixin
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, validate_output_path, sanitize_error_message

logger = logging.getLogger(__name__)

# Lazy import for reportlab (optional dependency)
_reportlab_available = None

def _check_reportlab():
    """Check if reportlab is available, raise helpful error if not."""
    global _reportlab_available
    if _reportlab_available is None:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.units import inch
            from reportlab.lib.utils import ImageReader
            from reportlab.pdfgen import canvas
            _reportlab_available = True
        except ImportError:
            _reportlab_available = False

    if not _reportlab_available:
        raise ImportError(
            "reportlab is required for permit form tools. "
            "Install with: pip install mcp-pdf[forms]"
        )

def _get_reportlab():
    """Get reportlab modules, raising error if not available."""
    _check_reportlab()
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    return {
        'letter': letter,
        'inch': inch,
        'ImageReader': ImageReader,
        'canvas': canvas,
    }

# Page dimensions: 612 x 792 points (letter size)
# Y coordinates in PDF are from bottom, so we convert from top-origin
PAGE_HEIGHT = 792
PAGE_WIDTH = 612

# Margins for attachment pages (in points, 72 points = 1 inch)
MARGIN_TOP = 54      # 0.75 inch
MARGIN_BOTTOM = 36   # 0.5 inch
MARGIN_LEFT = 36     # 0.5 inch
MARGIN_RIGHT = 36    # 0.5 inch

# Header styling for attachment pages
HEADER_HEIGHT = 36   # 0.5 inch
HEADER_FONT_SIZE = 14


class ImageFitMode(str, Enum):
    """How images should fit within page bounds for attachments."""
    CONTAIN = "contain"  # Fit entirely within bounds, maintain aspect ratio
    COVER = "cover"      # Fill bounds entirely, crop if needed
    STRETCH = "stretch"  # Stretch to fill bounds exactly (may distort)


@dataclass
class AttachmentConfig:
    """Configuration for a page attachment/insert."""
    name: str                           # Field name for tracking
    page_title: str                     # Title shown in header
    insert_after_page: Optional[int]    # None = end of document
    content_type: str                   # "image" or "text"
    image_fit: ImageFitMode             # How image fits on page
    show_header: bool                   # Whether to show title header
    # Field position for "See page X" annotation
    field_page: Optional[int] = None    # Page where marker appears (1-indexed)
    field_x: Optional[float] = None     # X position for annotation
    field_y: Optional[float] = None     # Y position for annotation
    field_width: Optional[float] = None # Width of marker area
    field_height: Optional[float] = None # Height of marker area


class FieldCoordinates:
    """Helper class to look up field coordinates and tuning properties from field definitions.

    Handles the coordinate system conversion:
    - Input: y from top (0 = top of page) - how humans think
    - Output: y from bottom (0 = bottom of page) - how PDF works
    """

    def __init__(self, fields_by_page: Dict[str, List[Dict[str, Any]]]):
        """Initialize with field definitions organized by page.

        Args:
            fields_by_page: Dict like {"1": [field_dicts...], "2": [...]}
        """
        self._fields: Dict[str, Dict[str, Any]] = {}
        self._fields_by_page: Dict[int, List[str]] = {}

        for page_num, fields in fields_by_page.items():
            page_int = int(page_num)
            self._fields_by_page[page_int] = []

            for field in fields:
                name = field.get("name", "")
                if name:
                    self._fields[name] = {
                        "x": field.get("x", 0),
                        "y": field.get("y", 0),
                        "width": field.get("width", 150),
                        "height": field.get("height", 12),
                        "type": field.get("type", "text"),
                        "page": page_int,
                        # Text tuning properties
                        "font_size": field.get("font_size"),
                        "font_name": field.get("font_name"),
                        "text_align": field.get("text_align"),
                        "max_chars": field.get("max_chars"),
                        "multiline": field.get("multiline"),
                        "line_spacing": field.get("line_spacing"),
                        "y_offset": field.get("y_offset"),
                        "x_offset": field.get("x_offset"),
                    }
                    self._fields_by_page[page_int].append(name)

    def get(self, field_name: str) -> Optional[tuple]:
        """Get (x, y) coordinates for a field, converting y from top to bottom.

        Returns None if field not found.
        Applies x_offset and y_offset if configured.
        """
        field = self._fields.get(field_name)
        if not field:
            return None
        # Convert y from top-origin to bottom-origin for ReportLab
        x = field["x"] + (field.get("x_offset") or 0)
        y = PAGE_HEIGHT - field["y"] - field["height"] + (field.get("y_offset") or 0)
        return (x, y)

    def get_raw(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Get raw field data including width, height, type, and tuning properties."""
        return self._fields.get(field_name)

    def has(self, field_name: str) -> bool:
        """Check if a field exists."""
        return field_name in self._fields

    def get_page(self, field_name: str) -> Optional[int]:
        """Get the page number for a field (1-indexed)."""
        field = self._fields.get(field_name)
        return field["page"] if field else None

    def get_fields_for_page(self, page_num: int) -> List[str]:
        """Get all field names for a specific page."""
        return self._fields_by_page.get(page_num, [])

    def get_all_fields(self) -> Dict[str, Dict[str, Any]]:
        """Get all fields."""
        return self._fields.copy()

    def get_pages(self) -> List[int]:
        """Get list of page numbers that have fields."""
        return sorted(self._fields_by_page.keys())


def _wrap_text(text: str, max_chars: int) -> List[str]:
    """Wrap text into lines of max_chars length, breaking on word boundaries."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if len(test) <= max_chars:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_aligned_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    align: str,
) -> None:
    """Draw text with specified alignment."""
    if align == "center":
        text_width = c.stringWidth(text)
        x = x + (width - text_width) / 2
    elif align == "right":
        text_width = c.stringWidth(text)
        x = x + width - text_width
    c.drawString(x, y, text)


def _draw_text_at_field(
    c: canvas.Canvas,
    coords: FieldCoordinates,
    field_name: str,
    value: Optional[str],
    default_font_size: int = 9,
) -> bool:
    """Draw text at the field's coordinates.

    Returns True if text was drawn, False if field not found or no value.
    """
    if not value:
        return False
    pos = coords.get(field_name)
    if not pos:
        return False

    # Get field tuning properties
    field = coords.get_raw(field_name)
    if field:
        font_size = field.get("font_size") or default_font_size
        font_name = field.get("font_name") or "Helvetica"
        text_align = field.get("text_align") or "left"
        max_chars = field.get("max_chars")
        multiline = field.get("multiline")
        line_spacing = field.get("line_spacing") or 12
        width = field.get("width", 150)
    else:
        font_size = default_font_size
        font_name = "Helvetica"
        text_align = "left"
        max_chars = None
        multiline = False
        line_spacing = 12
        width = 150

    c.setFont(font_name, font_size)
    text = str(value)

    # Handle text wrapping for multiline fields
    if multiline and max_chars:
        lines = _wrap_text(text, max_chars)
        for i, line in enumerate(lines):
            y_pos = pos[1] - (i * line_spacing)
            _draw_aligned_text(c, line, pos[0], y_pos, width, text_align)
    else:
        # Single line - truncate if max_chars specified
        if max_chars and len(text) > max_chars:
            text = text[:max_chars - 3] + "..."
        _draw_aligned_text(c, text, pos[0], pos[1], width, text_align)

    return True


def _draw_checkbox_at_field(
    c: canvas.Canvas,
    coords: FieldCoordinates,
    field_name: str,
    checked: bool,
) -> bool:
    """Draw checkbox mark at the field's coordinates.

    Returns True if checkbox was drawn, False if field not found.
    """
    pos = coords.get(field_name)
    if not pos:
        return False

    if not checked:
        return True  # Field exists but not checked - still counts as processed

    # Get field tuning properties
    field = coords.get_raw(field_name)
    font_name = "Helvetica-Bold"
    font_size = 10
    if field:
        font_name = field.get("font_name") or font_name
        font_size = field.get("font_size") or font_size

    c.setFont(font_name, font_size)
    c.drawString(pos[0], pos[1], "X")
    return True


def _create_attachment_page_with_image(
    title: str,
    image_data: bytes,
    fit_mode: ImageFitMode = ImageFitMode.CONTAIN,
    show_header: bool = True,
    filename: Optional[str] = None,
) -> io.BytesIO:
    """Create a new PDF page with an image attachment.

    Args:
        title: Title to display in the header
        image_data: Raw image bytes (PNG, JPEG, etc.)
        fit_mode: How the image should fit on the page
        show_header: Whether to show the title header
        filename: Original filename to display below title

    Returns:
        BytesIO buffer containing the single-page PDF
    """
    rl = _get_reportlab()
    buffer = io.BytesIO()
    c = rl['canvas'].Canvas(buffer, pagesize=rl['letter'])

    # Calculate content area
    content_top = PAGE_HEIGHT - MARGIN_TOP
    content_bottom = MARGIN_BOTTOM
    content_left = MARGIN_LEFT
    content_right = PAGE_WIDTH - MARGIN_RIGHT
    content_width = content_right - content_left
    content_height = content_top - content_bottom

    # Reserve space for header if needed
    if show_header:
        c.setFont("Helvetica-Bold", HEADER_FONT_SIZE)
        c.drawCentredString(PAGE_WIDTH / 2, content_top - 20, title)

        if filename:
            c.setFont("Helvetica", 9)
            c.drawCentredString(PAGE_WIDTH / 2, content_top - 35, f"({filename})")
            header_space = HEADER_HEIGHT + 15
        else:
            header_space = HEADER_HEIGHT

        # Draw separator line
        c.setStrokeColorRGB(0.7, 0.7, 0.7)
        c.setLineWidth(0.5)
        line_y = content_top - header_space + 5
        c.line(content_left, line_y, content_right, line_y)

        # Adjust content area
        content_top = line_y - 10
        content_height = content_top - content_bottom

    # Load and process image
    img_buffer = io.BytesIO(image_data)
    try:
        pil_image = Image.open(img_buffer)
        img_width, img_height = pil_image.size

        # Calculate placement based on fit mode
        if fit_mode == ImageFitMode.STRETCH:
            draw_x = content_left
            draw_y = content_bottom
            draw_width = content_width
            draw_height = content_height

        elif fit_mode == ImageFitMode.COVER:
            scale_x = content_width / img_width
            scale_y = content_height / img_height
            scale = max(scale_x, scale_y)
            draw_width = img_width * scale
            draw_height = img_height * scale
            draw_x = content_left + (content_width - draw_width) / 2
            draw_y = content_bottom + (content_height - draw_height) / 2

        else:  # CONTAIN (default)
            scale_x = content_width / img_width
            scale_y = content_height / img_height
            scale = min(scale_x, scale_y)
            draw_width = img_width * scale
            draw_height = img_height * scale
            draw_x = content_left + (content_width - draw_width) / 2
            draw_y = content_bottom + (content_height - draw_height) / 2

        # Draw the image
        img_buffer.seek(0)
        img_reader = rl['ImageReader'](img_buffer)
        c.drawImage(
            img_reader,
            draw_x, draw_y,
            width=draw_width, height=draw_height,
            preserveAspectRatio=False,
        )

    except Exception as e:
        # Draw error message if image loading fails
        c.setFont("Helvetica", 12)
        c.setFillColorRGB(0.8, 0, 0)
        c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT / 2, f"Error loading image: {str(e)}")

    c.save()
    buffer.seek(0)
    return buffer


def _create_attachment_page_with_text(
    title: str,
    text_content: str,
    show_header: bool = True,
) -> io.BytesIO:
    """Create a new PDF page with text content.

    Args:
        title: Title to display in the header
        text_content: The text to display on the page
        show_header: Whether to show the title header

    Returns:
        BytesIO buffer containing the single-page PDF
    """
    rl = _get_reportlab()
    buffer = io.BytesIO()
    c = rl['canvas'].Canvas(buffer, pagesize=rl['letter'])

    content_top = PAGE_HEIGHT - MARGIN_TOP
    content_bottom = MARGIN_BOTTOM
    content_left = MARGIN_LEFT
    content_right = PAGE_WIDTH - MARGIN_RIGHT

    if show_header:
        c.setFont("Helvetica-Bold", HEADER_FONT_SIZE)
        c.drawCentredString(PAGE_WIDTH / 2, content_top - 20, title)

        c.setStrokeColorRGB(0.7, 0.7, 0.7)
        c.setLineWidth(0.5)
        line_y = content_top - HEADER_HEIGHT + 5
        c.line(content_left, line_y, content_right, line_y)
        content_top = line_y - 15

    # Draw text content with simple line wrapping
    c.setFont("Helvetica", 11)
    c.setFillColorRGB(0, 0, 0)
    max_chars_per_line = 85
    line_height = 14
    y = content_top

    for paragraph in text_content.split('\n'):
        if not paragraph.strip():
            y -= line_height
            continue

        words = paragraph.split()
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    c.drawString(content_left, y, current_line)
                    y -= line_height
                current_line = word

                if y < content_bottom:
                    c.showPage()
                    c.setFont("Helvetica", 11)
                    y = PAGE_HEIGHT - MARGIN_TOP

        if current_line:
            c.drawString(content_left, y, current_line)
            y -= line_height

    c.save()
    buffer.seek(0)
    return buffer


def _create_see_page_annotation(
    x: float,
    y: float,
    width: float,
    height: float,
    target_page: int,
    title: str,
) -> io.BytesIO:
    """Create a PDF overlay with 'See page X' annotation.

    Args:
        x: X position of the annotation (from left)
        y: Y position from top (will be converted to PDF coordinates)
        width: Width of the annotation box
        height: Height of the annotation box
        target_page: The page number to reference
        title: The title of the insert

    Returns:
        BytesIO buffer containing a single-page PDF with the annotation
    """
    rl = _get_reportlab()
    buffer = io.BytesIO()
    c = rl['canvas'].Canvas(buffer, pagesize=rl['letter'])

    # Convert y from top-down to PDF bottom-up coordinates
    pdf_y = PAGE_HEIGHT - y - height

    # Draw a subtle box with the reference text
    c.setStrokeColorRGB(0.4, 0.4, 0.4)
    c.setFillColorRGB(0.97, 0.97, 0.97)
    c.setLineWidth(0.5)
    c.roundRect(x, pdf_y, width, height, 3, stroke=1, fill=1)

    # Draw the text - center it in the box
    c.setFillColorRGB(0.3, 0.3, 0.3)
    c.setFont("Helvetica", 8)
    text = f"See page {target_page}"
    if title:
        text = f"{title} - See page {target_page}"

    # Truncate text if too long for the box
    text_width = c.stringWidth(text, "Helvetica", 8)
    if text_width > width - 6:
        text = f"See page {target_page}"
        text_width = c.stringWidth(text, "Helvetica", 8)

    text_x = x + (width - text_width) / 2
    text_y = pdf_y + (height - 8) / 2 + 1

    c.drawString(text_x, text_y, text)

    c.save()
    buffer.seek(0)
    return buffer


def _create_page_overlay(
    coords: FieldCoordinates,
    form_data: Dict[str, Any],
    page_num: int,
) -> io.BytesIO:
    """Create overlay for a specific page with form data."""
    rl = _get_reportlab()
    buffer = io.BytesIO()
    c = rl['canvas'].Canvas(buffer, pagesize=rl['letter'])
    c.setFont("Helvetica", 9)

    # Get fields for this page
    field_names = coords.get_fields_for_page(page_num)

    for field_name in field_names:
        if field_name not in form_data:
            continue

        value = form_data[field_name]
        field_info = coords.get_raw(field_name)
        field_type = field_info.get("type", "text") if field_info else "text"

        if field_type == "checkbox":
            # Checkbox: value should be boolean or truthy
            checked = bool(value) if value is not None else False
            _draw_checkbox_at_field(c, coords, field_name, checked)
        else:
            # Text field
            _draw_text_at_field(c, coords, field_name, str(value) if value else None)

    c.save()
    buffer.seek(0)
    return buffer


class PermitFormMixin(MCPMixin):
    """
    Handles coordinate-based PDF form filling using the overlay technique.

    This approach works with ANY PDF (scanned, flat, non-interactive) by:
    1. Creating a transparent overlay with text/checkboxes at specified coordinates
    2. Merging the overlay with the original template PDF

    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()

    def _load_field_definitions(
        self,
        field_definitions: Optional[str],
        field_definitions_path: Optional[str],
    ) -> Dict[str, Any]:
        """Load field definitions from inline JSON or file path.

        Raises ValueError if neither or both are provided.
        """
        if field_definitions and field_definitions_path:
            raise ValueError("Provide either field_definitions OR field_definitions_path, not both")

        if not field_definitions and not field_definitions_path:
            raise ValueError("Must provide either field_definitions or field_definitions_path")

        if field_definitions:
            return json.loads(field_definitions)
        else:
            path = Path(field_definitions_path)
            if not path.exists():
                raise FileNotFoundError(f"Field definitions file not found: {field_definitions_path}")
            with open(path, "r") as f:
                return json.load(f)

    @mcp_tool(
        name="fill_permit_form",
        description="""Fill a PDF form using coordinate-based overlay technique.

This works with ANY PDF (scanned, flat, non-interactive) by drawing text and
checkboxes at specified (x, y) coordinates, then merging with the template.

Args:
    template_path: Path to the PDF template file
    form_data: JSON object with field names as keys and values to fill
    field_definitions: Inline JSON with field coordinates (mutually exclusive with field_definitions_path)
    field_definitions_path: Path to JSON file with field coordinates
    output_path: Optional path to save filled PDF (if not provided, returns base64)

Returns:
    Dictionary with success status, filled PDF (base64 or path), and statistics
"""
    )
    async def fill_permit_form(
        self,
        template_path: str,
        form_data: str,
        field_definitions: Optional[str] = None,
        field_definitions_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fill a PDF form using coordinate-based overlay technique."""
        start_time = time.time()

        try:
            # Validate template path
            template_file = await validate_pdf_path(template_path)

            # Load field definitions
            field_defs = self._load_field_definitions(field_definitions, field_definitions_path)

            # Parse form data
            data = json.loads(form_data)
            if not isinstance(data, dict):
                raise ValueError("form_data must be a JSON object")

            # Extract pages from field definitions
            pages_def = field_defs.get("pages", {})
            if not pages_def:
                raise ValueError("field_definitions must contain a 'pages' object")

            # Create coordinate lookup
            coords = FieldCoordinates(pages_def)

            # Read template
            template_pdf = PdfReader(str(template_file))
            writer = PdfWriter()

            fields_filled = 0
            pages_filled = 0

            # Process each page
            for page_idx, template_page in enumerate(template_pdf.pages):
                page_num = page_idx + 1  # 1-indexed

                # Check if this page has any fields
                page_fields = coords.get_fields_for_page(page_num)

                if page_fields:
                    # Create overlay for this page
                    overlay_buffer = _create_page_overlay(coords, data, page_num)
                    overlay_pdf = PdfReader(overlay_buffer)

                    # Merge overlay onto template page
                    template_page.merge_page(overlay_pdf.pages[0])
                    pages_filled += 1

                    # Count filled fields
                    for field_name in page_fields:
                        if field_name in data and data[field_name] is not None:
                            fields_filled += 1

                writer.add_page(template_page)

            # Write output
            output_buffer = io.BytesIO()
            writer.write(output_buffer)
            output_buffer.seek(0)
            pdf_bytes = output_buffer.getvalue()

            result = {
                "success": True,
                "pages_total": len(template_pdf.pages),
                "pages_filled": pages_filled,
                "fields_filled": fields_filled,
                "fields_total": len(coords.get_all_fields()),
                "processing_time": round(time.time() - start_time, 2),
            }

            # Output to file or base64
            if output_path:
                out_path = validate_output_path(output_path)
                with open(out_path, "wb") as f:
                    f.write(pdf_bytes)
                result["output_path"] = str(out_path)
                result["output_size_bytes"] = len(pdf_bytes)
            else:
                result["pdf_base64"] = base64.b64encode(pdf_bytes).decode("utf-8")
                result["output_size_bytes"] = len(pdf_bytes)

            return result

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {sanitize_error_message(str(e))}"
            logger.error(f"Form filling failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2),
            }
        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Form filling failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2),
            }

    @mcp_tool(
        name="get_field_schema",
        description="""Get field schema from field definitions for validation or UI generation.

Returns a list of all fields with their names, types, pages, and constraints.
Useful for building dynamic forms or validating data before filling.

Args:
    field_definitions: Inline JSON with field definitions
    field_definitions_path: Path to JSON file with field definitions

Returns:
    Dictionary with field schema including names, types, pages, and constraints
"""
    )
    async def get_field_schema(
        self,
        field_definitions: Optional[str] = None,
        field_definitions_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get field schema from field definitions."""
        start_time = time.time()

        try:
            # Load field definitions
            field_defs = self._load_field_definitions(field_definitions, field_definitions_path)

            # Extract metadata
            version = field_defs.get("version", "unknown")
            template = field_defs.get("template", "unknown")
            pages_def = field_defs.get("pages", {})

            # Build schema
            fields = []
            for page_num, page_fields in pages_def.items():
                for field in page_fields:
                    field_info = {
                        "name": field.get("name"),
                        "type": field.get("type", "text"),
                        "page": int(page_num),
                        "required": field.get("required", False),
                    }

                    # Add optional constraints
                    if field.get("max_chars"):
                        field_info["max_chars"] = field["max_chars"]
                    if field.get("multiline"):
                        field_info["multiline"] = True

                    fields.append(field_info)

            # Sort by page then by name
            fields.sort(key=lambda f: (f["page"], f["name"]))

            # Group by type for summary
            type_counts = {}
            for f in fields:
                t = f["type"]
                type_counts[t] = type_counts.get(t, 0) + 1

            return {
                "success": True,
                "version": version,
                "template": template,
                "total_fields": len(fields),
                "total_pages": len(pages_def),
                "field_types": type_counts,
                "fields": fields,
                "processing_time": round(time.time() - start_time, 2),
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Schema extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2),
            }

    @mcp_tool(
        name="validate_permit_form_data",
        description="""Validate form data against field definitions before filling.

Checks for:
- Missing required fields
- Extra fields not in schema
- Type mismatches (checkbox vs text)

Args:
    form_data: JSON object with field names and values
    field_definitions: Inline JSON with field definitions
    field_definitions_path: Path to JSON file with field definitions

Returns:
    Validation results with missing, extra, and invalid fields
"""
    )
    async def validate_permit_form_data(
        self,
        form_data: str,
        field_definitions: Optional[str] = None,
        field_definitions_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate form data against field definitions."""
        start_time = time.time()

        try:
            # Load field definitions
            field_defs = self._load_field_definitions(field_definitions, field_definitions_path)

            # Parse form data
            data = json.loads(form_data)
            if not isinstance(data, dict):
                raise ValueError("form_data must be a JSON object")

            # Build field lookup
            pages_def = field_defs.get("pages", {})
            coords = FieldCoordinates(pages_def)
            all_fields = coords.get_all_fields()

            # Check for missing and extra fields
            schema_field_names = set(all_fields.keys())
            data_field_names = set(data.keys())

            missing_fields = list(schema_field_names - data_field_names)
            extra_fields = list(data_field_names - schema_field_names)

            # Check required fields
            missing_required = []
            for field_name, field_info in all_fields.items():
                if field_info.get("required", False) and field_name not in data:
                    missing_required.append(field_name)

            # Type validation
            type_errors = []
            for field_name, value in data.items():
                if field_name not in all_fields:
                    continue
                field_info = all_fields[field_name]
                field_type = field_info.get("type", "text")

                if field_type == "checkbox":
                    if not isinstance(value, bool) and value not in [0, 1, "true", "false", True, False]:
                        type_errors.append({
                            "field": field_name,
                            "expected": "boolean",
                            "got": type(value).__name__,
                        })

            is_valid = len(missing_required) == 0 and len(type_errors) == 0

            return {
                "success": True,
                "valid": is_valid,
                "total_schema_fields": len(schema_field_names),
                "total_data_fields": len(data_field_names),
                "fields_matched": len(schema_field_names & data_field_names),
                "missing_required": missing_required,
                "missing_optional": [f for f in missing_fields if f not in missing_required],
                "extra_fields": extra_fields,
                "type_errors": type_errors,
                "processing_time": round(time.time() - start_time, 2),
            }

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {sanitize_error_message(str(e))}"
            logger.error(f"Validation failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2),
            }
        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Validation failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2),
            }

    @mcp_tool(
        name="preview_field_positions",
        description="""Generate a preview PDF showing field positions overlaid on template.

Creates a visualization with:
- Red rectangles showing field boundaries
- Field names as labels
- Page numbers

Useful for debugging field coordinate alignment.

Args:
    template_path: Path to the PDF template file
    field_definitions: Inline JSON with field definitions
    field_definitions_path: Path to JSON file with field definitions
    output_path: Optional path to save preview PDF

Returns:
    Preview PDF with field positions visualized
"""
    )
    async def preview_field_positions(
        self,
        template_path: str,
        field_definitions: Optional[str] = None,
        field_definitions_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a preview PDF showing field positions."""
        start_time = time.time()

        try:
            # Get reportlab (optional dependency)
            rl = _get_reportlab()

            # Validate template path
            template_file = await validate_pdf_path(template_path)

            # Load field definitions
            field_defs = self._load_field_definitions(field_definitions, field_definitions_path)
            pages_def = field_defs.get("pages", {})
            coords = FieldCoordinates(pages_def)

            # Read template
            template_pdf = PdfReader(str(template_file))
            writer = PdfWriter()

            # Process each page
            for page_idx, template_page in enumerate(template_pdf.pages):
                page_num = page_idx + 1

                # Create overlay with field boxes
                buffer = io.BytesIO()
                c = rl['canvas'].Canvas(buffer, pagesize=rl['letter'])

                # Semi-transparent red for boxes
                c.setStrokeColorRGB(1, 0, 0)  # Red stroke
                c.setFillColorRGB(1, 0, 0, 0.1)  # Light red fill
                c.setLineWidth(1)

                page_fields = coords.get_fields_for_page(page_num)

                for field_name in page_fields:
                    field = coords.get_raw(field_name)
                    if not field:
                        continue

                    # Get coordinates (convert y)
                    x = field["x"]
                    y = PAGE_HEIGHT - field["y"] - field["height"]
                    w = field["width"]
                    h = field["height"]

                    # Draw rectangle
                    c.rect(x, y, w, h, stroke=1, fill=1)

                    # Draw label above the box
                    c.setFillColorRGB(1, 0, 0)  # Red text
                    c.setFont("Helvetica", 6)
                    label = f"{field_name} ({field.get('type', 'text')})"
                    c.drawString(x, y + h + 2, label)

                    # Reset fill for next box
                    c.setFillColorRGB(1, 0, 0, 0.1)

                c.save()
                buffer.seek(0)

                # Merge overlay
                if page_fields:
                    overlay_pdf = PdfReader(buffer)
                    template_page.merge_page(overlay_pdf.pages[0])

                writer.add_page(template_page)

            # Write output
            output_buffer = io.BytesIO()
            writer.write(output_buffer)
            output_buffer.seek(0)
            pdf_bytes = output_buffer.getvalue()

            result = {
                "success": True,
                "total_pages": len(template_pdf.pages),
                "total_fields": len(coords.get_all_fields()),
                "processing_time": round(time.time() - start_time, 2),
            }

            if output_path:
                out_path = validate_output_path(output_path)
                with open(out_path, "wb") as f:
                    f.write(pdf_bytes)
                result["output_path"] = str(out_path)
            else:
                result["pdf_base64"] = base64.b64encode(pdf_bytes).decode("utf-8")

            return result

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Preview generation failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2),
            }

    @mcp_tool(
        name="insert_attachment_pages",
        description="""Insert attachment pages (images or text) into a PDF document.

Creates new pages with attachments and optionally adds "See page X" annotations
at specified field positions to reference the inserted pages.

Attachment JSON format:
[
  {
    "name": "field_name",           // Field name for tracking
    "page_title": "Site Photo",     // Title shown in header
    "insert_after_page": 2,         // Page to insert after (null = end)
    "content_type": "image",        // "image" or "text"
    "image_path": "/path/to/img",   // For images: path to image file
    "image_base64": "...",          // OR base64-encoded image data
    "text_content": "...",          // For text: the text to display
    "image_fit": "contain",         // "contain", "cover", or "stretch"
    "show_header": true,            // Show title header
    "add_reference": true,          // Add "See page X" at field position
    "field_page": 1,                // Page for "See page X" annotation
    "field_x": 100,                 // X position for annotation
    "field_y": 200,                 // Y position (from top)
    "field_width": 80,              // Annotation width
    "field_height": 14              // Annotation height
  }
]

Args:
    source_pdf_path: Path to the PDF to modify
    attachments: JSON array of attachment configurations
    output_path: Optional path to save output (else returns base64)

Returns:
    Modified PDF with attachment pages inserted
"""
    )
    async def insert_attachment_pages(
        self,
        source_pdf_path: str,
        attachments: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert attachment pages into a PDF document."""
        start_time = time.time()

        try:
            # Validate source path
            source_file = await validate_pdf_path(source_pdf_path)

            # Parse attachments JSON
            attachment_list = json.loads(attachments)
            if not isinstance(attachment_list, list):
                raise ValueError("attachments must be a JSON array")

            # Read source PDF
            source_pdf = PdfReader(str(source_file))
            original_page_count = len(source_pdf.pages)

            # Track insertions: {insert_position: [pdf_buffers...]}
            # insert_position is the 0-indexed page after which to insert
            # -1 means "at the end"
            insertions: Dict[int, List[io.BytesIO]] = {}

            # Track annotations to add: {page_index: [annotation_buffers...]}
            annotations: Dict[int, List[Dict[str, Any]]] = {}

            pages_created = 0
            attachment_results = []

            for idx, att in enumerate(attachment_list):
                try:
                    name = att.get("name", f"attachment_{idx}")
                    page_title = att.get("page_title", name)
                    insert_after = att.get("insert_after_page")  # 1-indexed or None
                    content_type = att.get("content_type", "image")
                    show_header = att.get("show_header", True)
                    add_reference = att.get("add_reference", False)

                    # Determine insertion position (convert to 0-indexed)
                    if insert_after is None:
                        insert_pos = -1  # End of document
                    else:
                        insert_pos = int(insert_after) - 1  # Convert to 0-indexed

                    # Create the attachment page
                    if content_type == "image":
                        # Load image data
                        image_data = None
                        filename = None

                        if "image_base64" in att:
                            image_data = base64.b64decode(att["image_base64"])
                        elif "image_path" in att:
                            img_path = Path(att["image_path"])
                            if not img_path.exists():
                                raise FileNotFoundError(f"Image not found: {att['image_path']}")
                            with open(img_path, "rb") as f:
                                image_data = f.read()
                            filename = img_path.name
                        else:
                            raise ValueError(f"Attachment {name}: must provide image_base64 or image_path")

                        # Get fit mode
                        fit_mode_str = att.get("image_fit", "contain")
                        try:
                            fit_mode = ImageFitMode(fit_mode_str)
                        except ValueError:
                            fit_mode = ImageFitMode.CONTAIN

                        # Create the page
                        page_buffer = _create_attachment_page_with_image(
                            title=page_title,
                            image_data=image_data,
                            fit_mode=fit_mode,
                            show_header=show_header,
                            filename=filename,
                        )

                    elif content_type == "text":
                        text_content = att.get("text_content", "")
                        page_buffer = _create_attachment_page_with_text(
                            title=page_title,
                            text_content=text_content,
                            show_header=show_header,
                        )
                    else:
                        raise ValueError(f"Attachment {name}: unknown content_type '{content_type}'")

                    # Queue the insertion
                    if insert_pos not in insertions:
                        insertions[insert_pos] = []
                    insertions[insert_pos].append(page_buffer)
                    pages_created += 1

                    # Track annotation if requested
                    if add_reference:
                        field_page = att.get("field_page")
                        if field_page:
                            page_idx = int(field_page) - 1  # Convert to 0-indexed
                            if page_idx not in annotations:
                                annotations[page_idx] = []
                            annotations[page_idx].append({
                                "name": name,
                                "title": page_title,
                                "x": att.get("field_x", 100),
                                "y": att.get("field_y", 100),
                                "width": att.get("field_width", 80),
                                "height": att.get("field_height", 14),
                            })

                    attachment_results.append({
                        "name": name,
                        "status": "success",
                        "content_type": content_type,
                    })

                except Exception as att_err:
                    attachment_results.append({
                        "name": att.get("name", f"attachment_{idx}"),
                        "status": "error",
                        "error": sanitize_error_message(str(att_err)),
                    })

            # Build the output PDF
            writer = PdfWriter()

            # Calculate final page numbers for inserted pages
            # This is complex because insertions shift page numbers
            inserted_page_numbers: Dict[str, int] = {}

            # First pass: calculate where each attachment will end up
            current_page = 0
            insert_count_before: Dict[int, int] = {}

            # Sort insertion positions
            sorted_positions = sorted(
                [p for p in insertions.keys() if p >= 0]
            )

            # Count insertions at each position
            for pos in sorted_positions:
                insert_count_before[pos] = len(insertions.get(pos, []))

            # Calculate cumulative offset at each original page
            cumulative_inserts = 0
            page_offset: Dict[int, int] = {}

            for orig_page in range(original_page_count):
                page_offset[orig_page] = cumulative_inserts
                if orig_page in insert_count_before:
                    cumulative_inserts += insert_count_before[orig_page]

            # Now build the PDF, adding pages in order
            current_output_page = 0

            for orig_page_idx in range(original_page_count):
                original_page = source_pdf.pages[orig_page_idx]

                # Add any annotations to this page
                if orig_page_idx in annotations:
                    for ann in annotations[orig_page_idx]:
                        # Calculate the target page number
                        # The inserted page comes after this original page
                        # plus any previous insertions
                        target_page = (
                            orig_page_idx + 1 +  # 1-indexed original position
                            page_offset[orig_page_idx] +  # Pages inserted before
                            1  # This annotation's page (first in the batch)
                        )

                        # Create annotation overlay
                        ann_buffer = _create_see_page_annotation(
                            x=ann["x"],
                            y=ann["y"],
                            width=ann["width"],
                            height=ann["height"],
                            target_page=target_page,
                            title=ann["title"],
                        )

                        # Merge annotation onto original page
                        ann_pdf = PdfReader(ann_buffer)
                        original_page.merge_page(ann_pdf.pages[0])

                        inserted_page_numbers[ann["name"]] = target_page

                writer.add_page(original_page)
                current_output_page += 1

                # Insert any pages that go after this original page
                if orig_page_idx in insertions:
                    for page_buffer in insertions[orig_page_idx]:
                        page_buffer.seek(0)
                        insert_pdf = PdfReader(page_buffer)
                        writer.add_page(insert_pdf.pages[0])
                        current_output_page += 1

            # Insert any pages that go at the end
            if -1 in insertions:
                for page_buffer in insertions[-1]:
                    page_buffer.seek(0)
                    insert_pdf = PdfReader(page_buffer)
                    writer.add_page(insert_pdf.pages[0])
                    current_output_page += 1

            # Write output
            output_buffer = io.BytesIO()
            writer.write(output_buffer)
            output_buffer.seek(0)
            pdf_bytes = output_buffer.getvalue()

            result = {
                "success": True,
                "original_pages": original_page_count,
                "pages_inserted": pages_created,
                "total_pages": current_output_page,
                "attachments": attachment_results,
                "inserted_page_numbers": inserted_page_numbers,
                "processing_time": round(time.time() - start_time, 2),
            }

            if output_path:
                out_path = validate_output_path(output_path)
                with open(out_path, "wb") as f:
                    f.write(pdf_bytes)
                result["output_path"] = str(out_path)
                result["output_size_bytes"] = len(pdf_bytes)
            else:
                result["pdf_base64"] = base64.b64encode(pdf_bytes).decode("utf-8")
                result["output_size_bytes"] = len(pdf_bytes)

            return result

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {sanitize_error_message(str(e))}"
            logger.error(f"Attachment insertion failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2),
            }
        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Attachment insertion failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2),
            }
