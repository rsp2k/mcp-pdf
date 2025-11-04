"""
Annotations Mixin - PDF annotation and markup operations
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


class AnnotationsMixin(MCPMixin):
    """
    Handles PDF annotation operations including sticky notes, highlights, and stamps.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()
        self.max_file_size = 100 * 1024 * 1024  # 100MB

    @mcp_tool(
        name="add_sticky_notes",
        description="Add sticky note annotations to PDF"
    )
    async def add_sticky_notes(
        self,
        input_path: str,
        output_path: str,
        notes: str
    ) -> Dict[str, Any]:
        """
        Add sticky note annotations to specific locations in PDF.

        Args:
            input_path: Path to input PDF file
            output_path: Path where annotated PDF will be saved
            notes: JSON string containing note definitions

        Returns:
            Dictionary containing annotation results
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_pdf_path = await validate_output_path(output_path)

            # Parse notes data
            try:
                notes_list = json.loads(notes)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON in notes: {e}",
                    "annotation_time": round(time.time() - start_time, 2)
                }

            if not isinstance(notes_list, list):
                return {
                    "success": False,
                    "error": "notes must be a list of note objects",
                    "annotation_time": round(time.time() - start_time, 2)
                }

            # Open PDF document
            doc = fitz.open(str(input_pdf_path))
            total_pages = len(doc)
            notes_added = 0
            notes_failed = 0
            failed_notes = []

            for i, note_def in enumerate(notes_list):
                try:
                    page_num = note_def.get("page", 1) - 1  # Convert to 0-based
                    if page_num < 0 or page_num >= total_pages:
                        failed_notes.append({
                            "note_index": i + 1,
                            "error": f"Page {page_num + 1} out of range (1-{total_pages})"
                        })
                        notes_failed += 1
                        continue

                    page = doc[page_num]

                    # Get position
                    x = note_def.get("x", 100)
                    y = note_def.get("y", 100)
                    content = note_def.get("content", "Note")
                    author = note_def.get("author", "User")

                    # Create sticky note annotation
                    point = fitz.Point(x, y)
                    text_annot = page.add_text_annot(point, content)

                    # Set annotation properties
                    text_annot.set_info(content=content, title=author)
                    text_annot.set_colors({"stroke": (1, 1, 0)})  # Yellow
                    text_annot.update()

                    notes_added += 1

                except Exception as e:
                    failed_notes.append({
                        "note_index": i + 1,
                        "error": str(e)
                    })
                    notes_failed += 1

            # Save annotated PDF
            doc.save(str(output_pdf_path), incremental=False)
            output_size = output_pdf_path.stat().st_size
            doc.close()

            return {
                "success": True,
                "annotation_summary": {
                    "notes_requested": len(notes_list),
                    "notes_added": notes_added,
                    "notes_failed": notes_failed,
                    "output_size_bytes": output_size
                },
                "failed_notes": failed_notes,
                "output_info": {
                    "output_path": str(output_pdf_path),
                    "total_pages": total_pages
                },
                "annotation_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Sticky notes annotation failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "annotation_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="add_highlights",
        description="Add text highlights to PDF"
    )
    async def add_highlights(
        self,
        input_path: str,
        output_path: str,
        highlights: str
    ) -> Dict[str, Any]:
        """
        Add text highlights to specific areas in PDF.

        Args:
            input_path: Path to input PDF file
            output_path: Path where highlighted PDF will be saved
            highlights: JSON string containing highlight definitions

        Returns:
            Dictionary containing highlighting results
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_pdf_path = await validate_output_path(output_path)

            # Parse highlights data
            try:
                highlights_list = json.loads(highlights)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON in highlights: {e}",
                    "highlight_time": round(time.time() - start_time, 2)
                }

            # Open PDF document
            doc = fitz.open(str(input_pdf_path))
            total_pages = len(doc)
            highlights_added = 0
            highlights_failed = 0
            failed_highlights = []

            for i, highlight_def in enumerate(highlights_list):
                try:
                    page_num = highlight_def.get("page", 1) - 1  # Convert to 0-based
                    if page_num < 0 or page_num >= total_pages:
                        failed_highlights.append({
                            "highlight_index": i + 1,
                            "error": f"Page {page_num + 1} out of range (1-{total_pages})"
                        })
                        highlights_failed += 1
                        continue

                    page = doc[page_num]

                    # Get highlight area
                    if "text" in highlight_def:
                        # Search for text to highlight
                        search_text = highlight_def["text"]
                        text_instances = page.search_for(search_text)

                        for rect in text_instances:
                            highlight = page.add_highlight_annot(rect)
                            # Set color (default yellow)
                            color = highlight_def.get("color", "yellow")
                            color_map = {
                                "yellow": (1, 1, 0),
                                "green": (0, 1, 0),
                                "blue": (0, 0, 1),
                                "red": (1, 0, 0),
                                "orange": (1, 0.5, 0),
                                "pink": (1, 0.75, 0.8)
                            }
                            highlight.set_colors({"stroke": color_map.get(color, (1, 1, 0))})
                            highlight.update()
                            highlights_added += 1

                    elif all(k in highlight_def for k in ["x1", "y1", "x2", "y2"]):
                        # Manual rectangle highlighting
                        rect = fitz.Rect(
                            highlight_def["x1"],
                            highlight_def["y1"],
                            highlight_def["x2"],
                            highlight_def["y2"]
                        )
                        highlight = page.add_highlight_annot(rect)

                        # Set color
                        color = highlight_def.get("color", "yellow")
                        color_map = {
                            "yellow": (1, 1, 0),
                            "green": (0, 1, 0),
                            "blue": (0, 0, 1),
                            "red": (1, 0, 0),
                            "orange": (1, 0.5, 0),
                            "pink": (1, 0.75, 0.8)
                        }
                        highlight.set_colors({"stroke": color_map.get(color, (1, 1, 0))})
                        highlight.update()
                        highlights_added += 1

                    else:
                        failed_highlights.append({
                            "highlight_index": i + 1,
                            "error": "Missing text or coordinates (x1, y1, x2, y2)"
                        })
                        highlights_failed += 1

                except Exception as e:
                    failed_highlights.append({
                        "highlight_index": i + 1,
                        "error": str(e)
                    })
                    highlights_failed += 1

            # Save highlighted PDF
            doc.save(str(output_pdf_path), incremental=False)
            output_size = output_pdf_path.stat().st_size
            doc.close()

            return {
                "success": True,
                "highlight_summary": {
                    "highlights_requested": len(highlights_list),
                    "highlights_added": highlights_added,
                    "highlights_failed": highlights_failed,
                    "output_size_bytes": output_size
                },
                "failed_highlights": failed_highlights,
                "output_info": {
                    "output_path": str(output_pdf_path),
                    "total_pages": total_pages
                },
                "highlight_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Text highlighting failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "highlight_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="add_stamps",
        description="Add approval stamps to PDF"
    )
    async def add_stamps(
        self,
        input_path: str,
        output_path: str,
        stamps: str
    ) -> Dict[str, Any]:
        """
        Add approval stamps (Approved, Draft, Confidential, etc) to PDF.

        Args:
            input_path: Path to input PDF file
            output_path: Path where stamped PDF will be saved
            stamps: JSON string containing stamp definitions

        Returns:
            Dictionary containing stamping results
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_pdf_path = await validate_output_path(output_path)

            # Parse stamps data
            try:
                stamps_list = json.loads(stamps)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON in stamps: {e}",
                    "stamp_time": round(time.time() - start_time, 2)
                }

            # Open PDF document
            doc = fitz.open(str(input_pdf_path))
            total_pages = len(doc)
            stamps_added = 0
            stamps_failed = 0
            failed_stamps = []

            for i, stamp_def in enumerate(stamps_list):
                try:
                    page_num = stamp_def.get("page", 1) - 1  # Convert to 0-based
                    if page_num < 0 or page_num >= total_pages:
                        failed_stamps.append({
                            "stamp_index": i + 1,
                            "error": f"Page {page_num + 1} out of range (1-{total_pages})"
                        })
                        stamps_failed += 1
                        continue

                    page = doc[page_num]

                    # Get stamp properties
                    x = stamp_def.get("x", 400)
                    y = stamp_def.get("y", 50)
                    stamp_type = stamp_def.get("type", "APPROVED")
                    size = stamp_def.get("size", "medium")

                    # Size mapping
                    size_map = {
                        "small": (80, 30),
                        "medium": (120, 40),
                        "large": (160, 50)
                    }
                    width, height = size_map.get(size, (120, 40))

                    # Color mapping for different stamp types
                    color_map = {
                        "APPROVED": (0, 0.7, 0),    # Green
                        "REJECTED": (0.8, 0, 0),    # Red
                        "DRAFT": (0, 0, 0.8),       # Blue
                        "CONFIDENTIAL": (0.8, 0, 0.8), # Purple
                        "REVIEWED": (0.5, 0.5, 0),  # Olive
                        "FINAL": (0, 0, 0),         # Black
                        "COPY": (0.5, 0.5, 0.5)    # Gray
                    }

                    # Create stamp rectangle
                    stamp_rect = fitz.Rect(x, y, x + width, y + height)

                    # Add rectangular annotation for stamp background
                    stamp_annot = page.add_rect_annot(stamp_rect)
                    stamp_color = color_map.get(stamp_type.upper(), (0.8, 0, 0))
                    stamp_annot.set_colors({"stroke": stamp_color, "fill": stamp_color})
                    stamp_annot.set_border(width=2)
                    stamp_annot.update()

                    # Add text on top of the stamp
                    text_point = fitz.Point(x + width/2, y + height/2)
                    text_annot = page.add_text_annot(text_point, stamp_type.upper())
                    text_annot.set_info(content=stamp_type.upper())
                    text_annot.update()

                    # Add text using insert_text for better visibility
                    page.insert_text(
                        text_point,
                        stamp_type.upper(),
                        fontsize=12,
                        color=(1, 1, 1),  # White text
                        fontname="helv-bold"
                    )

                    stamps_added += 1

                except Exception as e:
                    failed_stamps.append({
                        "stamp_index": i + 1,
                        "error": str(e)
                    })
                    stamps_failed += 1

            # Save stamped PDF
            doc.save(str(output_pdf_path), incremental=False)
            output_size = output_pdf_path.stat().st_size
            doc.close()

            return {
                "success": True,
                "stamp_summary": {
                    "stamps_requested": len(stamps_list),
                    "stamps_added": stamps_added,
                    "stamps_failed": stamps_failed,
                    "output_size_bytes": output_size
                },
                "failed_stamps": failed_stamps,
                "available_stamp_types": list(color_map.keys()),
                "output_info": {
                    "output_path": str(output_pdf_path),
                    "total_pages": total_pages
                },
                "stamp_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Stamp annotation failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "stamp_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="extract_all_annotations",
        description="Extract all annotations from PDF"
    )
    async def extract_all_annotations(
        self,
        pdf_path: str,
        export_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Extract all annotations (notes, highlights, stamps) from PDF.

        Args:
            pdf_path: Path to PDF file
            export_format: Output format ("json", "csv", "text")

        Returns:
            Dictionary containing all annotations
        """
        start_time = time.time()

        try:
            # Validate path
            input_pdf_path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(input_pdf_path))

            all_annotations = []
            annotation_stats = {
                "text": 0,
                "highlight": 0,
                "ink": 0,
                "square": 0,
                "circle": 0,
                "line": 0,
                "freetext": 0,
                "stamp": 0,
                "other": 0
            }

            for page_num in range(len(doc)):
                page = doc[page_num]

                try:
                    annotations = page.annots()

                    for annot in annotations:
                        annot_dict = annot.info

                        annotation_data = {
                            "page": page_num + 1,
                            "type": annot_dict.get("name", "unknown"),
                            "content": annot_dict.get("content", ""),
                            "title": annot_dict.get("title", ""),
                            "subject": annot_dict.get("subject", ""),
                            "creation_date": annot_dict.get("creationDate", ""),
                            "modification_date": annot_dict.get("modDate", ""),
                            "coordinates": {
                                "x1": round(annot.rect.x0, 2),
                                "y1": round(annot.rect.y0, 2),
                                "x2": round(annot.rect.x1, 2),
                                "y2": round(annot.rect.y1, 2)
                            }
                        }

                        all_annotations.append(annotation_data)

                        # Update statistics
                        annot_type = annotation_data["type"].lower()
                        if annot_type in annotation_stats:
                            annotation_stats[annot_type] += 1
                        else:
                            annotation_stats["other"] += 1

                except Exception as e:
                    logger.warning(f"Failed to extract annotations from page {page_num + 1}: {e}")

            doc.close()

            # Format output based on requested format
            if export_format == "csv":
                # Convert to CSV-like structure
                csv_data = []
                for annot in all_annotations:
                    csv_data.append({
                        "Page": annot["page"],
                        "Type": annot["type"],
                        "Content": annot["content"],
                        "Title": annot["title"],
                        "X1": annot["coordinates"]["x1"],
                        "Y1": annot["coordinates"]["y1"],
                        "X2": annot["coordinates"]["x2"],
                        "Y2": annot["coordinates"]["y2"]
                    })
                formatted_data = csv_data

            elif export_format == "text":
                # Convert to readable text format
                text_lines = []
                for annot in all_annotations:
                    text_lines.append(
                        f"Page {annot['page']} [{annot['type']}]: {annot['content']} "
                        f"by {annot['title']} at ({annot['coordinates']['x1']}, {annot['coordinates']['y1']})"
                    )
                formatted_data = "\n".join(text_lines)

            else:  # json (default)
                formatted_data = all_annotations

            return {
                "success": True,
                "annotation_summary": {
                    "total_annotations": len(all_annotations),
                    "annotation_types": annotation_stats,
                    "export_format": export_format
                },
                "annotations": formatted_data,
                "file_info": {
                    "path": str(input_pdf_path),
                    "total_pages": len(doc) if 'doc' in locals() else 0
                },
                "extraction_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Annotation extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "extraction_time": round(time.time() - start_time, 2)
            }