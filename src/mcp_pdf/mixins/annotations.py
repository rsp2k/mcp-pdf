"""
Annotations Mixin - PDF annotations, markup, and multimedia content
"""

import json
import time
import hashlib
import os
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


class AnnotationsMixin(MCPMixin):
    """
    Handles all PDF annotation operations including sticky notes, highlights,
    video notes, and annotation extraction.

    Tools provided:
    - add_sticky_notes: Add sticky note annotations to PDF
    - add_highlights: Add text highlights to PDF
    - add_video_notes: Add video annotations to PDF
    - extract_all_annotations: Extract all annotations from PDF
    """

    def get_mixin_name(self) -> str:
        return "Annotations"

    def get_required_permissions(self) -> List[str]:
        return ["read_files", "write_files", "annotation_processing"]

    def _setup(self):
        """Initialize annotations specific configuration"""
        self.color_map = {
            "yellow": (1, 1, 0),
            "red": (1, 0, 0),
            "green": (0, 1, 0),
            "blue": (0, 0, 1),
            "orange": (1, 0.5, 0),
            "purple": (0.5, 0, 1),
            "pink": (1, 0.75, 0.8),
            "gray": (0.5, 0.5, 0.5)
        }
        self.supported_video_formats = ['.mp4', '.mov', '.avi', '.mkv', '.webm']

    @mcp_tool(
        name="add_sticky_notes",
        description="Add sticky note annotations to PDF"
    )
    async def add_sticky_notes(
        self,
        input_path: str,
        output_path: str,
        notes: str  # JSON array of note definitions
    ) -> Dict[str, Any]:
        """
        Add sticky note annotations to PDF at specified locations.

        Args:
            input_path: Path to the existing PDF
            output_path: Path where PDF with notes should be saved
            notes: JSON array of note definitions

        Note format:
        [
            {
                "page": 1,
                "x": 100, "y": 200,
                "content": "This is a note",
                "author": "John Doe",
                "subject": "Review Comment",
                "color": "yellow"
            }
        ]

        Returns:
            Dictionary containing annotation results
        """
        start_time = time.time()

        try:
            # Parse notes
            try:
                note_definitions = self._safe_json_parse(notes) if notes else []
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid notes JSON: {str(e)}",
                    "annotation_time": 0
                }

            if not note_definitions:
                return {
                    "success": False,
                    "error": "At least one note is required",
                    "annotation_time": 0
                }

            # Validate input path
            input_file = await validate_pdf_path(input_path)
            output_file = validate_output_path(output_path)
            doc = fitz.open(str(input_file))

            annotation_info = {
                "notes_added": [],
                "annotation_errors": []
            }

            # Process each note
            for i, note_def in enumerate(note_definitions):
                try:
                    page_num = note_def.get("page", 1) - 1  # Convert to 0-indexed
                    x = note_def.get("x", 100)
                    y = note_def.get("y", 100)
                    content = note_def.get("content", "")
                    author = note_def.get("author", "Anonymous")
                    subject = note_def.get("subject", "Note")
                    color_name = note_def.get("color", "yellow").lower()

                    # Validate page number
                    if page_num >= len(doc) or page_num < 0:
                        annotation_info["annotation_errors"].append({
                            "note_index": i,
                            "error": f"Page {page_num + 1} does not exist"
                        })
                        continue

                    page = doc[page_num]

                    # Get color
                    color = self.color_map.get(color_name, (1, 1, 0))  # Default to yellow

                    # Create realistic sticky note appearance
                    note_width = 80
                    note_height = 60
                    note_rect = fitz.Rect(x, y, x + note_width, y + note_height)

                    # Add colored rectangle background (sticky note paper)
                    page.draw_rect(note_rect, color=color, fill=color, width=1)

                    # Add slight shadow effect for depth
                    shadow_rect = fitz.Rect(x + 2, y - 2, x + note_width + 2, y + note_height - 2)
                    page.draw_rect(shadow_rect, color=(0.7, 0.7, 0.7), fill=(0.7, 0.7, 0.7), width=0)

                    # Add the main sticky note rectangle on top
                    page.draw_rect(note_rect, color=color, fill=color, width=1)

                    # Add border for definition
                    border_color = (min(1, color[0] * 0.8), min(1, color[1] * 0.8), min(1, color[2] * 0.8))
                    page.draw_rect(note_rect, color=border_color, width=1)

                    # Add "folded corner" effect (small triangle)
                    fold_size = 8
                    fold_points = [
                        fitz.Point(x + note_width - fold_size, y),
                        fitz.Point(x + note_width, y),
                        fitz.Point(x + note_width, y + fold_size)
                    ]
                    page.draw_polyline(fold_points, color=(1, 1, 1), fill=(1, 1, 1), width=1)

                    # Add text content on the sticky note
                    words = content.split()
                    lines = []
                    current_line = []

                    for word in words:
                        test_line = " ".join(current_line + [word])
                        if len(test_line) > 12:  # Approximate character limit per line
                            if current_line:
                                lines.append(" ".join(current_line))
                                current_line = [word]
                            else:
                                lines.append(word[:12] + "...")
                                break
                        else:
                            current_line.append(word)

                    if current_line:
                        lines.append(" ".join(current_line))

                    # Limit to 4 lines to fit in sticky note
                    if len(lines) > 4:
                        lines = lines[:3] + [lines[3][:8] + "..."]

                    # Draw text lines
                    line_height = 10
                    text_y = y + 10
                    text_color = (0, 0, 0)  # Black text

                    for line in lines[:4]:  # Max 4 lines
                        if text_y + line_height <= y + note_height - 4:
                            page.insert_text((x + 6, text_y), line, fontname="helv", fontsize=8, color=text_color)
                            text_y += line_height

                    # Create invisible text annotation for PDF annotation system compatibility
                    annot = page.add_text_annot(fitz.Point(x + note_width/2, y + note_height/2), content)
                    annot.set_info(content=content, title=subject)
                    annot.set_colors(stroke=(0, 0, 0, 0), fill=color)
                    annot.set_flags(fitz.PDF_ANNOT_IS_PRINT | fitz.PDF_ANNOT_IS_INVISIBLE)
                    annot.update()

                    annotation_info["notes_added"].append({
                        "page": page_num + 1,
                        "position": {"x": x, "y": y},
                        "content": content[:50] + "..." if len(content) > 50 else content,
                        "author": author,
                        "subject": subject,
                        "color": color_name
                    })

                except Exception as e:
                    annotation_info["annotation_errors"].append({
                        "note_index": i,
                        "error": f"Failed to add note: {str(e)}"
                    })

            # Save PDF with annotations
            doc.save(str(output_file), garbage=4, deflate=True, clean=True)
            doc.close()

            file_size = output_file.stat().st_size

            return {
                "success": True,
                "input_path": str(input_file),
                "output_path": str(output_file),
                "notes_requested": len(note_definitions),
                "notes_added": len(annotation_info["notes_added"]),
                "notes_failed": len(annotation_info["annotation_errors"]),
                "note_details": annotation_info["notes_added"],
                "errors": annotation_info["annotation_errors"],
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "annotation_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Sticky notes addition failed: {error_msg}")
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
        highlights: str  # JSON array of highlight definitions
    ) -> Dict[str, Any]:
        """
        Add highlight annotations to PDF text or specific areas.

        Args:
            input_path: Path to the existing PDF
            output_path: Path where PDF with highlights should be saved
            highlights: JSON array of highlight definitions

        Highlight format:
        [
            {
                "page": 1,
                "text": "text to highlight",  // Optional: search for this text
                "rect": [x0, y0, x1, y1],  // Optional: specific rectangle
                "color": "yellow",
                "author": "John Doe",
                "note": "Important point"
            }
        ]

        Returns:
            Dictionary containing highlight results
        """
        start_time = time.time()

        try:
            # Parse highlights
            try:
                highlight_definitions = self._safe_json_parse(highlights) if highlights else []
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid highlights JSON: {str(e)}",
                    "highlight_time": 0
                }

            if not highlight_definitions:
                return {
                    "success": False,
                    "error": "At least one highlight is required",
                    "highlight_time": 0
                }

            # Validate input path
            input_file = await validate_pdf_path(input_path)
            output_file = validate_output_path(output_path)
            doc = fitz.open(str(input_file))

            highlight_info = {
                "highlights_added": [],
                "highlight_errors": []
            }

            # Process each highlight
            for i, highlight_def in enumerate(highlight_definitions):
                try:
                    page_num = highlight_def.get("page", 1) - 1  # Convert to 0-indexed
                    text_to_find = highlight_def.get("text", "")
                    rect_coords = highlight_def.get("rect", None)
                    color_name = highlight_def.get("color", "yellow").lower()
                    author = highlight_def.get("author", "Anonymous")
                    note = highlight_def.get("note", "")

                    # Validate page number
                    if page_num >= len(doc) or page_num < 0:
                        highlight_info["highlight_errors"].append({
                            "highlight_index": i,
                            "error": f"Page {page_num + 1} does not exist"
                        })
                        continue

                    page = doc[page_num]
                    color = self.color_map.get(color_name, (1, 1, 0))

                    highlights_added_this_item = 0

                    # Method 1: Search for text and highlight
                    if text_to_find:
                        text_instances = page.search_for(text_to_find)
                        for rect in text_instances:
                            # Create highlight annotation
                            annot = page.add_highlight_annot(rect)
                            annot.set_colors(stroke=color)
                            annot.set_info(content=note)
                            annot.update()
                            highlights_added_this_item += 1

                    # Method 2: Highlight specific rectangle
                    elif rect_coords and len(rect_coords) == 4:
                        highlight_rect = fitz.Rect(rect_coords[0], rect_coords[1],
                                                 rect_coords[2], rect_coords[3])
                        annot = page.add_highlight_annot(highlight_rect)
                        annot.set_colors(stroke=color)
                        annot.set_info(content=note)
                        annot.update()
                        highlights_added_this_item += 1

                    else:
                        highlight_info["highlight_errors"].append({
                            "highlight_index": i,
                            "error": "Must specify either 'text' to search for or 'rect' coordinates"
                        })
                        continue

                    if highlights_added_this_item > 0:
                        highlight_info["highlights_added"].append({
                            "page": page_num + 1,
                            "text_searched": text_to_find,
                            "rect_used": rect_coords,
                            "instances_highlighted": highlights_added_this_item,
                            "color": color_name,
                            "author": author,
                            "note": note[:50] + "..." if len(note) > 50 else note
                        })
                    else:
                        highlight_info["highlight_errors"].append({
                            "highlight_index": i,
                            "error": f"No text found to highlight: '{text_to_find}'"
                        })

                except Exception as e:
                    highlight_info["highlight_errors"].append({
                        "highlight_index": i,
                        "error": f"Failed to add highlight: {str(e)}"
                    })

            # Save PDF with highlights
            doc.save(str(output_file), garbage=4, deflate=True, clean=True)
            doc.close()

            file_size = output_file.stat().st_size

            return {
                "success": True,
                "input_path": str(input_file),
                "output_path": str(output_file),
                "highlights_requested": len(highlight_definitions),
                "highlights_added": len(highlight_info["highlights_added"]),
                "highlights_failed": len(highlight_info["highlight_errors"]),
                "highlight_details": highlight_info["highlights_added"],
                "errors": highlight_info["highlight_errors"],
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "highlight_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Highlight addition failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "highlight_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="add_video_notes",
        description="Add video annotations to PDF"
    )
    async def add_video_notes(
        self,
        input_path: str,
        output_path: str,
        video_notes: str  # JSON array of video note definitions
    ) -> Dict[str, Any]:
        """
        Add video sticky notes that embed video files and launch on click.

        Args:
            input_path: Path to the existing PDF
            output_path: Path where PDF with video notes should be saved
            video_notes: JSON array of video note definitions

        Video note format:
        [
            {
                "page": 1,
                "x": 100, "y": 200,
                "video_path": "/path/to/video.mp4",
                "title": "Demo Video",
                "color": "red",
                "size": "medium"
            }
        ]

        Returns:
            Dictionary containing video embedding results
        """
        start_time = time.time()

        try:
            # Parse video notes
            try:
                note_definitions = self._safe_json_parse(video_notes) if video_notes else []
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid video notes JSON: {str(e)}",
                    "embedding_time": 0
                }

            if not note_definitions:
                return {
                    "success": False,
                    "error": "At least one video note is required",
                    "embedding_time": 0
                }

            # Validate input path
            input_file = await validate_pdf_path(input_path)
            output_file = validate_output_path(output_path)
            doc = fitz.open(str(input_file))

            embedding_info = {
                "videos_embedded": [],
                "embedding_errors": []
            }

            # Size mapping
            size_map = {
                "small": (60, 45),
                "medium": (80, 60),
                "large": (100, 75)
            }

            # Process each video note
            for i, note_def in enumerate(note_definitions):
                try:
                    page_num = note_def.get("page", 1) - 1  # Convert to 0-indexed
                    x = note_def.get("x", 100)
                    y = note_def.get("y", 100)
                    video_path = note_def.get("video_path", "")
                    title = note_def.get("title", "Video")
                    color_name = note_def.get("color", "red").lower()
                    size_name = note_def.get("size", "medium").lower()

                    # Validate inputs
                    if not video_path or not os.path.exists(video_path):
                        embedding_info["embedding_errors"].append({
                            "note_index": i,
                            "error": f"Video file not found: {video_path}"
                        })
                        continue

                    # Check video format
                    video_ext = os.path.splitext(video_path)[1].lower()
                    if video_ext not in self.supported_video_formats:
                        embedding_info["embedding_errors"].append({
                            "note_index": i,
                            "error": f"Unsupported video format: {video_ext}. Supported: {', '.join(self.supported_video_formats)}",
                            "conversion_suggestion": f"Convert with FFmpeg: ffmpeg -i '{os.path.basename(video_path)}' -c:v libx264 -c:a aac -preset medium '{os.path.splitext(os.path.basename(video_path))[0]}.mp4'"
                        })
                        continue

                    # Validate page number
                    if page_num >= len(doc) or page_num < 0:
                        embedding_info["embedding_errors"].append({
                            "note_index": i,
                            "error": f"Page {page_num + 1} does not exist"
                        })
                        continue

                    page = doc[page_num]
                    color = self.color_map.get(color_name, (1, 0, 0))  # Default to red
                    note_width, note_height = size_map.get(size_name, (80, 60))

                    # Create video note visual
                    note_rect = fitz.Rect(x, y, x + note_width, y + note_height)

                    # Add colored background
                    page.draw_rect(note_rect, color=color, fill=color, width=1)

                    # Add play button icon
                    play_size = min(note_width, note_height) // 3
                    play_center_x = x + note_width // 2
                    play_center_y = y + note_height // 2

                    # Draw play triangle
                    play_points = [
                        fitz.Point(play_center_x - play_size//2, play_center_y - play_size//2),
                        fitz.Point(play_center_x - play_size//2, play_center_y + play_size//2),
                        fitz.Point(play_center_x + play_size//2, play_center_y)
                    ]
                    page.draw_polyline(play_points, color=(1, 1, 1), fill=(1, 1, 1), width=1)

                    # Add title text
                    title_rect = fitz.Rect(x, y + note_height + 2, x + note_width, y + note_height + 15)
                    page.insert_text(title_rect.tl, title[:15], fontname="helv", fontsize=8, color=(0, 0, 0))

                    # Embed video file as attachment
                    video_name = f"video_{i}_{os.path.basename(video_path)}"
                    with open(video_path, 'rb') as video_file:
                        video_data = video_file.read()

                    # Create file attachment
                    file_spec = doc.embfile_add(video_name, video_data, filename=os.path.basename(video_path))

                    # Create file attachment annotation
                    attachment_annot = page.add_file_annot(fitz.Point(x + note_width//2, y + note_height//2), video_data, filename=video_name)
                    attachment_annot.set_info(content=f"Video: {title}")
                    attachment_annot.update()

                    embedding_info["videos_embedded"].append({
                        "page": page_num + 1,
                        "position": {"x": x, "y": y},
                        "video_file": os.path.basename(video_path),
                        "title": title,
                        "color": color_name,
                        "size": size_name,
                        "file_size_mb": round(len(video_data) / (1024 * 1024), 2)
                    })

                except Exception as e:
                    embedding_info["embedding_errors"].append({
                        "note_index": i,
                        "error": f"Failed to embed video: {str(e)}"
                    })

            # Save PDF with video notes
            doc.save(str(output_file), garbage=4, deflate=True, clean=True)
            doc.close()

            file_size = output_file.stat().st_size

            return {
                "success": True,
                "input_path": str(input_file),
                "output_path": str(output_file),
                "videos_requested": len(note_definitions),
                "videos_embedded": len(embedding_info["videos_embedded"]),
                "videos_failed": len(embedding_info["embedding_errors"]),
                "video_details": embedding_info["videos_embedded"],
                "errors": embedding_info["embedding_errors"],
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "embedding_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Video notes addition failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "embedding_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="extract_all_annotations",
        description="Extract all annotations from PDF"
    )
    async def extract_all_annotations(
        self,
        pdf_path: str,
        export_format: str = "json"  # json, csv
    ) -> Dict[str, Any]:
        """
        Extract all annotations from PDF and export to JSON or CSV format.

        Args:
            pdf_path: Path to the PDF file to analyze
            export_format: Output format (json or csv)

        Returns:
            Dictionary containing all extracted annotations
        """
        start_time = time.time()

        try:
            # Validate input path
            input_file = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(input_file))

            all_annotations = []
            annotation_summary = {
                "total_annotations": 0,
                "by_type": {},
                "by_page": {},
                "authors": set()
            }

            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_annotations = []

                # Get all annotations on this page
                for annot in page.annots():
                    try:
                        annot_info = {
                            "page": page_num + 1,
                            "type": annot.type[1],  # Get annotation type name
                            "content": annot.info.get("content", ""),
                            "author": annot.info.get("title", "") or annot.info.get("author", ""),
                            "subject": annot.info.get("subject", ""),
                            "creation_date": str(annot.info.get("creationDate", "")),
                            "modification_date": str(annot.info.get("modDate", "")),
                            "rect": {
                                "x0": round(annot.rect.x0, 2),
                                "y0": round(annot.rect.y0, 2),
                                "x1": round(annot.rect.x1, 2),
                                "y1": round(annot.rect.y1, 2)
                            }
                        }

                        # Get colors if available
                        try:
                            stroke_color = annot.colors.get("stroke")
                            fill_color = annot.colors.get("fill")
                            if stroke_color:
                                annot_info["stroke_color"] = stroke_color
                            if fill_color:
                                annot_info["fill_color"] = fill_color
                        except:
                            pass

                        # For highlight annotations, try to get highlighted text
                        if annot.type[1] == "Highlight":
                            try:
                                highlighted_text = page.get_textbox(annot.rect)
                                if highlighted_text.strip():
                                    annot_info["highlighted_text"] = highlighted_text.strip()
                            except:
                                pass

                        all_annotations.append(annot_info)
                        page_annotations.append(annot_info)

                        # Update summary
                        annotation_type = annot_info["type"]
                        annotation_summary["by_type"][annotation_type] = annotation_summary["by_type"].get(annotation_type, 0) + 1

                        if annot_info["author"]:
                            annotation_summary["authors"].add(annot_info["author"])

                    except Exception as e:
                        # Skip problematic annotations
                        continue

                # Update page summary
                if page_annotations:
                    annotation_summary["by_page"][page_num + 1] = len(page_annotations)

            doc.close()

            annotation_summary["total_annotations"] = len(all_annotations)
            annotation_summary["authors"] = list(annotation_summary["authors"])

            # Format output based on requested format
            if export_format.lower() == "csv":
                # Convert to CSV-friendly format
                csv_data = []
                for annot in all_annotations:
                    csv_row = {
                        "page": annot["page"],
                        "type": annot["type"],
                        "content": annot["content"],
                        "author": annot["author"],
                        "subject": annot["subject"],
                        "x0": annot["rect"]["x0"],
                        "y0": annot["rect"]["y0"],
                        "x1": annot["rect"]["x1"],
                        "y1": annot["rect"]["y1"],
                        "highlighted_text": annot.get("highlighted_text", "")
                    }
                    csv_data.append(csv_row)

                return {
                    "success": True,
                    "input_path": str(input_file),
                    "export_format": "csv",
                    "csv_data": csv_data,
                    "summary": annotation_summary,
                    "extraction_time": round(time.time() - start_time, 2)
                }
            else:
                # JSON format (default)
                return {
                    "success": True,
                    "input_path": str(input_file),
                    "export_format": "json",
                    "annotations": all_annotations,
                    "summary": annotation_summary,
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

    # Private helper methods (synchronous for proper async pattern)
    def _safe_json_parse(self, json_str: str, max_size: int = MAX_JSON_SIZE) -> list:
        """Safely parse JSON with size limits"""
        if not json_str:
            return []

        if len(json_str) > max_size:
            raise ValueError(f"JSON input too large: {len(json_str)} > {max_size}")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")