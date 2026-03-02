"""
Structure Detection Mixin - Detect document structure via bookmarks, font analysis,
and numbering/regex patterns. Produces hierarchical section trees and flat boundary
lists suitable for downstream splitting and batch extraction.

Uses official fastmcp.contrib.mcp_mixin pattern.
"""

import json
import re
import time
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import fitz  # PyMuPDF

from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, validate_output_path, sanitize_error_message
from .utils import parse_pages_parameter
from .image_processing import ImageProcessingMixin

logger = logging.getLogger(__name__)

# Common section-heading patterns (case-insensitive)
_NUMBERING_PATTERNS = [
    # "Chapter 1", "CHAPTER IV"
    (r"^(?:chapter|ch\.?)\s+(?:\d+|[IVXLCDM]+)", 1),
    # "Part 1", "PART III"
    (r"^(?:part)\s+(?:\d+|[IVXLCDM]+)", 1),
    # "ANNEX A", "Annex 1"
    (r"^(?:annex|appendix)\s+[A-Z0-9]+", 1),
    # "Section 2.3"
    (r"^(?:section)\s+\d+(?:\.\d+)*", 2),
    # "1.2.3 Title text" (numbered headings like 1., 1.2, 1.2.3)
    (r"^\d+\.\d+\.\d+(?:\.\d+)*\s+\S", 3),
    (r"^\d+\.\d+\s+\S", 2),
    (r"^\d+\.\s+\S", 1),
]


class StructureDetectionMixin(MCPMixin):
    """
    Detects document structure from bookmarks, font-size analysis, and
    numbering/regex patterns.  Produces a hierarchical section tree and a
    flat boundary list that downstream tools (split_pdf_by_structure,
    batch_extract) can consume directly.

    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Public MCP tool
    # ------------------------------------------------------------------

    @mcp_tool(
        name="detect_structure",
        description=(
            "Detect logical structure (chapters, sections, headings) of a PDF "
            "using bookmarks, font-size analysis, and numbering patterns. "
            "Returns a hierarchical section tree and a flat boundary list "
            "with confidence scores for each detected heading."
        ),
    )
    async def detect_structure(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        strategies: str = "auto",
        heading_pattern: Optional[str] = None,
        max_heading_levels: int = 3,
        min_confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Detect logical document structure.

        Args:
            pdf_path: Path to PDF file or HTTPS URL.
            pages: Pages to analyse (comma-separated, 1-based). None = all.
            strategies: Detection strategy —
                "auto"      try bookmarks first, always run fonts, cross-validate.
                "bookmarks" bookmarks only.
                "fonts"     font-size heuristic only.
                "numbering" regex / numbering patterns only.
                "all"       run every strategy and merge.
            heading_pattern: Optional user-supplied regex for headings.
            max_heading_levels: Maximum heading depth to report (1-6).
            min_confidence: Drop boundaries below this confidence (0-1).

        Returns:
            Dict with success flag, hierarchical structure, flat boundaries,
            detection metadata, and timing.
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))
            total_pages = len(doc)

            # Determine which pages to process
            parsed_pages = parse_pages_parameter(pages)
            if parsed_pages is not None:
                pages_to_process = sorted(
                    p for p in parsed_pages if 0 <= p < total_pages
                )
            else:
                pages_to_process = list(range(total_pages))

            if not pages_to_process:
                pages_to_process = list(range(total_pages))

            max_heading_levels = max(1, min(6, max_heading_levels))

            # Collect detections per strategy
            all_detections: List[List[Dict[str, Any]]] = []
            strategies_used: List[str] = []
            bookmarks_found = 0
            body_font_info: Dict[str, Any] = {}
            heading_font_info: Dict[int, Dict[str, Any]] = {}

            strategies_lower = strategies.strip().lower()

            # --- Bookmarks ---
            run_bookmarks = strategies_lower in ("auto", "bookmarks", "all")
            bookmark_detections: List[Dict[str, Any]] = []
            if run_bookmarks:
                try:
                    bookmark_detections = self._detect_by_bookmarks(doc)
                    bookmarks_found = len(bookmark_detections)
                    if bookmark_detections:
                        strategies_used.append("bookmarks")
                        all_detections.append(bookmark_detections)
                except Exception as exc:
                    logger.warning("Bookmark detection failed: %s", exc)

            # --- Fonts ---
            run_fonts = strategies_lower in ("auto", "fonts", "all")
            if run_fonts:
                try:
                    font_detections, body_info, heading_info = (
                        self._detect_by_fonts(doc, pages_to_process, max_heading_levels)
                    )
                    body_font_info = body_info
                    heading_font_info = heading_info
                    if font_detections:
                        strategies_used.append("fonts")
                        all_detections.append(font_detections)
                except Exception as exc:
                    logger.warning("Font-based detection failed: %s", exc)

            # --- Numbering / built-in patterns ---
            run_numbering = strategies_lower in ("auto", "numbering", "all")
            if run_numbering:
                try:
                    numbering_detections = self._detect_by_numbering(
                        doc, pages_to_process
                    )
                    if numbering_detections:
                        strategies_used.append("numbering")
                        all_detections.append(numbering_detections)
                except Exception as exc:
                    logger.warning("Numbering detection failed: %s", exc)

            # --- User-supplied regex ---
            if heading_pattern:
                try:
                    user_detections = self._detect_by_pattern(
                        doc, pages_to_process, heading_pattern
                    )
                    if user_detections:
                        strategies_used.append("user_regex")
                        all_detections.append(user_detections)
                except Exception as exc:
                    logger.warning("User-regex detection failed: %s", exc)

            # Auto-mode cross-validation: if bookmarks are sparse but exist,
            # still include font detections; if bookmarks are rich (>=3),
            # treat them as primary and boost font matches on the same pages.
            # (The merge step handles the boosting automatically.)

            doc.close()

            # Merge all detections
            merged = self._merge_detections(*all_detections)

            # Filter by min_confidence and max_heading_levels
            filtered = [
                b for b in merged
                if b["confidence"] >= min_confidence
                and b["level"] <= max_heading_levels
            ]

            # Sort by page then by position within page (implicit from detection order)
            filtered.sort(key=lambda b: (b["page"], b.get("_sort_y", 0)))

            # Strip internal sort keys
            flat_boundaries = []
            for b in filtered:
                entry = {
                    "title": b["title"],
                    "level": b["level"],
                    "page": b["page"],
                    "confidence": round(b["confidence"], 3),
                    "detection_method": b["detection_method"],
                }
                flat_boundaries.append(entry)

            # Build hierarchical tree
            sections = self._boundaries_to_sections(flat_boundaries, total_pages)

            return {
                "success": True,
                "structure": {
                    "sections": sections,
                    "flat_boundaries": flat_boundaries,
                },
                "detection_info": {
                    "strategies_used": strategies_used,
                    "bookmarks_found": bookmarks_found,
                    "body_font": body_font_info,
                    "heading_fonts": heading_font_info,
                    "total_pages": total_pages,
                },
                "detection_time": round(time.time() - start_time, 2),
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error("Structure detection failed: %s", error_msg)
            return {
                "success": False,
                "error": error_msg,
                "detection_time": round(time.time() - start_time, 2),
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_by_bookmarks(
        self, doc: fitz.Document
    ) -> List[Dict[str, Any]]:
        """Extract boundaries from PDF bookmarks / table of contents."""
        toc = doc.get_toc()
        boundaries: List[Dict[str, Any]] = []
        for level, title, page_num in toc:
            title_clean = title.strip()
            if not title_clean:
                continue
            boundaries.append(
                {
                    "title": title_clean,
                    "level": level,
                    "page": page_num,  # 1-based from fitz
                    "confidence": 0.95,
                    "detection_method": "bookmarks",
                    "_sort_y": 0,
                }
            )
        return boundaries

    def _detect_by_fonts(
        self,
        doc: fitz.Document,
        pages_to_process: List[int],
        max_levels: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[int, Dict[str, Any]]]:
        """
        Detect headings by font-size histogram analysis.

        Returns (boundaries, body_font_info, heading_font_map).
        """
        # Pass 1: build a histogram of font sizes weighted by character count
        size_char_count: Dict[float, int] = defaultdict(int)
        size_font_name: Dict[float, str] = {}

        for page_idx in pages_to_process:
            page = doc[page_idx]
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sz = round(span["size"], 1)
                        chars = len(span["text"])
                        if chars == 0:
                            continue
                        size_char_count[sz] += chars
                        # Keep the most-seen font name for each size
                        if sz not in size_font_name or size_char_count[sz] > 0:
                            size_font_name[sz] = span.get("font", "")

        if not size_char_count:
            return [], {}, {}

        # Body size = font size with highest total character count
        body_size = max(size_char_count, key=size_char_count.get)
        body_font_name = size_font_name.get(body_size, "")
        body_font_info = {"size": body_size, "name": body_font_name}

        # Heading candidates: sizes > body_size * 1.15
        threshold = body_size * 1.15
        heading_sizes = sorted(
            [sz for sz in size_char_count if sz > threshold], reverse=True
        )

        if not heading_sizes:
            return [], body_font_info, {}

        # Cluster heading sizes into at most max_levels levels.
        # Sizes within 1pt of each other collapse into one level.
        levels: List[List[float]] = []
        for sz in heading_sizes:
            placed = False
            for cluster in levels:
                if abs(sz - cluster[0]) <= 1.0:
                    cluster.append(sz)
                    placed = True
                    break
            if not placed:
                if len(levels) < max_levels:
                    levels.append([sz])
                # else: ignore smaller heading sizes beyond max_levels

        # Map each font size to its heading level (1 = largest)
        size_to_level: Dict[float, int] = {}
        heading_font_map: Dict[int, Dict[str, Any]] = {}
        for idx, cluster in enumerate(levels):
            level = idx + 1
            representative = max(cluster)
            heading_font_map[level] = {
                "size": representative,
                "name": size_font_name.get(representative, ""),
            }
            for sz in cluster:
                size_to_level[sz] = level

        # Pass 2: collect heading spans
        boundaries: List[Dict[str, Any]] = []
        for page_idx in pages_to_process:
            page = doc[page_idx]
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                for line in block.get("lines", []):
                    line_text_parts: List[str] = []
                    line_size: Optional[float] = None
                    line_is_bold = False
                    line_y = line.get("bbox", [0, 0, 0, 0])[1]

                    spans = line.get("spans", [])

                    # First pass: identify which spans are heading-sized
                    span_roles = []
                    for span in spans:
                        sz = round(span["size"], 1)
                        is_heading = sz in size_to_level
                        span_roles.append((span, sz, is_heading))

                    # Second pass: collect heading spans AND sandwiched
                    # non-heading spans (superscripts like ² in I²C)
                    for idx, (span, sz, is_heading) in enumerate(span_roles):
                        if is_heading:
                            line_text_parts.append(span["text"])
                            line_size = sz
                            if span.get("flags", 0) & 16:
                                line_is_bold = True
                        elif line_text_parts and idx + 1 < len(span_roles):
                            # Non-heading span between heading spans —
                            # likely a superscript/subscript (e.g. ² in I²C)
                            if span_roles[idx + 1][2]:  # next span is heading
                                line_text_parts.append(span["text"])

                    if not line_text_parts or line_size is None:
                        continue

                    heading_text = "".join(line_text_parts).strip()
                    if not heading_text:
                        continue

                    # Confidence scoring
                    confidence = 0.70
                    # Boost for bold
                    if line_is_bold:
                        confidence += 0.07
                    # Boost for short text (likely a heading, not a paragraph)
                    if len(heading_text) < 100:
                        confidence += 0.06
                    # Boost if text matches a common numbering pattern
                    for pat, _ in _NUMBERING_PATTERNS:
                        if re.match(pat, heading_text, re.IGNORECASE):
                            confidence += 0.07
                            break
                    confidence = min(confidence, 0.90)

                    level = size_to_level[line_size]
                    # page is 1-based for the boundary dict
                    boundaries.append(
                        {
                            "title": heading_text,
                            "level": level,
                            "page": page_idx + 1,
                            "confidence": confidence,
                            "detection_method": "fonts",
                            "_sort_y": line_y,
                        }
                    )

        # De-duplicate near-identical entries on the same page (same text, same page)
        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for b in boundaries:
            key = (b["page"], b["title"][:60])
            if key not in seen:
                seen.add(key)
                deduped.append(b)

        return deduped, body_font_info, heading_font_map

    def _detect_by_numbering(
        self, doc: fitz.Document, pages_to_process: List[int]
    ) -> List[Dict[str, Any]]:
        """Detect headings using built-in numbering/chapter patterns."""
        boundaries: List[Dict[str, Any]] = []

        for page_idx in pages_to_process:
            page = doc[page_idx]
            text = page.get_text()
            # Look at the first 200 chars or first line, whichever is longer
            first_line = text.split("\n", 1)[0].strip() if text else ""
            search_text = text[:200] if len(text) > 200 else text

            for pat, default_level in _NUMBERING_PATTERNS:
                match = re.search(pat, search_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    matched_text = match.group(0).strip()
                    # Grab the heading title up to the first newline
                    line_end = search_text.find("\n", match.start())
                    if line_end == -1:
                        line_end = len(search_text)
                    title = search_text[match.start():line_end].strip()
                    # Cap title length to avoid grabbing full sentences
                    if len(title) > 80:
                        title = title[:80].rstrip()
                        # Try to break at a word boundary
                        last_space = title.rfind(" ", 40)
                        if last_space > 0:
                            title = title[:last_space]

                    # Confidence varies: exact first-line match is higher
                    confidence = 0.70
                    if matched_text.lower() == first_line.lower()[:len(matched_text)]:
                        confidence = 0.80

                    boundaries.append(
                        {
                            "title": title,
                            "level": default_level,
                            "page": page_idx + 1,
                            "confidence": confidence,
                            "detection_method": "numbering",
                            "_sort_y": 0,
                        }
                    )
                    # Only take the first matching pattern per page
                    break

        return boundaries

    def _detect_by_pattern(
        self,
        doc: fitz.Document,
        pages_to_process: List[int],
        pattern: str,
    ) -> List[Dict[str, Any]]:
        """Apply a user-supplied regex to page text."""
        try:
            compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as exc:
            logger.warning("Invalid user heading_pattern regex: %s", exc)
            return []

        boundaries: List[Dict[str, Any]] = []

        for page_idx in pages_to_process:
            page = doc[page_idx]
            text = page.get_text()

            for match in compiled.finditer(text):
                title = match.group(0).strip()
                if not title:
                    continue
                if len(title) > 120:
                    title = title[:120].rstrip()

                boundaries.append(
                    {
                        "title": title,
                        "level": 1,  # User patterns default to level 1
                        "page": page_idx + 1,
                        "confidence": 0.85,
                        "detection_method": "user_regex",
                        "_sort_y": match.start(),
                    }
                )

        return boundaries

    # ------------------------------------------------------------------
    # Merge and tree-building
    # ------------------------------------------------------------------

    def _merge_detections(
        self, *detection_lists: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge multiple detection lists, de-duplicating boundaries that
        refer to the same heading (same page +/-1, similar title).
        When merging, take the max confidence and combine method names.
        """
        if not detection_lists:
            return []

        # Flatten
        all_items: List[Dict[str, Any]] = []
        for dl in detection_lists:
            all_items.extend(dl)

        if not all_items:
            return []

        # Sort by page then sort_y
        all_items.sort(key=lambda b: (b["page"], b.get("_sort_y", 0)))

        merged: List[Dict[str, Any]] = []

        for item in all_items:
            matched = False
            for existing in merged:
                # Same page (+/-1) and similar title
                if abs(existing["page"] - item["page"]) <= 1:
                    if self._titles_similar(existing["title"], item["title"]):
                        # Merge: boost confidence, combine methods
                        existing["confidence"] = min(
                            0.99,
                            max(existing["confidence"], item["confidence"]) + 0.05,
                        )
                        methods = set(existing["detection_method"].split("+"))
                        methods.add(item["detection_method"])
                        existing["detection_method"] = "+".join(sorted(methods))
                        # Keep the smaller (more prominent) level
                        existing["level"] = min(existing["level"], item["level"])
                        matched = True
                        break
            if not matched:
                merged.append(dict(item))

        return merged

    @staticmethod
    def _titles_similar(a: str, b: str) -> bool:
        """Check whether two heading titles are similar enough to merge."""
        a_norm = re.sub(r"\s+", " ", a.strip().lower())
        b_norm = re.sub(r"\s+", " ", b.strip().lower())
        if a_norm == b_norm:
            return True
        # One contains the other (common with partial extractions)
        if a_norm in b_norm or b_norm in a_norm:
            return True
        # Compare first 40 chars (handles trailing differences)
        if len(a_norm) > 10 and len(b_norm) > 10:
            return a_norm[:40] == b_norm[:40]
        return False

    def _boundaries_to_sections(
        self,
        boundaries: List[Dict[str, Any]],
        total_pages: int,
    ) -> List[Dict[str, Any]]:
        """
        Convert a flat sorted boundary list into a hierarchical section tree.
        Each section gets page_start, page_end, and nested subsections.
        """
        if not boundaries:
            return []

        # Assign page_end to each boundary: runs until the next boundary's page - 1
        enriched: List[Dict[str, Any]] = []
        for i, b in enumerate(boundaries):
            page_start = b["page"]
            if i + 1 < len(boundaries):
                page_end = boundaries[i + 1]["page"] - 1
                # Ensure page_end >= page_start
                page_end = max(page_end, page_start)
            else:
                page_end = total_pages
            enriched.append(
                {
                    "title": b["title"],
                    "level": b["level"],
                    "page_start": page_start,
                    "page_end": page_end,
                    "confidence": b["confidence"],
                    "detection_method": b["detection_method"],
                    "subsections": [],
                }
            )

        # Build tree using a stack-based approach
        root_sections: List[Dict[str, Any]] = []
        stack: List[Dict[str, Any]] = []  # stack of currently open sections

        for section in enriched:
            # Pop sections from the stack that are at the same level or deeper
            while stack and stack[-1]["level"] >= section["level"]:
                stack.pop()

            if stack:
                # This section is a child of the top of the stack
                stack[-1]["subsections"].append(section)
            else:
                # Top-level section
                root_sections.append(section)

            stack.append(section)

        # Adjust page_end for parent sections to encompass children
        self._fix_parent_page_ends(root_sections, total_pages)

        return root_sections

    def _fix_parent_page_ends(
        self, sections: List[Dict[str, Any]], total_pages: int
    ) -> None:
        """Recursively ensure parent page_end covers all children."""
        for section in sections:
            if section["subsections"]:
                self._fix_parent_page_ends(section["subsections"], total_pages)
                child_max = max(
                    child["page_end"] for child in section["subsections"]
                )
                section["page_end"] = max(section["page_end"], child_max)

    # ------------------------------------------------------------------
    # Filesystem-safe name helper (for downstream splitting tools)
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_dirname(title: str) -> str:
        """
        Convert a heading title into a filesystem-safe directory name.

        Replaces special characters with underscores, strips leading/trailing
        underscores and whitespace, and truncates to 50 characters at a word
        boundary for clean directory listings.
        """
        # Replace anything that isn't alphanumeric, space, hyphen, or underscore
        safe = re.sub(r"[^\w\s-]", "_", title)
        # Collapse runs of whitespace / underscores
        safe = re.sub(r"[\s_]+", "_", safe)
        # Strip leading/trailing underscores and whitespace
        safe = safe.strip("_ ")
        # Truncate at word boundary for clean names
        if len(safe) > 50:
            truncated = safe[:50]
            last_sep = truncated.rfind("_", 20)
            if last_sep > 0:
                truncated = truncated[:last_sep]
            safe = truncated.rstrip("_")
        return safe or "untitled"

    # ------------------------------------------------------------------
    # Tool 2: split_pdf_by_structure
    # ------------------------------------------------------------------

    @mcp_tool(
        name="split_pdf_by_structure",
        description=(
            "Detect document structure then split the PDF into per-chapter/section "
            "directories. Each section gets its own PDF and optionally markdown + images. "
            "Combines detect_structure + split + pdf_to_markdown into one operation."
        ),
    )
    async def split_pdf_by_structure(
        self,
        pdf_path: str,
        output_directory: str,
        split_level: int = 1,
        include_markdown: bool = True,
        include_images: bool = True,
        include_vectors: bool = True,
        strategies: str = "auto",
        heading_pattern: Optional[str] = None,
        min_confidence: float = 0.5,
        output_format: str = "markdown",
    ) -> Dict[str, Any]:
        """
        Detect structure and split a PDF into per-section directories.

        Args:
            pdf_path: Path to PDF file or HTTPS URL.
            output_directory: Root directory for section output folders.
            split_level: Heading level to split on (1=chapters, 2=sections, etc.).
            include_markdown: Convert each split PDF to markdown.
            include_images: Extract raster images during markdown conversion.
            include_vectors: Extract vector graphics during markdown conversion.
            strategies: Detection strategy for structure detection.
            heading_pattern: Optional user-supplied regex for headings.
            min_confidence: Drop boundaries below this confidence (0-1).
            output_format: "markdown", "pdf", or "both".

        Returns:
            Dict with per-section results, paths, extraction counts, and
            the detected structure.
        """
        start_time = time.time()

        try:
            # Validate inputs
            path = await validate_pdf_path(pdf_path)
            output_dir = Path(validate_output_path(output_directory))
            output_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Detect structure
            structure_result = await self.detect_structure(
                pdf_path=pdf_path,
                strategies=strategies,
                heading_pattern=heading_pattern,
                min_confidence=min_confidence,
            )

            if not structure_result.get("success"):
                return {
                    "success": False,
                    "error": structure_result.get("error", "Structure detection failed"),
                    "split_time": round(time.time() - start_time, 2),
                }

            flat_boundaries = structure_result["structure"]["flat_boundaries"]

            # Step 2: Filter boundaries at the requested split_level
            split_boundaries = [
                b for b in flat_boundaries
                if b["level"] <= split_level and b["confidence"] >= min_confidence
            ]

            if not split_boundaries:
                return {
                    "success": False,
                    "error": (
                        f"No boundaries found at level <= {split_level} with "
                        f"confidence >= {min_confidence}. Try lowering min_confidence "
                        f"or increasing split_level."
                    ),
                    "detected_structure": structure_result["structure"],
                    "split_time": round(time.time() - start_time, 2),
                }

            # Get total page count
            source_doc = fitz.open(str(path))
            total_pages = len(source_doc)

            # Step 3: Compute page ranges from adjacent boundaries
            sections_results = []
            for i, boundary in enumerate(split_boundaries):
                page_start = boundary["page"]  # 1-based
                if i + 1 < len(split_boundaries):
                    page_end = split_boundaries[i + 1]["page"] - 1
                    page_end = max(page_end, page_start)
                else:
                    page_end = total_pages

                title = boundary["title"]
                clean_title = self._sanitize_dirname(title)
                section_dirname = f"{i:02d}_{clean_title}"
                section_dir = output_dir / section_dirname
                section_dir.mkdir(parents=True, exist_ok=True)

                # Step 4a: Create split PDF
                section_pdf_path = section_dir / f"{clean_title}.pdf"
                new_doc = fitz.open()
                new_doc.insert_pdf(
                    source_doc,
                    from_page=page_start - 1,  # convert to 0-based
                    to_page=page_end - 1,       # convert to 0-based
                )
                new_doc.save(str(section_pdf_path))
                new_doc.close()

                # Step 4b: Optionally convert to markdown
                md_path = None
                images_extracted = 0
                vectors_extracted = 0

                if include_markdown and output_format in ("markdown", "both"):
                    try:
                        img_mixin = ImageProcessingMixin()
                        md_result = await img_mixin.pdf_to_markdown(
                            pdf_path=str(section_pdf_path),
                            output_directory=str(section_dir),
                            output_filename=f"{clean_title}.md",
                            include_images=include_images,
                            include_vectors=include_vectors,
                        )
                        if md_result.get("success"):
                            md_path = md_result.get("output_file")
                            summary = md_result.get("conversion_summary", {})
                            images_extracted = summary.get("images_extracted", 0)
                            vectors_extracted = summary.get("vectors_extracted", 0)
                    except Exception as md_exc:
                        logger.warning(
                            "Markdown conversion failed for section '%s': %s",
                            title, md_exc,
                        )

                # If output_format is "markdown" only, remove the split PDF
                if output_format == "markdown" and md_path:
                    try:
                        section_pdf_path.unlink()
                        section_pdf_path = None
                    except OSError:
                        pass

                sections_results.append({
                    "title": title,
                    "page_start": page_start,
                    "page_end": page_end,
                    "directory": str(section_dir),
                    "pdf_path": str(section_pdf_path) if section_pdf_path else None,
                    "markdown_path": str(md_path) if md_path else None,
                    "images_extracted": images_extracted,
                    "vectors_extracted": vectors_extracted,
                })

            source_doc.close()

            return {
                "success": True,
                "sections_created": len(sections_results),
                "output_directory": str(output_dir),
                "sections": sections_results,
                "detected_structure": structure_result["structure"],
                "split_time": round(time.time() - start_time, 2),
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error("split_pdf_by_structure failed: %s", error_msg)
            return {
                "success": False,
                "error": error_msg,
                "split_time": round(time.time() - start_time, 2),
            }

    # ------------------------------------------------------------------
    # Tool 3: batch_extract
    # ------------------------------------------------------------------

    @mcp_tool(
        name="batch_extract",
        description=(
            "Extract multiple page ranges from a single PDF, each producing its own "
            "markdown + images + vectors in a separate output directory. Replaces "
            "24+ individual tool calls with a single operation."
        ),
    )
    async def batch_extract(
        self,
        pdf_path: str,
        sections: str,
        include_images: bool = True,
        include_vectors: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract multiple page ranges from a single PDF into separate directories.

        Args:
            pdf_path: Path to PDF file or HTTPS URL.
            sections: JSON string — a list of objects, each with:
                - "pages": page range string, e.g. "11-80"
                - "output_dir": output directory path for this section
                - "name": human-readable name for the section
            include_images: Extract raster images during markdown conversion.
            include_vectors: Extract vector graphics during markdown conversion.

        Returns:
            Dict with per-section extraction results and timing.
        """
        start_time = time.time()

        try:
            # Parse sections JSON
            try:
                section_list = json.loads(sections)
            except (json.JSONDecodeError, TypeError) as parse_err:
                return {
                    "success": False,
                    "error": f"Invalid sections JSON: {parse_err}",
                    "batch_time": round(time.time() - start_time, 2),
                }

            if not isinstance(section_list, list) or not section_list:
                return {
                    "success": False,
                    "error": "sections must be a non-empty JSON array",
                    "batch_time": round(time.time() - start_time, 2),
                }

            # Validate the source PDF once
            path = await validate_pdf_path(pdf_path)
            source_doc = fitz.open(str(path))
            total_pages = len(source_doc)

            results = []

            for idx, section in enumerate(section_list):
                section_name = section.get("name", f"section_{idx:02d}")
                pages_str = section.get("pages", "")
                section_output_dir = section.get("output_dir", "")

                if not pages_str or not section_output_dir:
                    results.append({
                        "name": section_name,
                        "pages": pages_str,
                        "success": False,
                        "error": "Missing 'pages' or 'output_dir' field",
                    })
                    continue

                try:
                    # Parse page range (e.g. "11-80")
                    out_dir = Path(validate_output_path(section_output_dir))
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # Parse "start-end" format
                    page_start, page_end = self._parse_page_range(pages_str, total_pages)

                    # Create split PDF
                    clean_name = self._sanitize_dirname(section_name)
                    section_pdf_path = out_dir / f"{clean_name}.pdf"
                    new_doc = fitz.open()
                    new_doc.insert_pdf(
                        source_doc,
                        from_page=page_start - 1,  # convert to 0-based
                        to_page=page_end - 1,       # convert to 0-based
                    )
                    new_doc.save(str(section_pdf_path))
                    new_doc.close()

                    # Convert to markdown
                    md_result = None
                    try:
                        img_mixin = ImageProcessingMixin()
                        md_result = await img_mixin.pdf_to_markdown(
                            pdf_path=str(section_pdf_path),
                            output_directory=str(out_dir),
                            output_filename=f"{clean_name}.md",
                            include_images=include_images,
                            include_vectors=include_vectors,
                        )
                    except Exception as md_exc:
                        logger.warning(
                            "Markdown conversion failed for '%s': %s",
                            section_name, md_exc,
                        )
                        md_result = {"success": False, "error": str(md_exc)}

                    results.append({
                        "name": section_name,
                        "pages": pages_str,
                        "output_directory": str(out_dir),
                        "pdf_path": str(section_pdf_path),
                        "markdown_result": md_result,
                    })

                except Exception as sec_exc:
                    error_msg = sanitize_error_message(str(sec_exc))
                    logger.warning(
                        "batch_extract section '%s' failed: %s",
                        section_name, error_msg,
                    )
                    results.append({
                        "name": section_name,
                        "pages": pages_str,
                        "success": False,
                        "error": error_msg,
                    })

            source_doc.close()

            return {
                "success": True,
                "sections_processed": len(results),
                "sections": results,
                "batch_time": round(time.time() - start_time, 2),
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error("batch_extract failed: %s", error_msg)
            return {
                "success": False,
                "error": error_msg,
                "batch_time": round(time.time() - start_time, 2),
            }

    # ------------------------------------------------------------------
    # Page range parsing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_page_range(pages_str: str, total_pages: int) -> Tuple[int, int]:
        """
        Parse a page range string like "11-80" into (start, end) 1-based ints.

        Supports formats:
            "11-80"   -> (11, 80)
            "5"       -> (5, 5)
            "11-end"  -> (11, total_pages)

        Raises ValueError on invalid input.
        """
        pages_str = pages_str.strip()

        if "-" in pages_str:
            parts = pages_str.split("-", 1)
            start_str = parts[0].strip()
            end_str = parts[1].strip()

            page_start = int(start_str)
            if end_str.lower() == "end":
                page_end = total_pages
            else:
                page_end = int(end_str)
        else:
            page_start = int(pages_str)
            page_end = page_start

        # Clamp to valid range
        page_start = max(1, min(page_start, total_pages))
        page_end = max(page_start, min(page_end, total_pages))

        return page_start, page_end
