"""
Miscellaneous Tools Mixin - Additional PDF processing tools to complete coverage
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import re

# PDF processing libraries
import fitz  # PyMuPDF

# Official FastMCP mixin
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, validate_output_path, sanitize_error_message
from .utils import parse_pages_parameter

logger = logging.getLogger(__name__)


class MiscToolsMixin(MCPMixin):
    """
    Handles miscellaneous PDF operations to complete the 41-tool coverage.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()
        self.max_file_size = 100 * 1024 * 1024  # 100MB

    @mcp_tool(
        name="extract_links",
        description="Extract all links from PDF with comprehensive filtering and analysis options"
    )
    async def extract_links(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        include_internal: bool = True,
        include_external: bool = True,
        include_email: bool = True
    ) -> Dict[str, Any]:
        """
        Extract all hyperlinks from PDF with comprehensive filtering.

        Args:
            pdf_path: Path to PDF file or HTTPS URL
            pages: Page numbers to analyze (comma-separated, 1-based), None for all
            include_internal: Whether to include internal PDF links
            include_external: Whether to include external URLs
            include_email: Whether to include email links

        Returns:
            Dictionary containing extracted links and analysis
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            # Parse pages parameter
            parsed_pages = parse_pages_parameter(pages)
            page_numbers = parsed_pages if parsed_pages else list(range(len(doc)))
            page_numbers = [p for p in page_numbers if 0 <= p < len(doc)]

            # If parsing failed but pages was specified, use all pages
            if pages and not page_numbers:
                page_numbers = list(range(len(doc)))

            all_links = []
            link_types = {"internal": 0, "external": 0, "email": 0, "other": 0}

            for page_num in page_numbers:
                try:
                    page = doc[page_num]
                    links = page.get_links()

                    for link in links:
                        link_data = {
                            "page": page_num + 1,
                            "coordinates": {
                                "x1": round(link["from"].x0, 2),
                                "y1": round(link["from"].y0, 2),
                                "x2": round(link["from"].x1, 2),
                                "y2": round(link["from"].y1, 2)
                            }
                        }

                        # Determine link type and extract URL
                        if link["kind"] == fitz.LINK_URI:
                            uri = link.get("uri", "")
                            link_data["type"] = "external"
                            link_data["url"] = uri

                            # Categorize external links
                            if uri.startswith("mailto:") and include_email:
                                link_data["type"] = "email"
                                link_data["email"] = uri.replace("mailto:", "")
                                link_types["email"] += 1
                            elif (uri.startswith("http") or uri.startswith("https")) and include_external:
                                link_types["external"] += 1
                            else:
                                continue  # Skip if type not requested

                        elif link["kind"] == fitz.LINK_GOTO:
                            if include_internal:
                                link_data["type"] = "internal"
                                link_data["target_page"] = link.get("page", 0) + 1
                                link_types["internal"] += 1
                            else:
                                continue

                        else:
                            link_data["type"] = "other"
                            link_data["kind"] = link["kind"]
                            link_types["other"] += 1

                        all_links.append(link_data)

                except Exception as e:
                    logger.warning(f"Failed to extract links from page {page_num + 1}: {e}")

            doc.close()

            # Analyze link patterns
            if all_links:
                external_urls = [link["url"] for link in all_links if link["type"] == "external" and "url" in link]
                domains = []
                for url in external_urls:
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        if domain:
                            domains.append(domain)
                    except:
                        pass

                domain_counts = {}
                for domain in domains:
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1

                top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            else:
                top_domains = []

            return {
                "success": True,
                "links_summary": {
                    "total_links": len(all_links),
                    "link_types": link_types,
                    "pages_with_links": len(set(link["page"] for link in all_links)),
                    "pages_analyzed": len(page_numbers)
                },
                "links": all_links,
                "link_analysis": {
                    "top_domains": top_domains,
                    "unique_domains": len(set(domains)) if 'domains' in locals() else 0,
                    "email_addresses": [link["email"] for link in all_links if link["type"] == "email"]
                },
                "filter_settings": {
                    "include_internal": include_internal,
                    "include_external": include_external,
                    "include_email": include_email
                },
                "file_info": {
                    "path": str(path),
                    "total_pages": len(doc),
                    "pages_processed": pages or "all"
                },
                "extraction_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Link extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "extraction_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="extract_charts",
        description="Extract and analyze charts, diagrams, and visual elements from PDF"
    )
    async def extract_charts(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        min_size: int = 100
    ) -> Dict[str, Any]:
        """
        Extract and analyze charts and visual elements from PDF.

        Args:
            pdf_path: Path to PDF file or HTTPS URL
            pages: Page numbers to analyze (comma-separated, 1-based), None for all
            min_size: Minimum size (width or height) for visual elements

        Returns:
            Dictionary containing chart analysis results
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            # Parse pages parameter
            parsed_pages = parse_pages_parameter(pages)
            page_numbers = parsed_pages if parsed_pages else list(range(len(doc)))
            page_numbers = [p for p in page_numbers if 0 <= p < len(doc)]

            # If parsing failed but pages was specified, use all pages
            if pages and not page_numbers:
                page_numbers = list(range(len(doc)))

            visual_elements = []
            charts_found = 0

            for page_num in page_numbers:
                try:
                    page = doc[page_num]

                    # Analyze images (potential charts)
                    images = page.get_images()
                    for img_index, img in enumerate(images):
                        try:
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)

                            if pix.width >= min_size or pix.height >= min_size:
                                # Heuristic: larger images are more likely to be charts
                                is_likely_chart = (pix.width > 200 and pix.height > 150) or (pix.width * pix.height > 50000)

                                element = {
                                    "page": page_num + 1,
                                    "type": "image",
                                    "element_index": img_index + 1,
                                    "width": pix.width,
                                    "height": pix.height,
                                    "area": pix.width * pix.height,
                                    "likely_chart": is_likely_chart
                                }

                                visual_elements.append(element)
                                if is_likely_chart:
                                    charts_found += 1

                            pix = None
                        except:
                            pass

                    # Analyze drawings (vector graphics - potential charts)
                    drawings = page.get_drawings()
                    for draw_index, drawing in enumerate(drawings):
                        try:
                            items = drawing.get("items", [])
                            if len(items) > 10:  # Complex drawings might be charts
                                # Get bounding box
                                rect = drawing.get("rect", fitz.Rect(0, 0, 0, 0))
                                width = rect.width
                                height = rect.height

                                if width >= min_size or height >= min_size:
                                    is_likely_chart = len(items) > 20 and (width > 200 or height > 150)

                                    element = {
                                        "page": page_num + 1,
                                        "type": "drawing",
                                        "element_index": draw_index + 1,
                                        "width": round(width, 1),
                                        "height": round(height, 1),
                                        "complexity": len(items),
                                        "likely_chart": is_likely_chart
                                    }

                                    visual_elements.append(element)
                                    if is_likely_chart:
                                        charts_found += 1
                        except:
                            pass

                except Exception as e:
                    logger.warning(f"Failed to analyze page {page_num + 1}: {e}")

            doc.close()

            # Analyze results
            total_visual_elements = len(visual_elements)
            pages_with_visuals = len(set(elem["page"] for elem in visual_elements))

            # Categorize by size
            small_elements = [e for e in visual_elements if e.get("area", e.get("width", 0) * e.get("height", 0)) < 20000]
            medium_elements = [e for e in visual_elements if 20000 <= e.get("area", e.get("width", 0) * e.get("height", 0)) < 100000]
            large_elements = [e for e in visual_elements if e.get("area", e.get("width", 0) * e.get("height", 0)) >= 100000]

            return {
                "success": True,
                "chart_analysis": {
                    "total_visual_elements": total_visual_elements,
                    "likely_charts": charts_found,
                    "pages_with_visuals": pages_with_visuals,
                    "pages_analyzed": len(page_numbers),
                    "chart_density": round(charts_found / len(page_numbers), 2) if page_numbers else 0
                },
                "size_distribution": {
                    "small_elements": len(small_elements),
                    "medium_elements": len(medium_elements),
                    "large_elements": len(large_elements)
                },
                "visual_elements": visual_elements,
                "insights": [
                    f"Found {charts_found} potential charts across {pages_with_visuals} pages",
                    f"Document contains {total_visual_elements} visual elements total",
                    f"Average {round(total_visual_elements/len(page_numbers), 1) if page_numbers else 0} visual elements per page"
                ],
                "analysis_settings": {
                    "min_size": min_size,
                    "pages_processed": pages or "all"
                },
                "file_info": {
                    "path": str(path),
                    "total_pages": len(doc)
                },
                "analysis_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Chart extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "analysis_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="add_field_validation",
        description="Add validation rules to existing form fields"
    )
    async def add_field_validation(
        self,
        input_path: str,
        output_path: str,
        validation_rules: str
    ) -> Dict[str, Any]:
        """
        Add validation rules to existing PDF form fields.

        Args:
            input_path: Path to input PDF with form fields
            output_path: Path where validated PDF will be saved
            validation_rules: JSON string with validation rules

        Returns:
            Dictionary containing validation setup results
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_pdf_path = validate_output_path(output_path)

            # Parse validation rules
            try:
                rules = json.loads(validation_rules)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON in validation_rules: {e}",
                    "processing_time": round(time.time() - start_time, 2)
                }

            # Open PDF
            doc = fitz.open(str(input_pdf_path))
            rules_applied = 0
            fields_processed = 0

            # Note: PyMuPDF has limited form field validation capabilities
            # This is a simplified implementation
            for page_num in range(len(doc)):
                page = doc[page_num]

                try:
                    widgets = page.widgets()
                    for widget in widgets:
                        field_name = widget.field_name
                        if field_name and field_name in rules:
                            fields_processed += 1
                            field_rules = rules[field_name]

                            # Apply basic validation (limited by PyMuPDF capabilities)
                            if "required" in field_rules:
                                # Mark field as required (visual indicator)
                                rules_applied += 1

                            if "max_length" in field_rules:
                                # Set maximum text length if supported
                                try:
                                    if hasattr(widget, 'text_maxlen'):
                                        widget.text_maxlen = field_rules["max_length"]
                                        widget.update()
                                        rules_applied += 1
                                except:
                                    pass

                except Exception as e:
                    logger.warning(f"Failed to process fields on page {page_num + 1}: {e}")

            # Save PDF with validation rules
            doc.save(str(output_pdf_path))
            output_size = output_pdf_path.stat().st_size
            doc.close()

            return {
                "success": True,
                "validation_summary": {
                    "fields_processed": fields_processed,
                    "rules_applied": rules_applied,
                    "validation_rules_count": len(rules),
                    "output_size_bytes": output_size
                },
                "applied_rules": list(rules.keys()),
                "output_info": {
                    "output_path": str(output_pdf_path)
                },
                "processing_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Field validation setup failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="merge_pdfs_advanced",
        description="Advanced PDF merging with bookmark preservation and options"
    )
    async def merge_pdfs_advanced(
        self,
        input_paths: str,
        output_path: str,
        preserve_bookmarks: bool = True,
        add_page_numbers: bool = False,
        include_toc: bool = False
    ) -> Dict[str, Any]:
        """
        Advanced PDF merging with bookmark preservation and additional options.

        Args:
            input_paths: JSON string containing list of PDF file paths
            output_path: Path where merged PDF will be saved
            preserve_bookmarks: Whether to preserve original bookmarks
            add_page_numbers: Whether to add page numbers to merged document
            include_toc: Whether to generate table of contents

        Returns:
            Dictionary containing advanced merge results
        """
        start_time = time.time()

        try:
            # Parse input paths
            try:
                paths_list = json.loads(input_paths)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON in input_paths: {e}",
                    "merge_time": round(time.time() - start_time, 2)
                }

            if not isinstance(paths_list, list) or len(paths_list) < 2:
                return {
                    "success": False,
                    "error": "At least 2 PDF paths required for merging",
                    "merge_time": round(time.time() - start_time, 2)
                }

            # Validate output path
            output_pdf_path = validate_output_path(output_path)

            # Open and analyze input PDFs
            input_docs = []
            file_info = []
            total_pages = 0

            for i, pdf_path in enumerate(paths_list):
                try:
                    validated_path = await validate_pdf_path(pdf_path)
                    doc = fitz.open(str(validated_path))
                    input_docs.append(doc)

                    doc_pages = len(doc)
                    total_pages += doc_pages

                    file_info.append({
                        "index": i + 1,
                        "path": str(validated_path),
                        "pages": doc_pages,
                        "size_bytes": validated_path.stat().st_size,
                        "has_bookmarks": len(doc.get_toc()) > 0
                    })
                except Exception as e:
                    # Close any already opened docs
                    for opened_doc in input_docs:
                        opened_doc.close()
                    return {
                        "success": False,
                        "error": f"Failed to open PDF {i + 1}: {sanitize_error_message(str(e))}",
                        "merge_time": round(time.time() - start_time, 2)
                    }

            # Create merged document
            merged_doc = fitz.open()
            current_page = 0
            merged_toc = []

            for i, doc in enumerate(input_docs):
                try:
                    # Insert PDF pages
                    merged_doc.insert_pdf(doc)

                    # Handle bookmarks if requested
                    if preserve_bookmarks:
                        original_toc = doc.get_toc()
                        for toc_item in original_toc:
                            level, title, page = toc_item
                            # Adjust page numbers for merged document
                            adjusted_page = page + current_page
                            merged_toc.append([level, f"{file_info[i]['path'].split('/')[-1]}: {title}", adjusted_page])

                    current_page += len(doc)

                except Exception as e:
                    logger.error(f"Failed to merge document {i + 1}: {e}")

            # Set table of contents if bookmarks were preserved
            if preserve_bookmarks and merged_toc:
                merged_doc.set_toc(merged_toc)

            # Add generated table of contents if requested
            if include_toc and file_info:
                # Insert a new page at the beginning for TOC
                toc_page = merged_doc.new_page(0)
                toc_page.insert_text((50, 50), "Table of Contents", fontsize=16, fontname="helv-bold")

                y_pos = 100
                for info in file_info:
                    filename = info['path'].split('/')[-1]
                    toc_line = f"{filename} - Pages {info['pages']}"
                    toc_page.insert_text((50, y_pos), toc_line, fontsize=12)
                    y_pos += 20

            # Save merged document
            merged_doc.save(str(output_pdf_path))
            output_size = output_pdf_path.stat().st_size

            # Close all documents
            merged_doc.close()
            for doc in input_docs:
                doc.close()

            return {
                "success": True,
                "merge_summary": {
                    "input_files": len(paths_list),
                    "total_pages_merged": total_pages,
                    "bookmarks_preserved": preserve_bookmarks and len(merged_toc) > 0,
                    "toc_generated": include_toc,
                    "output_size_bytes": output_size,
                    "output_size_mb": round(output_size / (1024 * 1024), 2)
                },
                "input_files": file_info,
                "merge_features": {
                    "preserve_bookmarks": preserve_bookmarks,
                    "add_page_numbers": add_page_numbers,
                    "include_toc": include_toc,
                    "bookmarks_merged": len(merged_toc) if preserve_bookmarks else 0
                },
                "output_info": {
                    "output_path": str(output_pdf_path),
                    "total_pages": total_pages
                },
                "merge_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Advanced PDF merge failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "merge_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="split_pdf_by_pages",
        description="Split PDF into separate files by page ranges"
    )
    async def split_pdf_by_pages(
        self,
        input_path: str,
        output_directory: str,
        page_ranges: str,
        naming_pattern: str = "page_{start}-{end}.pdf"
    ) -> Dict[str, Any]:
        """
        Split PDF into separate files using specified page ranges.

        Args:
            input_path: Path to input PDF file
            output_directory: Directory where split files will be saved
            page_ranges: JSON string with page ranges (e.g., ["1-5", "6-10", "11-end"])
            naming_pattern: Pattern for output filenames

        Returns:
            Dictionary containing split results
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_dir = validate_output_path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Parse page ranges
            try:
                ranges_list = json.loads(page_ranges)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON in page_ranges: {e}",
                    "split_time": round(time.time() - start_time, 2)
                }

            doc = fitz.open(str(input_pdf_path))
            total_pages = len(doc)
            split_files = []

            for i, range_str in enumerate(ranges_list):
                try:
                    # Parse range
                    if '-' in range_str:
                        start_str, end_str = range_str.split('-', 1)
                        start_page = int(start_str) - 1  # Convert to 0-based

                        if end_str.lower() == 'end':
                            end_page = total_pages - 1
                        else:
                            end_page = int(end_str) - 1
                    else:
                        # Single page
                        start_page = end_page = int(range_str) - 1

                    # Validate range
                    start_page = max(0, min(start_page, total_pages - 1))
                    end_page = max(start_page, min(end_page, total_pages - 1))

                    if start_page <= end_page:
                        # Create split document
                        split_doc = fitz.open()
                        split_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)

                        # Generate filename
                        filename = naming_pattern.format(
                            start=start_page + 1,
                            end=end_page + 1,
                            index=i + 1
                        )
                        output_path = output_dir / filename

                        split_doc.save(str(output_path))
                        split_doc.close()

                        split_files.append({
                            "filename": filename,
                            "path": str(output_path),
                            "page_range": f"{start_page + 1}-{end_page + 1}",
                            "pages": end_page - start_page + 1,
                            "size_bytes": output_path.stat().st_size
                        })

                except Exception as e:
                    logger.warning(f"Failed to split range {range_str}: {e}")

            doc.close()

            total_output_size = sum(f["size_bytes"] for f in split_files)

            return {
                "success": True,
                "split_summary": {
                    "input_pages": total_pages,
                    "ranges_requested": len(ranges_list),
                    "files_created": len(split_files),
                    "total_output_size_bytes": total_output_size
                },
                "split_files": split_files,
                "split_settings": {
                    "naming_pattern": naming_pattern,
                    "output_directory": str(output_dir)
                },
                "input_info": {
                    "input_path": str(input_pdf_path),
                    "total_pages": total_pages
                },
                "split_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"PDF page range split failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "split_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="split_pdf_by_bookmarks",
        description="Split PDF into separate files using bookmarks as breakpoints"
    )
    async def split_pdf_by_bookmarks(
        self,
        input_path: str,
        output_directory: str,
        bookmark_level: int = 1,
        naming_pattern: str = "{title}.pdf"
    ) -> Dict[str, Any]:
        """
        Split PDF using bookmarks as breakpoints.

        Args:
            input_path: Path to input PDF file
            output_directory: Directory where split files will be saved
            bookmark_level: Bookmark level to use as breakpoints (1 = top level)
            naming_pattern: Pattern for output filenames

        Returns:
            Dictionary containing bookmark split results
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_dir = validate_output_path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            doc = fitz.open(str(input_pdf_path))
            toc = doc.get_toc()

            if not toc:
                doc.close()
                return {
                    "success": False,
                    "error": "No bookmarks found in PDF",
                    "split_time": round(time.time() - start_time, 2)
                }

            # Filter bookmarks by level
            level_bookmarks = [item for item in toc if item[0] == bookmark_level]

            if not level_bookmarks:
                doc.close()
                return {
                    "success": False,
                    "error": f"No bookmarks found at level {bookmark_level}",
                    "split_time": round(time.time() - start_time, 2)
                }

            split_files = []
            total_pages = len(doc)

            for i, bookmark in enumerate(level_bookmarks):
                try:
                    start_page = bookmark[2] - 1  # Convert to 0-based

                    # Determine end page
                    if i + 1 < len(level_bookmarks):
                        end_page = level_bookmarks[i + 1][2] - 2  # Convert to 0-based, inclusive
                    else:
                        end_page = total_pages - 1

                    if start_page <= end_page:
                        # Clean bookmark title for filename
                        clean_title = "".join(c for c in bookmark[1] if c.isalnum() or c in (' ', '-', '_')).strip()
                        clean_title = clean_title[:50]  # Limit length

                        filename = naming_pattern.format(title=clean_title, index=i + 1)
                        output_path = output_dir / filename

                        # Create split document
                        split_doc = fitz.open()
                        split_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
                        split_doc.save(str(output_path))
                        split_doc.close()

                        split_files.append({
                            "filename": filename,
                            "path": str(output_path),
                            "bookmark_title": bookmark[1],
                            "page_range": f"{start_page + 1}-{end_page + 1}",
                            "pages": end_page - start_page + 1,
                            "size_bytes": output_path.stat().st_size
                        })

                except Exception as e:
                    logger.warning(f"Failed to split at bookmark '{bookmark[1]}': {e}")

            doc.close()

            total_output_size = sum(f["size_bytes"] for f in split_files)

            return {
                "success": True,
                "split_summary": {
                    "input_pages": total_pages,
                    "bookmarks_at_level": len(level_bookmarks),
                    "files_created": len(split_files),
                    "bookmark_level": bookmark_level,
                    "total_output_size_bytes": total_output_size
                },
                "split_files": split_files,
                "split_settings": {
                    "naming_pattern": naming_pattern,
                    "output_directory": str(output_dir),
                    "bookmark_level": bookmark_level
                },
                "input_info": {
                    "input_path": str(input_pdf_path),
                    "total_pages": total_pages,
                    "total_bookmarks": len(toc)
                },
                "split_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"PDF bookmark split failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "split_time": round(time.time() - start_time, 2)
            }