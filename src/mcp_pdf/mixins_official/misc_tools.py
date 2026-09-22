"""
Miscellaneous Tools Mixin - Additional PDF processing tools to complete coverage
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import time
import json
from typing import Dict, Any, Optional
import logging

# PDF processing libraries
import pymupdf

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

    @mcp_tool(
        name="extract_links",
        description=(
            "List the hyperlinks in a PDF with their page, rectangle, target "
            "and type, plus a domain rollup and the email addresses found. "
            "Read-only: nothing is written and the file is not modified.\n"
            "\n"
            "It reads real PDF LINK ANNOTATIONS, the clickable regions. A "
            "URL that merely appears as text with no link attached to it is "
            "NOT returned — use extract_text and scan the text yourself for "
            "those.\n"
            "\n"
            "Three link kinds are reported: 'external' for http/https URLs "
            "(with `url`), 'email' for mailto: targets (with `email`), and "
            "'internal' for jumps within the document (with `target_page`, "
            "1-based). The three include_* flags switch those off "
            "individually; all default to true. Anything else — launch, "
            "remote-file and named actions — comes back as type 'other' and "
            "is ALWAYS included no matter how the flags are set. Note that a "
            "URL with some other scheme (ftp:, file:) is dropped entirely "
            "and appears in no category.\n"
            "\n"
            "`pages` is a plain comma/range STRING of 1-based page numbers, "
            "NOT JSON: \"3\", \"1,3,5\", \"2-7\", or mixed \"1,4-6,9\"; "
            "ranges are inclusive. Omit it for the whole document. Careful: "
            "if the string is malformed, or names only pages that do not "
            "exist, this tool SILENTLY FALLS BACK to scanning every page "
            "instead of failing — check links_summary.pages_analyzed to see "
            "what it really looked at."
        ),
        annotations={
            "readOnlyHint": True,        # reads only, writes nothing
            "idempotentHint": True,
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
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
            pdf_path: Path to the PDF, or an HTTPS URL to fetch.
            pages: Comma/range string of 1-based page numbers, e.g.
                "1,4-6,9". Ranges are inclusive. None (the default) scans
                every page, as does an unparseable or out-of-range string.
            include_internal: Keep same-document jump links (type
                "internal"). Default true.
            include_external: Keep http/https links (type "external").
                Default true.
            include_email: Keep mailto: links (type "email"). Default true.

        Returns:
            Dict with success, links_summary (total_links, per-type counts,
            pages_with_links, pages_analyzed), the links list with page,
            coordinates, type and target, link_analysis (top_domains,
            unique_domains, email_addresses) and the filter settings used.
            Links of type "other" ignore the include_* flags.
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = pymupdf.open(str(path))
            total_pages = len(doc)

            # Parse pages parameter
            parsed_pages = parse_pages_parameter(pages)
            page_numbers = parsed_pages if parsed_pages else list(range(total_pages))
            page_numbers = [p for p in page_numbers if 0 <= p < total_pages]

            # If parsing failed but pages was specified, use all pages
            if pages and not page_numbers:
                page_numbers = list(range(total_pages))

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
                        if link["kind"] == pymupdf.LINK_URI:
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

                        elif link["kind"] == pymupdf.LINK_GOTO:
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
                    "total_pages": total_pages,
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
        description=(
            "Survey a PDF's visual elements, guess which are charts or "
            "diagrams, and optionally crop each one out to a PNG.\n"
            "\n"
            "Without output_directory it writes nothing and returns only the "
            "inventory: page, kind, dimensions and a likely_chart flag per "
            "element. Pass output_directory to also render every flagged "
            "element, cropped from its page at render_dpi, with the file path "
            "on each element as image_path. Cropping the PAGE rather than "
            "pulling the underlying object means a chart assembled from many "
            "vector strokes comes out as one picture, and a raster chart "
            "comes out composited with anything drawn over it.\n"
            "\n"
            "Related tools, for when this is the wrong shape: extract_images "
            "for the embedded bitmaps as stored, extract_vector_graphics for "
            "SVG line art you want to stay vector, convert_to_images to "
            "render whole pages.\n"
            "\n"
            "Two kinds of element are counted. Embedded raster images are "
            "measured in PIXELS and flagged likely_chart when they are "
            "wider than 200 and taller than 150, or larger than 50,000 px "
            "in area. Vector drawings are only considered at all if they "
            "have more than 10 path items, are measured in PDF POINTS, and "
            "are flagged likely_chart when they have more than 20 items and "
            "exceed 200 wide or 150 tall.\n"
            "\n"
            "The judgement is PURE GEOMETRY — nothing reads the content — so "
            "a photograph, a full-page scan or a large logo is happily "
            "reported as a likely chart, and a small tidy bar chart is "
            "missed. Treat likely_chart as 'big enough to be worth a look'.\n"
            "\n"
            "min_size (default 100) is the floor for an element to be listed "
            "at all, applied to width OR height, and it is unhelpfully "
            "compared against PIXELS for images but POINTS for drawings; "
            "lower it to catch small figures, raise it to cut noise. "
            "`pages` is a plain comma/range STRING of 1-based page numbers, "
            "NOT JSON: \"3\", \"1,3,5\", \"2-7\", \"1,4-6,9\". A malformed "
            "or out-of-range string SILENTLY falls back to the whole "
            "document, so check chart_analysis.pages_analyzed."
        ),
        annotations={
            # Writes PNGs only when output_directory is supplied; a call
            # without it touches nothing. The hints describe what the tool
            # CAN do, so they have to assume the writing path.
            "readOnlyHint": False,
            "destructiveHint": True,     # overwrites same-named PNGs in the target dir
            "idempotentHint": True,      # same PDF and dpi produce the same crops
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
    )
    async def extract_charts(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        min_size: int = 100,
        output_directory: Optional[str] = None,
        render_dpi: int = 150
    ) -> Dict[str, Any]:
        """
        Inventory and score the visual elements in a PDF. Writes nothing.

        Args:
            pdf_path: Path to the PDF, or an HTTPS URL to fetch.
            pages: Comma/range string of 1-based page numbers, e.g.
                "1,4-6,9". Ranges are inclusive. None (the default) scans
                every page, as does an unparseable or out-of-range string.
            min_size: Minimum width or height for an element to be listed,
                default 100. Pixels for embedded images, PDF points for
                vector drawings.
            output_directory: Where to write cropped PNGs of the elements
                flagged likely_chart, created if missing. Omit it (the
                default) to get the inventory only and write nothing.
            render_dpi: Resolution for those crops, default 150, clamped to
                36-600. Only used when output_directory is supplied.

        Returns:
            Dict with success, chart_analysis (total_visual_elements,
            likely_charts, pages_with_visuals, pages_analyzed, chart_density,
            charts_written, output_directory), size_distribution bucketed by
            area (small <20000, medium <100000, large >=100000), the
            visual_elements inventory, and plain-language insights. Elements
            that were rendered carry image_path and image_size_bytes; the rest
            do not, so check for the key rather than assuming it.
        """
        start_time = time.time()

        try:
            path = await validate_pdf_path(pdf_path)
            doc = pymupdf.open(str(path))
            total_pages = len(doc)

            # Parse pages parameter
            parsed_pages = parse_pages_parameter(pages)
            page_numbers = parsed_pages if parsed_pages else list(range(total_pages))
            page_numbers = [p for p in page_numbers if 0 <= p < total_pages]

            # If parsing failed but pages was specified, use all pages
            if pages and not page_numbers:
                page_numbers = list(range(total_pages))

            visual_elements = []
            charts_found = 0

            # (page_index, clip_rect, element) for every detected region, so
            # the rendering pass below can crop each one out of its page.
            _render_queue = []

            for page_num in page_numbers:
                try:
                    page = doc[page_num]

                    # Analyze images (potential charts)
                    images = page.get_images()
                    for img_index, img in enumerate(images):
                        try:
                            xref = img[0]
                            pix = pymupdf.Pixmap(doc, xref)

                            if pix.width >= min_size or pix.height >= min_size:
                                # Heuristic: larger images are more likely to be charts
                                is_likely_chart = (pix.width > 200 and pix.height > 150) or (pix.width * pix.height > 50000)

                                # Placement rectangle on the page, needed to
                                # render the region. May be empty if the image
                                # is referenced but never drawn.
                                placements = page.get_image_rects(xref)
                                clip_rect = placements[0] if placements else None

                                element = {
                                    "page": page_num + 1,
                                    "type": "image",
                                    "element_index": img_index + 1,
                                    "width": pix.width,
                                    "height": pix.height,
                                    "area": pix.width * pix.height,
                                    "likely_chart": is_likely_chart
                                }
                                if clip_rect is not None:
                                    _render_queue.append((page_num, clip_rect, element))

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
                                rect = drawing.get("rect", pymupdf.Rect(0, 0, 0, 0))
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
                                    _render_queue.append((page_num, rect, element))

                                    visual_elements.append(element)
                                    if is_likely_chart:
                                        charts_found += 1
                        except:
                            pass

                except Exception as e:
                    logger.warning(f"Failed to analyze page {page_num + 1}: {e}")

            # Render the detected regions to PNG, if asked. Until 2026-09-22
            # this tool wrote nothing at all despite being called "extract":
            # it returned an inventory with a geometry-derived likely_chart
            # flag and no way to actually see any of them. Cropping the page
            # at each region handles raster and vector identically, and
            # captures the chart as it is composed on the page rather than as
            # whatever isolated image object happens to sit underneath.
            charts_written = 0
            output_dir = None
            if output_directory:
                output_dir = validate_output_path(output_directory)
                output_dir.mkdir(parents=True, exist_ok=True)
                dpi = max(36, min(600, render_dpi))
                for page_index, clip, element in _render_queue:
                    if not element.get("likely_chart"):
                        continue          # only the flagged ones
                    try:
                        if clip is None or clip.is_empty or clip.is_infinite:
                            continue
                        pix = doc[page_index].get_pixmap(clip=clip, dpi=dpi)
                        name = (f"{path.stem}_p{element['page']}"
                                f"_{element['type']}{element['element_index']}.png")
                        target = output_dir / name
                        pix.save(str(target))
                        element["image_path"] = str(target)
                        element["image_size_bytes"] = target.stat().st_size
                        charts_written += 1
                    except Exception as exc:
                        logger.warning(
                            "Could not render chart on page %s: %s",
                            element.get("page"), exc
                        )

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
                    "chart_density": round(charts_found / len(page_numbers), 2) if page_numbers else 0,
                    "charts_written": charts_written,
                    "output_directory": str(output_dir) if output_dir else None
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
                    "total_pages": total_pages
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
        description=(
            "Attach input constraints to the AcroForm fields of an existing "
            "PDF, writing a NEW PDF to output_path. It changes the FORM "
            "ITSELF so a person typing into it is constrained; it does not "
            "check any data. To check values you already have against rules, "
            "use validate_form_data (pure data, writes nothing). To put "
            "values INTO the form, use fill_form_pdf.\n"
            "\n"
            "`validation_rules` is a JSON OBJECT keyed by EXACT form field "
            "name (not an array), each mapping to a rules object:\n"
            '  {"applicant_name": {"max_length": 40},\n'
            '   "parcel_id":      {"max_length": 12, "required": true}}\n'
            "\n"
            "ONLY TWO KEYS ARE READ, and only one of them does anything:\n"
            "  max_length (int) — really applied, caps the characters the "
            "field will accept\n"
            "  required (bool)  — counted in the response but NOT enforced "
            "in the output PDF; it sets no flag a reader honours\n"
            "Any other key (pattern, min_length, type, format...) is ignored "
            "silently. So this tool is effectively a max-length setter "
            "today, and the output is a valid PDF either way.\n"
            "\n"
            "Get the field names from extract_form_data first: a name that "
            "does not match a real field is skipped without error, and "
            "validation_summary.fields_processed will simply be lower than "
            "your rule count. Only classic AcroForm text widgets can be "
            "changed — dynamic XFA forms have no widgets to touch (check "
            "with is_xfa_pdf)."
        ),
        annotations={
            "readOnlyHint": False,       # writes a new PDF
            "destructiveHint": True,     # overwrites output_path if it exists
            "idempotentHint": True,      # same args produce the same output file
            "openWorldHint": True,       # input_path may be an HTTPS URL
        },
    )
    async def add_field_validation(
        self,
        input_path: str,
        output_path: str,
        validation_rules: str
    ) -> Dict[str, Any]:
        """
        Add input constraints to existing PDF AcroForm fields.

        Args:
            input_path: Path to the source PDF with form fields, or an HTTPS
                URL to fetch.
            output_path: Where to write the modified PDF. Overwritten if it
                exists; the source is never modified in place.
            validation_rules: JSON object mapping exact field name to a rules
                object. Recognised keys are "max_length" (applied) and
                "required" (counted only, not enforced); everything else is
                ignored.
                Example: {"parcel_id": {"max_length": 12}}

        Returns:
            Dict with success, validation_summary (fields_processed,
            rules_applied, validation_rules_count, output size),
            applied_rules listing the field names you supplied, and the
            output path. fields_processed lower than your rule count means
            some names did not match a real field.
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
            doc = pymupdf.open(str(input_pdf_path))
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
        description=(
            "Concatenate two or more PDFs into one NEW PDF at output_path, "
            "CARRYING THE SOURCE BOOKMARKS ACROSS. That is the whole reason "
            "to choose this over merge_pdfs, which is otherwise identical "
            "but throws every bookmark away. If none of the sources have "
            "bookmarks and you do not want a contents page, merge_pdfs is "
            "the simpler call.\n"
            "\n"
            "`input_paths` is a JSON array of path strings, in the order you "
            "want them concatenated:\n"
            '  ["/docs/part1.pdf", "/docs/part2.pdf"]\n'
            "AT LEAST TWO paths are required. Each may be a local path or an "
            "HTTPS URL. A source that fails to open aborts the whole call "
            "and writes nothing.\n"
            "\n"
            "preserve_bookmarks (default true) rebuilds a combined outline "
            "with each entry's page number shifted to its new position, and "
            "PREFIXES every title with its source filename, so 'Chapter 1' "
            "from part1.pdf becomes 'part1.pdf: Chapter 1'. Set it false for "
            "unprefixed titles, which discards the outline entirely.\n"
            "\n"
            "include_toc (default false) inserts one extra page at the very "
            "front listing each source file and its PAGE COUNT, as plain "
            "text with no clickable links. It is a cover sheet, not a real "
            "index; the bookmark outline above is the navigable one and "
            "stays correct when both options are used together.\n"
            "\n"
            "add_page_numbers stamps \"N / total\" in grey at the bottom "
            "centre of every page. It runs after the contents page is "
            "inserted, so that page is numbered too and the totals match the "
            "finished document. merge_summary.pages_numbered reports how "
            "many were stamped."
        ),
        annotations={
            "readOnlyHint": False,       # writes a new PDF
            "destructiveHint": True,     # overwrites output_path if it exists
            "idempotentHint": True,      # same inputs produce the same output file
            "openWorldHint": True,       # entries in input_paths may be HTTPS URLs
        },
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
        Merge PDFs, carrying the source bookmark outlines into the result.

        Args:
            input_paths: JSON array of PDF paths (local paths or HTTPS URLs)
                in concatenation order. Minimum of 2 entries.
                Example: ["/tmp/part1.pdf", "/tmp/part2.pdf"]
            output_path: Where to write the merged PDF. Overwritten if it
                exists; the sources are never modified.
            preserve_bookmarks: Carry each source's outline across with
                page numbers rewritten and titles prefixed by source
                filename. Default true.
            add_page_numbers: Stamp "N / total" at the bottom centre of each
                page, 9pt grey, after any contents page is inserted so the
                numbering covers it and matches the final page count.
                Default false.
            include_toc: Prepend a plain-text cover page listing each source
                file and its page count. No links. Default false.

        Returns:
            Dict with success, merge_summary (input_files,
            total_pages_merged, bookmarks_preserved, toc_generated, output
            size), per-input file_info including has_bookmarks,
            merge_features echoing the options, and the output path.
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
                    doc = pymupdf.open(str(validated_path))
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
            merged_doc = pymupdf.open()
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
                # "hebo" is PyMuPDF's base-14 name for Helvetica-Bold.
                # "helv-bold" is not a recognized name and raises
                # "need font file or buffer", which failed the whole merge.
                toc_page.insert_text((50, 50), "Table of Contents", fontsize=16, fontname="hebo")

                y_pos = 100
                for info in file_info:
                    filename = info['path'].split('/')[-1]
                    toc_line = f"{filename} - Pages {info['pages']}"
                    toc_page.insert_text((50, y_pos), toc_line, fontsize=12)
                    y_pos += 20

            # Stamp page numbers. Previously this parameter was accepted,
            # echoed back in merge_features, and never acted on: no stamping
            # code existed anywhere in the method.
            #
            # Runs AFTER the TOC insert on purpose, so the generated contents
            # page is itself numbered and the numbers match the final document
            # rather than being off by one wherever include_toc was used.
            pages_numbered = 0
            if add_page_numbers:
                total = merged_doc.page_count
                for idx in range(total):
                    try:
                        page = merged_doc[idx]
                        label = f"{idx + 1} / {total}"
                        rect = page.rect
                        # Bottom centre, 28pt up from the trim edge. insert_text
                        # takes the text BASELINE, so this clears the margin on
                        # a Letter or A4 page without colliding with body text.
                        page.insert_text(
                            (rect.width / 2 - len(label) * 2.5, rect.height - 28),
                            label,
                            fontsize=9,
                            fontname="helv",
                            color=(0.35, 0.35, 0.35),
                        )
                        pages_numbered += 1
                    except Exception as exc:
                        logger.warning("Could not number page %d: %s", idx + 1, exc)

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
                    "pages_numbered": pages_numbered,
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
        description=(
            "Split a PDF into several new PDFs at PAGE RANGES YOU CHOOSE, "
            "written into an output_directory you choose (created if it does "
            "not exist) with filenames you control. This is the splitting "
            "tool to reach for by default. Prefer split_pdf_by_bookmarks "
            "when the document's own outline already marks the boundaries, "
            "split_pdf_by_structure when it has visible headings but no "
            "bookmarks, and reorder_pdf_pages when you want the selected "
            "pages in ONE file rather than several.\n"
            "\n"
            "`page_ranges` MUST BE A JSON ARRAY OF STRINGS. Quote every "
            "element, and keep the outer brackets even for a single range:\n"
            '  \'["1-5", "6-10", "11-end"]\'   three files\n'
            '  \'["1-1"]\'                      just page 1, as one file\n'
            '  \'["7"]\'                        also just page 7\n'
            '  \'["1-3", "1-3"]\'               overlaps are allowed\n'
            "\n"
            "These all FAIL, and they are the easy mistakes to make:\n"
            "  \"1\"      -> parses as the number 1, not a list: "
            "\"'int' object is not iterable\"\n"
            "  \"1-1\"    -> not JSON at all: \"Invalid JSON in "
            "page_ranges\"\n"
            "  '[1, 5]' -> array of NUMBERS; each element is skipped and "
            "you get zero files with success still true\n"
            "The element itself is a 1-BASED inclusive range 'first-last', "
            "or a bare page number. The literal keyword 'end' is allowed as "
            "the last page only, as in \"11-end\".\n"
            "\n"
            "Ranges that run past the document are CLAMPED rather than "
            "rejected, so [\"50-60\"] on a 10-page PDF quietly yields one "
            "file holding page 10. A range whose text will not parse is "
            "skipped and the call still reports success, so always compare "
            "split_summary.files_created with ranges_requested.\n"
            "\n"
            "naming_pattern understands exactly three placeholders: {start} "
            "and {end} (the resolved 1-based page numbers) and {index} (the "
            "range's position, starting at 1). Default "
            "\"page_{start}-{end}.pdf\". Any OTHER placeholder makes that "
            "file silently fail, and two ranges that resolve to the same "
            "filename overwrite each other."
        ),
        annotations={
            "readOnlyHint": False,       # writes one new PDF per range
            "destructiveHint": True,     # clobbers same-named files in output_directory
            "idempotentHint": True,      # same args rewrite the same set of files
            "openWorldHint": True,       # input_path may be an HTTPS URL
        },
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
            input_path: Path to the source PDF, or an HTTPS URL to fetch.
            output_directory: Directory for the split files. Created along
                with any missing parents.
            page_ranges: JSON array of range STRINGS, 1-based and inclusive.
                Each element is "first-last", a bare page number, or
                "first-end" to run to the last page. Ranges beyond the
                document are clamped, not rejected.
                Example: ["1-5", "6-10", "11-end"]
            naming_pattern: Output filename template. Only {start}, {end}
                and {index} are substituted. Default
                "page_{start}-{end}.pdf".

        Returns:
            Dict with success, split_summary (input_pages, ranges_requested,
            files_created, total size), split_files listing each written
            file's name, path, page_range and page count, and the settings
            used. files_created below ranges_requested means some ranges
            were skipped.
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

            doc = pymupdf.open(str(input_pdf_path))
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
                        split_doc = pymupdf.open()
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
        description=(
            "Split a PDF at its own BOOKMARK (outline) entries, one new PDF "
            "per bookmark, into an output_directory you choose (created if "
            "missing). Each piece runs from its bookmark's page up to the "
            "page before the next bookmark AT THE SAME LEVEL, and the last "
            "piece runs to the end of the document.\n"
            "\n"
            "It REQUIRES a real embedded outline and fails with an error if "
            "the PDF has none, or none at the level you asked for. When "
            "there are no bookmarks but the pages do show headings, use "
            "split_pdf_by_structure, which detects them (and can emit "
            "markdown and images per section). Use split_pdf_by_pages when "
            "you want to name the page ranges yourself, or split_pdf for the "
            "zero-configuration level-1 split that writes beside the input "
            "instead of into a directory you pick.\n"
            "\n"
            "bookmark_level matches the outline depth EXACTLY, it is not "
            "'this level and above': 1 (the default) splits on top-level "
            "chapters, 2 splits on their subsections only. Run "
            "get_document_structure first if you are unsure what depths "
            "exist.\n"
            "\n"
            "naming_pattern understands exactly two placeholders, {title} "
            "and {index} (the 1-based position). Default \"{title}.pdf\". "
            "{start} and {end} DO NOT WORK here and would silently skip the "
            "file. The title is sanitised down to letters, digits, spaces, "
            "hyphens and underscores and truncated to 50 characters, so "
            "punctuation disappears and two chapters with similar names can "
            "collide and overwrite each other — prefer something like "
            "\"{index}_{title}.pdf\" for a clean set.\n"
            "\n"
            "Two things go missing quietly. Any pages BEFORE the first "
            "bookmark (cover, front matter) end up in no output file at "
            "all. And two bookmarks landing on the same page produce an "
            "empty span that is skipped. Compare "
            "split_summary.files_created with bookmarks_at_level to catch "
            "both."
        ),
        annotations={
            "readOnlyHint": False,       # writes one new PDF per bookmark
            "destructiveHint": True,     # clobbers same-named files in output_directory
            "idempotentHint": True,      # same args rewrite the same set of files
            "openWorldHint": True,       # input_path may be an HTTPS URL
        },
    )
    async def split_pdf_by_bookmarks(
        self,
        input_path: str,
        output_directory: str,
        bookmark_level: int = 1,
        naming_pattern: str = "{title}.pdf"
    ) -> Dict[str, Any]:
        """
        Split PDF using its embedded bookmarks as breakpoints.

        Args:
            input_path: Path to the source PDF, or an HTTPS URL to fetch.
                Must contain an embedded outline.
            output_directory: Directory for the split files. Created along
                with any missing parents.
            bookmark_level: Exact outline depth to split on. 1 (default) is
                top level, 2 is the next level down. Levels are not nested
                together.
            naming_pattern: Output filename template. Only {title} (the
                sanitised bookmark text, max 50 chars) and {index} (1-based)
                are substituted. Default "{title}.pdf".

        Returns:
            Dict with success, split_summary (input_pages,
            bookmarks_at_level, files_created, bookmark_level, total size),
            split_files listing each file's name, path, bookmark_title,
            page_range and page count, and the settings used.
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(input_path)
            output_dir = validate_output_path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            doc = pymupdf.open(str(input_pdf_path))
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
                        split_doc = pymupdf.open()
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