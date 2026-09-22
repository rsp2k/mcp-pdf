"""
Document Assembly Mixin - PDF merging, splitting, and page manipulation
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import time
import json
from typing import Dict, Any, Literal
import logging

# PDF processing libraries
import pymupdf

# Official FastMCP mixin
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, validate_output_path, sanitize_error_message

logger = logging.getLogger(__name__)


class DocumentAssemblyMixin(MCPMixin):
    """
    Handles PDF document assembly operations including merging, splitting, and reordering.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()

    @mcp_tool(
        name="merge_pdfs",
        description=(
            "Concatenate two or more PDFs end-to-end into one NEW PDF at "
            "output_path, copying pages in the order given and carrying each "
            "source's bookmarks across with their page targets rebased onto "
            "the merged document. The response reports bookmarks_preserved "
            "and bookmarks_dropped (a bookmark whose target page is missing "
            "or non-positive cannot be rebased and is counted, not silently "
            "lost).\n"
            "\n"
            "Use merge_pdfs_advanced instead when you want each bookmark "
            "title prefixed with its source filename, a generated contents "
            "page, or per-file page ranges.\n"
            "\n"
            "`pdf_paths` is a JSON array of path strings, in the order you "
            "want them concatenated:\n"
            '  ["/docs/cover.pdf", "/docs/body.pdf", "/docs/appendix.pdf"]\n'
            "\n"
            "AT LEAST TWO paths are required; a single-element array is "
            "rejected. Each entry may be a local path or an HTTPS URL. A "
            "source that fails to OPEN aborts the whole call; a source that "
            "opens but fails to copy is skipped with the call still "
            "reporting success, so check merge_summary.total_pages_merged "
            "against the pages you expected."
        ),
        annotations={
            "readOnlyHint": False,       # writes a new PDF
            "destructiveHint": True,     # overwrites output_path if it exists
            "idempotentHint": True,      # same inputs produce the same output file
            "openWorldHint": True,       # entries in pdf_paths may be HTTPS URLs
        },
    )
    async def merge_pdfs(
        self,
        pdf_paths: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Merge multiple PDF files into a single document.

        Args:
            pdf_paths: JSON array of PDF paths (local paths or HTTPS URLs) in
                concatenation order. Minimum of 2 entries.
                Example: ["/tmp/a.pdf", "/tmp/b.pdf"]
            output_path: Where to write the merged PDF. Overwritten if it
                already exists; the sources are never modified.

        Returns:
            Dict with success, merge_summary (input_files,
            total_pages_merged, bookmarks_preserved, bookmarks_dropped,
            output size), per-input file_info, and the output path.
            Bookmarks from every source are carried across with their page
            targets rebased onto the merged document.
        """
        start_time = time.time()

        try:
            # Parse input paths
            try:
                paths_list = json.loads(pdf_paths)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON in pdf_paths: {e}",
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

            # Validate and open all input PDFs
            input_docs = []
            file_info = []

            for i, pdf_path in enumerate(paths_list):
                try:
                    validated_path = await validate_pdf_path(pdf_path)
                    doc = pymupdf.open(str(validated_path))
                    input_docs.append(doc)

                    file_info.append({
                        "index": i + 1,
                        "path": str(validated_path),
                        "pages": len(doc),
                        "size_bytes": validated_path.stat().st_size
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
            total_pages_merged = 0
            # Document.insert_pdf() copies pages but NOT the outline, and
            # nothing here ever called set_toc(), so every bookmark in every
            # input was silently dropped. Accumulate them instead, shifting
            # each source's page targets by the number of pages already
            # merged, and write the combined outline once at the end.
            combined_toc = []
            bookmarks_dropped = 0

            for i, doc in enumerate(input_docs):
                try:
                    page_offset = total_pages_merged
                    for entry in (doc.get_toc() or []):
                        level, title, page = entry[0], entry[1], entry[2]
                        if page and page > 0:
                            combined_toc.append([level, title, page + page_offset])
                        else:
                            # A non-positive target means the bookmark points
                            # at no page (some producers emit -1); it cannot be
                            # rebased, so count it rather than silently lose it.
                            bookmarks_dropped += 1
                    merged_doc.insert_pdf(doc)
                    total_pages_merged += len(doc)
                    logger.info(f"Merged document {i + 1}: {len(doc)} pages")
                except Exception as e:
                    logger.error(f"Failed to merge document {i + 1}: {e}")

            if combined_toc:
                try:
                    merged_doc.set_toc(combined_toc)
                except Exception as e:
                    # A malformed outline should not cost the caller the merge
                    # itself, which is the expensive part and already done.
                    logger.warning(f"Could not write merged bookmarks: {e}")
                    bookmarks_dropped += len(combined_toc)
                    combined_toc = []

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
                    "total_pages_merged": total_pages_merged,
                    "bookmarks_preserved": len(combined_toc),
                    "bookmarks_dropped": bookmarks_dropped,
                    "output_size_bytes": output_size,
                    "output_size_mb": round(output_size / (1024 * 1024), 2)
                },
                "input_files": file_info,
                "output_info": {
                    "output_path": str(output_pdf_path),
                    "total_pages": total_pages_merged
                },
                "merge_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"PDF merge failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "merge_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="split_pdf",
        description=(
            "Split a PDF into several new PDFs with ZERO configuration. It "
            "takes no output location and no range list: the pieces are "
            "written NEXT TO THE INPUT FILE, in the input's own directory, "
            "using names derived from the input's filename. Reach for this "
            "only when the built-in behaviour is exactly what you want; "
            "every other splitting need has a better-targeted tool.\n"
            "\n"
            "split_method picks one of three fixed behaviours:\n"
            "  'pages'     (default) — one file per page:\n"
            "                {stem}_page_1.pdf, {stem}_page_2.pdf, ...\n"
            "  'bookmarks' — one file per TOP-LEVEL (level 1) bookmark:\n"
            "                {stem}_{Bookmark Title}.pdf. FAILS with an "
            "error if the PDF has no bookmarks.\n"
            "  'ranges'    — FIXED 10-page chunks, NOT a list you supply:\n"
            "                {stem}_pages_1-10.pdf, {stem}_pages_11-20.pdf, "
            "...\n"
            "\n"
            "Choose a different tool when:\n"
            "  - you want YOUR OWN page ranges, or output in a chosen "
            "directory, or control over filenames -> split_pdf_by_pages\n"
            "  - you want bookmark splitting at a level other than 1, or a "
            "chosen output directory -> split_pdf_by_bookmarks\n"
            "  - the PDF has NO bookmarks but does have visible headings, or "
            "you want markdown/images per chapter -> split_pdf_by_structure\n"
            "  - you want to keep only some pages, or reorder/duplicate them "
            "into ONE file -> reorder_pdf_pages\n"
            "\n"
            "The PDF must have MORE THAN ONE page or the call fails. If "
            "pdf_path is an HTTPS URL the outputs land beside the downloaded "
            "copy in the temp cache directory, not in any directory of "
            "yours, so prefer a local path here."
        ),
        annotations={
            "readOnlyHint": False,       # writes one new PDF per piece
            "destructiveHint": True,     # clobbers same-named files beside the input
            "idempotentHint": True,      # same input + method rewrites the same set
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
    )
    async def split_pdf(
        self,
        pdf_path: str,
        split_method: Literal["pages", "bookmarks", "ranges"] = "pages"
    ) -> Dict[str, Any]:
        """
        Split PDF document into separate files.

        Args:
            pdf_path: Path to the PDF to split, or an HTTPS URL. Must have
                more than 1 page. Output files are written to this file's
                parent directory.
            split_method: Which fixed strategy to use.
                - "pages": one file per page, {stem}_page_N.pdf
                - "bookmarks": one file per level-1 bookmark,
                  {stem}_{title}.pdf; errors if the PDF has no TOC
                - "ranges": consecutive 10-page chunks (the chunk size is
                  hard-coded, not a parameter), {stem}_pages_A-B.pdf

        Returns:
            Dict with success, split_summary (method, input_pages,
            output_files, total size), and split_files listing each written
            file's path, page count and page_range.
        """
        start_time = time.time()

        try:
            # Validate input path
            input_pdf_path = await validate_pdf_path(pdf_path)
            doc = pymupdf.open(str(input_pdf_path))
            total_pages = len(doc)

            if total_pages <= 1:
                doc.close()
                return {
                    "success": False,
                    "error": "PDF must have more than 1 page to split",
                    "split_time": round(time.time() - start_time, 2)
                }

            split_files = []
            base_path = input_pdf_path.parent
            base_name = input_pdf_path.stem

            if split_method == "pages":
                # Split into individual pages
                for page_num in range(total_pages):
                    output_path = base_path / f"{base_name}_page_{page_num + 1}.pdf"

                    page_doc = pymupdf.open()
                    page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    page_doc.save(str(output_path))
                    page_doc.close()

                    split_files.append({
                        "file_path": str(output_path),
                        "pages": 1,
                        "page_range": f"{page_num + 1}",
                        "size_bytes": output_path.stat().st_size
                    })

            elif split_method == "bookmarks":
                # Split by bookmarks/table of contents
                toc = doc.get_toc()

                if not toc:
                    doc.close()
                    return {
                        "success": False,
                        "error": "No bookmarks found in PDF for bookmark-based splitting",
                        "split_time": round(time.time() - start_time, 2)
                    }

                # Create splits based on top-level bookmarks
                top_level_bookmarks = [item for item in toc if item[0] == 1]  # Level 1 bookmarks

                for i, bookmark in enumerate(top_level_bookmarks):
                    start_page = bookmark[2] - 1  # Convert to 0-based

                    # Determine end page
                    if i + 1 < len(top_level_bookmarks):
                        end_page = top_level_bookmarks[i + 1][2] - 2  # Convert to 0-based, inclusive
                    else:
                        end_page = total_pages - 1

                    if start_page <= end_page:
                        # Clean bookmark title for filename
                        clean_title = "".join(c for c in bookmark[1] if c.isalnum() or c in (' ', '-', '_')).strip()
                        clean_title = clean_title[:50]  # Limit length

                        output_path = base_path / f"{base_name}_{clean_title}.pdf"

                        split_doc = pymupdf.open()
                        split_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
                        split_doc.save(str(output_path))
                        split_doc.close()

                        split_files.append({
                            "file_path": str(output_path),
                            "pages": end_page - start_page + 1,
                            "page_range": f"{start_page + 1}-{end_page + 1}",
                            "bookmark_title": bookmark[1],
                            "size_bytes": output_path.stat().st_size
                        })

            elif split_method == "ranges":
                # Split into chunks of 10 pages each
                chunk_size = 10
                chunks = (total_pages + chunk_size - 1) // chunk_size

                for chunk in range(chunks):
                    start_page = chunk * chunk_size
                    end_page = min(start_page + chunk_size - 1, total_pages - 1)

                    output_path = base_path / f"{base_name}_pages_{start_page + 1}-{end_page + 1}.pdf"

                    chunk_doc = pymupdf.open()
                    chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
                    chunk_doc.save(str(output_path))
                    chunk_doc.close()

                    split_files.append({
                        "file_path": str(output_path),
                        "pages": end_page - start_page + 1,
                        "page_range": f"{start_page + 1}-{end_page + 1}",
                        "size_bytes": output_path.stat().st_size
                    })

            doc.close()

            total_output_size = sum(f["size_bytes"] for f in split_files)

            return {
                "success": True,
                "split_summary": {
                    "split_method": split_method,
                    "input_pages": total_pages,
                    "output_files": len(split_files),
                    "total_output_size_bytes": total_output_size,
                    "total_output_size_mb": round(total_output_size / (1024 * 1024), 2)
                },
                "split_files": split_files,
                "input_info": {
                    "input_path": str(input_pdf_path),
                    "total_pages": total_pages
                },
                "split_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"PDF split failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "split_time": round(time.time() - start_time, 2)
            }

    @mcp_tool(
        name="reorder_pdf_pages",
        description=(
            "Rebuild a PDF's pages in an order you specify, writing ONE new "
            "PDF to output_path. The output contains exactly the pages you "
            "list, in the order you list them, so this single tool also "
            "covers three jobs that sound like other tools:\n"
            "  - EXTRACT a subset: list only the pages you want to keep\n"
            "  - DUPLICATE a page: list it more than once\n"
            "  - DELETE pages: leave them out of the list\n"
            "Unlike the split_* tools it never produces multiple files; use "
            "split_pdf_by_pages when you want several separate PDFs.\n"
            "\n"
            "`page_order` is a JSON array of 1-BASED page numbers:\n"
            '  [3, 1, 2]        -> a 3-page doc with page 3 first\n'
            '  [1, 1, 5]        -> page 1 twice, then page 5 (3 pages out)\n'
            '  [2, 3]           -> keeps only pages 2 and 3\n'
            "\n"
            "Every listed page must exist. A number below 1, above the page "
            "count, or not parseable as an integer fails the WHOLE call with "
            "no file written and the offending values echoed back, so "
            "nothing is written from a partly-bad list. The response's "
            "page_mapping reports pages_duplicated and pages_omitted so you "
            "can confirm you got the document you meant."
        ),
        annotations={
            "readOnlyHint": False,       # writes a new PDF
            "destructiveHint": True,     # overwrites output_path if it exists
            "idempotentHint": True,      # same args produce the same output file
            "openWorldHint": True,       # pdf_path may be an HTTPS URL
        },
    )
    async def reorder_pdf_pages(
        self,
        pdf_path: str,
        page_order: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Reorder pages in a PDF document according to specified order.

        Args:
            pdf_path: Path to the source PDF, or an HTTPS URL to fetch.
            page_order: JSON array of 1-based page numbers giving the output
                order. Repeats duplicate a page; omissions drop it. Any
                out-of-range or non-numeric entry aborts the call.
                Example: [3, 1, 2]
            output_path: Where to write the reordered PDF. Overwritten if it
                exists; the source is never modified in place.

        Returns:
            Dict with success, reorder_summary (input/output page counts,
            output size), page_mapping (new_order, pages_duplicated,
            pages_omitted), and the output path.
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(pdf_path)
            output_pdf_path = validate_output_path(output_path)

            # Parse page order
            try:
                order_list = json.loads(page_order)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON in page_order: {e}",
                    "reorder_time": round(time.time() - start_time, 2)
                }

            if not isinstance(order_list, list):
                return {
                    "success": False,
                    "error": "page_order must be a list of page numbers",
                    "reorder_time": round(time.time() - start_time, 2)
                }

            # Open input document
            input_doc = pymupdf.open(str(input_pdf_path))
            total_pages = len(input_doc)

            # Validate page numbers (convert to 0-based)
            valid_pages = []
            invalid_pages = []

            for page_num in order_list:
                try:
                    page_index = int(page_num) - 1  # Convert to 0-based
                    if 0 <= page_index < total_pages:
                        valid_pages.append(page_index)
                    else:
                        invalid_pages.append(page_num)
                except (ValueError, TypeError):
                    invalid_pages.append(page_num)

            if invalid_pages:
                input_doc.close()
                return {
                    "success": False,
                    "error": f"Invalid page numbers: {invalid_pages}. Pages must be between 1 and {total_pages}",
                    "reorder_time": round(time.time() - start_time, 2)
                }

            # Create reordered document
            output_doc = pymupdf.open()

            for page_index in valid_pages:
                try:
                    output_doc.insert_pdf(input_doc, from_page=page_index, to_page=page_index)
                except Exception as e:
                    logger.warning(f"Failed to copy page {page_index + 1}: {e}")

            # Save reordered document
            output_doc.save(str(output_pdf_path))
            output_size = output_pdf_path.stat().st_size

            input_doc.close()
            output_doc.close()

            return {
                "success": True,
                "reorder_summary": {
                    "input_pages": total_pages,
                    "output_pages": len(valid_pages),
                    "pages_reordered": len(valid_pages),
                    "output_size_bytes": output_size,
                    "output_size_mb": round(output_size / (1024 * 1024), 2)
                },
                "page_mapping": {
                    "original_order": list(range(1, total_pages + 1)),
                    "new_order": [p + 1 for p in valid_pages],
                    "pages_duplicated": len(valid_pages) - len(set(valid_pages)),
                    "pages_omitted": total_pages - len(set(valid_pages))
                },
                "output_info": {
                    "output_path": str(output_pdf_path),
                    "total_pages": len(valid_pages)
                },
                "reorder_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"PDF page reorder failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "reorder_time": round(time.time() - start_time, 2)
            }