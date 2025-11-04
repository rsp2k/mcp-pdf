"""
Document Assembly Mixin - PDF merging, splitting, and page manipulation
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


class DocumentAssemblyMixin(MCPMixin):
    """
    Handles PDF document assembly operations including merging, splitting, and reordering.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()
        self.max_file_size = 100 * 1024 * 1024  # 100MB

    @mcp_tool(
        name="merge_pdfs",
        description="Merge multiple PDFs into one document"
    )
    async def merge_pdfs(
        self,
        pdf_paths: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Merge multiple PDF files into a single document.

        Args:
            pdf_paths: JSON string containing list of PDF file paths
            output_path: Path where merged PDF will be saved

        Returns:
            Dictionary containing merge results
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
            output_pdf_path = await validate_output_path(output_path)

            # Validate and open all input PDFs
            input_docs = []
            file_info = []

            for i, pdf_path in enumerate(paths_list):
                try:
                    validated_path = await validate_pdf_path(pdf_path)
                    doc = fitz.open(str(validated_path))
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
            merged_doc = fitz.open()
            total_pages_merged = 0

            for i, doc in enumerate(input_docs):
                try:
                    merged_doc.insert_pdf(doc)
                    total_pages_merged += len(doc)
                    logger.info(f"Merged document {i + 1}: {len(doc)} pages")
                except Exception as e:
                    logger.error(f"Failed to merge document {i + 1}: {e}")

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
        description="Split PDF into separate documents"
    )
    async def split_pdf(
        self,
        pdf_path: str,
        split_method: str = "pages"
    ) -> Dict[str, Any]:
        """
        Split PDF document into separate files.

        Args:
            pdf_path: Path to PDF file to split
            split_method: Method to use ("pages", "bookmarks", "ranges")

        Returns:
            Dictionary containing split results
        """
        start_time = time.time()

        try:
            # Validate input path
            input_pdf_path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(input_pdf_path))
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

                    page_doc = fitz.open()
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

                        split_doc = fitz.open()
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

                    chunk_doc = fitz.open()
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
        description="Reorder pages in PDF document"
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
            pdf_path: Path to input PDF file
            page_order: JSON string with new page order (1-based page numbers)
            output_path: Path where reordered PDF will be saved

        Returns:
            Dictionary containing reorder results
        """
        start_time = time.time()

        try:
            # Validate paths
            input_pdf_path = await validate_pdf_path(pdf_path)
            output_pdf_path = await validate_output_path(output_path)

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
            input_doc = fitz.open(str(input_pdf_path))
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
            output_doc = fitz.open()

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