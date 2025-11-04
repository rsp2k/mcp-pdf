"""
Document Assembly Mixin - PDF merging, splitting, and reorganization
"""

import json
import time
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


class DocumentAssemblyMixin(MCPMixin):
    """
    Handles all PDF document assembly operations including merging, splitting, and reorganization.

    Tools provided:
    - merge_pdfs: Merge multiple PDFs into one document
    - split_pdf: Split PDF into multiple files
    - reorder_pdf_pages: Reorder pages in PDF document
    """

    def get_mixin_name(self) -> str:
        return "DocumentAssembly"

    def get_required_permissions(self) -> List[str]:
        return ["read_files", "write_files", "document_assembly"]

    def _setup(self):
        """Initialize document assembly specific configuration"""
        self.max_merge_files = 50
        self.max_split_parts = 100

    @mcp_tool(
        name="merge_pdfs",
        description="Merge multiple PDFs into one document"
    )
    async def merge_pdfs(
        self,
        pdf_paths: str,  # Comma-separated list of PDF file paths
        output_filename: str = "merged_document.pdf"
    ) -> Dict[str, Any]:
        """
        Merge multiple PDFs into a single file.

        Args:
            pdf_paths: Comma-separated list of PDF file paths or URLs
            output_filename: Name for the merged output file

        Returns:
            Dictionary containing merge results
        """
        start_time = time.time()

        try:
            # Parse PDF paths
            if isinstance(pdf_paths, str):
                path_list = [p.strip() for p in pdf_paths.split(',')]
            else:
                path_list = pdf_paths

            if len(path_list) < 2:
                return {
                    "success": False,
                    "error": "At least 2 PDF files are required for merging",
                    "merge_time": 0
                }

            # Validate all paths
            validated_paths = []
            for pdf_path in path_list:
                try:
                    validated_path = await validate_pdf_path(pdf_path)
                    validated_paths.append(validated_path)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Invalid path '{pdf_path}': {str(e)}",
                        "merge_time": 0
                    }

            # Validate output path
            output_file = validate_output_path(output_filename)

            # Create merged document
            merged_doc = fitz.open()
            merge_info = []

            for i, pdf_path in enumerate(validated_paths):
                try:
                    source_doc = fitz.open(str(pdf_path))
                    page_count = len(source_doc)

                    # Copy all pages from source to merged document
                    merged_doc.insert_pdf(source_doc)

                    merge_info.append({
                        "source_file": str(pdf_path),
                        "pages_added": page_count,
                        "page_range_in_merged": f"{len(merged_doc) - page_count + 1}-{len(merged_doc)}"
                    })

                    source_doc.close()

                except Exception as e:
                    logger.warning(f"Failed to merge {pdf_path}: {e}")
                    merge_info.append({
                        "source_file": str(pdf_path),
                        "error": str(e),
                        "pages_added": 0
                    })

            # Save merged document
            merged_doc.save(str(output_file))
            total_pages = len(merged_doc)
            merged_doc.close()

            return {
                "success": True,
                "output_path": str(output_file),
                "total_pages": total_pages,
                "files_merged": len(validated_paths),
                "merge_details": merge_info,
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
        description="Split PDF into multiple files at specified pages"
    )
    async def split_pdf(
        self,
        pdf_path: str,
        split_points: str,  # Page numbers where to split (comma-separated like "2,5,8")
        output_prefix: str = "split_part"
    ) -> Dict[str, Any]:
        """
        Split PDF into multiple files at specified pages.

        Args:
            pdf_path: Path to PDF file or URL
            split_points: Page numbers where to split (comma-separated like "2,5,8")
            output_prefix: Prefix for output files

        Returns:
            Dictionary containing split results
        """
        start_time = time.time()

        try:
            # Validate inputs
            path = await validate_pdf_path(pdf_path)
            doc = fitz.open(str(path))

            # Parse split points (convert from 1-based user input to 0-based internal)
            if isinstance(split_points, str):
                try:
                    if ',' in split_points:
                        user_split_list = [int(p.strip()) for p in split_points.split(',')]
                    else:
                        user_split_list = [int(split_points.strip())]
                    # Convert to 0-based for internal processing
                    split_list = [p - 1 for p in user_split_list]
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid split points format: {split_points}",
                        "split_time": 0
                    }
            else:
                split_list = split_points

            # Validate split points
            total_pages = len(doc)
            for split_point in split_list:
                if split_point < 0 or split_point >= total_pages:
                    return {
                        "success": False,
                        "error": f"Split point {split_point + 1} is out of range (1-{total_pages})",
                        "split_time": 0
                    }

            # Add document boundaries
            split_boundaries = [0] + sorted(split_list) + [total_pages]
            split_boundaries = list(set(split_boundaries))  # Remove duplicates
            split_boundaries.sort()

            created_files = []

            # Create split files
            for i in range(len(split_boundaries) - 1):
                start_page = split_boundaries[i]
                end_page = split_boundaries[i + 1]

                if start_page >= end_page:
                    continue

                # Create new document for this split
                split_doc = fitz.open()
                split_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)

                # Generate output filename
                output_filename = f"{output_prefix}_{i + 1}_pages_{start_page + 1}-{end_page}.pdf"
                output_path = validate_output_path(output_filename)

                split_doc.save(str(output_path))
                split_doc.close()

                created_files.append({
                    "filename": output_filename,
                    "path": str(output_path),
                    "page_range": f"{start_page + 1}-{end_page}",
                    "page_count": end_page - start_page
                })

            doc.close()

            return {
                "success": True,
                "original_file": str(path),
                "total_pages": total_pages,
                "files_created": len(created_files),
                "split_files": created_files,
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
        input_path: str,
        output_path: str,
        page_order: str  # JSON array of page numbers in desired order (1-indexed)
    ) -> Dict[str, Any]:
        """
        Reorder pages in a PDF document according to specified sequence.

        Args:
            input_path: Path to the PDF file to reorder
            output_path: Path where reordered PDF should be saved
            page_order: JSON array of page numbers in desired order (1-indexed)

        Returns:
            Dictionary containing reorder results
        """
        start_time = time.time()

        try:
            # Parse page order
            try:
                order = self._safe_json_parse(page_order) if page_order else []
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid page order JSON: {str(e)}",
                    "reorder_time": 0
                }

            if not order:
                return {
                    "success": False,
                    "error": "Page order array is required",
                    "reorder_time": 0
                }

            # Validate paths
            input_file = await validate_pdf_path(input_path)
            output_file = validate_output_path(output_path)

            source_doc = fitz.open(str(input_file))
            total_pages = len(source_doc)

            # Validate page numbers (convert from 1-based to 0-based)
            validated_order = []
            for page_num in order:
                if not isinstance(page_num, int):
                    return {
                        "success": False,
                        "error": f"Page number must be integer, got: {page_num}",
                        "reorder_time": 0
                    }
                if page_num < 1 or page_num > total_pages:
                    return {
                        "success": False,
                        "error": f"Page number {page_num} is out of range (1-{total_pages})",
                        "reorder_time": 0
                    }
                validated_order.append(page_num - 1)  # Convert to 0-based

            # Create reordered document
            reordered_doc = fitz.open()

            for page_num in validated_order:
                reordered_doc.insert_pdf(source_doc, from_page=page_num, to_page=page_num)

            # Save reordered document
            reordered_doc.save(str(output_file))
            reordered_doc.close()
            source_doc.close()

            return {
                "success": True,
                "input_path": str(input_file),
                "output_path": str(output_file),
                "original_pages": total_pages,
                "reordered_pages": len(validated_order),
                "page_mapping": [{"original": orig + 1, "new_position": i + 1} for i, orig in enumerate(validated_order)],
                "reorder_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"PDF reorder failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "reorder_time": round(time.time() - start_time, 2)
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