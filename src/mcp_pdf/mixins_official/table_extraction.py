"""
Table Extraction Mixin - PDF table extraction with intelligent method selection
Uses official fastmcp.contrib.mcp_mixin pattern
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional
import logging

# Required
import pandas as pd
import pdfplumber

# Optional — camelot and tabula are heavy deps with C/Java requirements.
# They're imported lazily in their extraction methods.

# Official FastMCP mixin
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool

from ..security import validate_pdf_path, sanitize_error_message

logger = logging.getLogger(__name__)


class TableExtractionMixin(MCPMixin):
    """
    Handles PDF table extraction operations with intelligent method selection.
    Uses the official FastMCP mixin pattern.
    """

    def __init__(self):
        super().__init__()

    @mcp_tool(
        name="extract_tables",
        description=(
            "Pull tabular data out of a PDF as rows, trying several extractors "
            "until one finds something. Read-only: results come back in the "
            "response and no file is written, so cap the volume with "
            "max_rows_per_table or summary_only on a big document.\n"
            "\n"
            "READ THIS ABOUT FAILURE. When no extractor finds a table you get "
            "success=false with error \"No tables found or all extraction "
            "methods failed\", and that single message covers both cases. In "
            "practice it almost always means THE PDF HAS NO DETECTABLE TABLES, "
            "not that anything broke — a document of flowing prose returns it "
            "every time. Do not report it to the user as an error or retry it; "
            "read `methods_tried` to see which extractors actually ran, and "
            "note that an explicitly requested method that is not installed "
            "lands in this same branch (methods_tried will name just that one). "
            "The extractors are genuinely broken only when the process cannot "
            "reach them at all, and that surfaces as a different message.\n"
            "\n"
            "method=\"auto\" tries camelot (if installed), then pdfplumber "
            "(always available), then tabula (if installed), and stops at the "
            "first that returns at least one table; `method_used` names the "
            "winner. They see different things, so if auto finds nothing a "
            "specific method rarely does better — but the tradeoffs are: "
            "camelot only runs its 'lattice' mode, so it needs ruled cell "
            "borders and it needs Ghostscript, and in return it reports an "
            "`accuracy` score; pdfplumber handles whitespace-aligned tables "
            "with no borders and treats the first row as the header; tabula "
            "needs a Java runtime and cannot tell you which page a table came "
            "from, so its `page` is always null.\n"
            "\n"
            "`pages` is a 1-based string: \"5\", \"1,3,5\", \"1-10\", or mixed "
            "\"1,3-5,12-20\". Omit for the whole document. `table_format` "
            "shapes each table's `data`: \"json\" gives a list of row objects "
            "keyed by column name (the default and the easiest to reason "
            "about), \"csv\" one CSV string, \"html\" one HTML table string. It "
            "is IGNORED when summary_only=true, which drops `data` entirely and "
            "leaves only the shape of each table. max_rows_per_table truncates "
            "`data` while `total_rows` keeps the true count, and adds "
            "rows_returned / rows_truncated so you can tell."
        ),
        annotations={
            "readOnlyHint": True,        # returns rows in the response, writes no file
            "idempotentHint": True,
            "openWorldHint": True,       # pdf_path may be an http(s) URL
        },
    )
    async def extract_tables(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        method: Literal["auto", "camelot", "pdfplumber", "tabula"] = "auto",
        table_format: Literal["json", "csv", "html"] = "json",
        max_rows_per_table: Optional[int] = None,
        summary_only: bool = False
    ) -> Dict[str, Any]:
        """
        Extract tables from PDF using intelligent method selection.

        Args:
            pdf_path: Path to the PDF, or an http(s) URL to fetch.
            pages: 1-based page selection, e.g. "5", "1,3,5", "1-10",
                "1,3-5,12-20". None (the default) means every page.
            method: "auto" walks camelot -> pdfplumber -> tabula and keeps the
                first result. Name one explicitly only when you know the table
                style: camelot for ruled/bordered tables (needs Ghostscript),
                pdfplumber for borderless whitespace-aligned ones, tabula for
                awkward layouts (needs Java).
            table_format: Shape of each table's "data" — "json" for row
                objects, "csv" for one CSV string, "html" for one HTML table.
                Ignored when summary_only is True.
            max_rows_per_table: Truncate the returned rows per table.
                "total_rows" still reports the real count.
            summary_only: Omit "data" (and rows_returned/rows_truncated) and
                return only each table's page, row count and column count.

        Returns:
            On success: success=true, tables_found, method_used, and `tables` —
            one entry per table with table_index (1-based), page (1-based;
            null from tabula), total_rows, columns, accuracy (camelot only),
            and unless summary_only the data plus rows_returned and, when
            truncated, rows_truncated.

            When nothing is found: success=false with error "No tables found or
            all extraction methods failed" and methods_tried. As the
            description says, this usually means there are no tables to find
            rather than that extraction broke.
        """
        start_time = time.time()

        try:
            # Validate and prepare inputs
            path = await validate_pdf_path(pdf_path)
            parsed_pages = self._parse_pages_parameter(pages)

            if method == "auto":
                # Try methods in order of reliability, skip unavailable ones
                methods_to_try = []
                try:
                    import camelot  # noqa: F401
                    methods_to_try.append("camelot")
                except ImportError:
                    pass
                methods_to_try.append("pdfplumber")  # always available
                try:
                    import tabula  # noqa: F401
                    methods_to_try.append("tabula")
                except ImportError:
                    pass
            else:
                methods_to_try = [method]

            extraction_results = []
            method_used = None
            total_tables = 0

            for extraction_method in methods_to_try:
                try:
                    logger.info(f"Attempting table extraction with {extraction_method}")

                    if extraction_method == "camelot":
                        result = await self._extract_with_camelot(path, parsed_pages, table_format, max_rows_per_table, summary_only)
                    elif extraction_method == "pdfplumber":
                        result = await self._extract_with_pdfplumber(path, parsed_pages, table_format, max_rows_per_table, summary_only)
                    elif extraction_method == "tabula":
                        result = await self._extract_with_tabula(path, parsed_pages, table_format, max_rows_per_table, summary_only)
                    else:
                        continue

                    if result.get("tables") and len(result["tables"]) > 0:
                        extraction_results = result["tables"]
                        total_tables = len(extraction_results)
                        method_used = extraction_method
                        logger.info(f"Successfully extracted {total_tables} tables with {extraction_method}")
                        break

                except Exception as e:
                    logger.warning(f"Table extraction failed with {extraction_method}: {e}")
                    continue

            if not extraction_results:
                return {
                    "success": False,
                    "error": "No tables found or all extraction methods failed",
                    "methods_tried": methods_to_try,
                    "extraction_time": round(time.time() - start_time, 2)
                }

            return {
                "success": True,
                "tables_found": total_tables,
                "tables": extraction_results,
                "method_used": method_used,
                "file_info": {
                    "path": str(path),
                    "pages_processed": pages or "all"
                },
                "extraction_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Table extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "extraction_time": round(time.time() - start_time, 2)
            }

    # Helper methods (synchronous)
    def _process_table_data(self, df, table_format: str, max_rows: Optional[int], summary_only: bool) -> Any:
        """Process table data with row limiting and summary options"""
        if summary_only:
            # Return None for data when in summary mode
            return None

        # Apply row limit if specified
        if max_rows and len(df) > max_rows:
            df_limited = df.head(max_rows)
        else:
            df_limited = df

        # Convert to requested format
        if table_format == "json":
            return df_limited.to_dict('records')
        elif table_format == "csv":
            return df_limited.to_csv(index=False)
        elif table_format == "html":
            return df_limited.to_html(index=False)
        else:
            return df_limited.to_dict('records')

    def _parse_pages_parameter(self, pages: Optional[str]) -> Optional[str]:
        """Parse pages parameter for different extraction methods

        Converts user input (supporting ranges like "11-30") into library format
        """
        if not pages:
            return None

        try:
            # Use shared parser from utils to handle ranges
            from .utils import parse_pages_parameter
            parsed = parse_pages_parameter(pages)

            if parsed is None:
                return None

            # Convert 0-based indices back to 1-based for library format
            page_list = [p + 1 for p in parsed]
            return ','.join(map(str, page_list))
        except (ValueError, ImportError):
            return None

    async def _extract_with_camelot(self, path: Path, pages: Optional[str], table_format: str,
                                     max_rows: Optional[int], summary_only: bool) -> Dict[str, Any]:
        """Extract tables using Camelot (best for complex tables)"""
        import camelot

        pages_param = pages if pages else "all"

        # Run camelot in thread to avoid blocking
        def extract_camelot():
            return camelot.read_pdf(str(path), pages=pages_param, flavor='lattice')

        tables = await asyncio.get_event_loop().run_in_executor(None, extract_camelot)

        extracted_tables = []
        for i, table in enumerate(tables):
            # Process table data with limits
            table_data = self._process_table_data(table.df, table_format, max_rows, summary_only)

            table_info = {
                "table_index": i + 1,
                "page": table.page,
                "accuracy": round(table.accuracy, 2) if hasattr(table, 'accuracy') else None,
                "total_rows": len(table.df),
                "columns": len(table.df.columns),
            }

            # Only include data if not summary_only
            if not summary_only:
                table_info["data"] = table_data
                if max_rows and len(table.df) > max_rows:
                    table_info["rows_returned"] = max_rows
                    table_info["rows_truncated"] = len(table.df) - max_rows
                else:
                    table_info["rows_returned"] = len(table.df)

            extracted_tables.append(table_info)

        return {"tables": extracted_tables}

    async def _extract_with_pdfplumber(self, path: Path, pages: Optional[str], table_format: str,
                                        max_rows: Optional[int], summary_only: bool) -> Dict[str, Any]:
        """Extract tables using pdfplumber (good for simple tables)"""

        def extract_pdfplumber():
            extracted_tables = []
            with pdfplumber.open(str(path)) as pdf:
                pages_to_process = self._get_page_range(pdf, pages)

                for page_num in pages_to_process:
                    if page_num < len(pdf.pages):
                        page = pdf.pages[page_num]
                        tables = page.extract_tables()

                        for i, table in enumerate(tables):
                            if table and len(table) > 0:
                                # Convert to DataFrame for consistent formatting
                                df = pd.DataFrame(table[1:], columns=table[0])

                                # Process table data with limits
                                table_data = self._process_table_data(df, table_format, max_rows, summary_only)

                                table_info = {
                                    "table_index": len(extracted_tables) + 1,
                                    "page": page_num + 1,
                                    "total_rows": len(df),
                                    "columns": len(df.columns),
                                }

                                # Only include data if not summary_only
                                if not summary_only:
                                    table_info["data"] = table_data
                                    if max_rows and len(df) > max_rows:
                                        table_info["rows_returned"] = max_rows
                                        table_info["rows_truncated"] = len(df) - max_rows
                                    else:
                                        table_info["rows_returned"] = len(df)

                                extracted_tables.append(table_info)

            return {"tables": extracted_tables}

        return await asyncio.get_event_loop().run_in_executor(None, extract_pdfplumber)

    async def _extract_with_tabula(self, path: Path, pages: Optional[str], table_format: str,
                                    max_rows: Optional[int], summary_only: bool) -> Dict[str, Any]:
        """Extract tables using Tabula (Java-based, good for complex layouts)"""
        import tabula

        def extract_tabula():
            pages_param = pages if pages else "all"

            # Read tables with tabula
            tables = tabula.read_pdf(str(path), pages=pages_param, multiple_tables=True)

            extracted_tables = []
            for i, df in enumerate(tables):
                if not df.empty:
                    # Process table data with limits
                    table_data = self._process_table_data(df, table_format, max_rows, summary_only)

                    table_info = {
                        "table_index": i + 1,
                        "page": None,  # Tabula doesn't provide page info easily
                        "total_rows": len(df),
                        "columns": len(df.columns),
                    }

                    # Only include data if not summary_only
                    if not summary_only:
                        table_info["data"] = table_data
                        if max_rows and len(df) > max_rows:
                            table_info["rows_returned"] = max_rows
                            table_info["rows_truncated"] = len(df) - max_rows
                        else:
                            table_info["rows_returned"] = len(df)

                    extracted_tables.append(table_info)

            return {"tables": extracted_tables}

        return await asyncio.get_event_loop().run_in_executor(None, extract_tabula)

    def _get_page_range(self, pdf, pages: Optional[str]) -> List[int]:
        """Convert pages parameter to list of 0-based page indices"""
        if not pages:
            return list(range(len(pdf.pages)))

        try:
            if ',' in pages:
                return [int(p.strip()) - 1 for p in pages.split(',')]
            else:
                return [int(pages.strip()) - 1]
        except ValueError:
            return list(range(len(pdf.pages)))