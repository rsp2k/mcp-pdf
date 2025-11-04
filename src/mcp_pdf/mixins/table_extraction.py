"""
Table Extraction Mixin - PDF table detection and extraction capabilities
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# PDF processing libraries
import camelot
import tabula
import pdfplumber
import pandas as pd

from .base import MCPMixin, mcp_tool
from ..security import validate_pdf_path, parse_pages_parameter, sanitize_error_message

logger = logging.getLogger(__name__)


class TableExtractionMixin(MCPMixin):
    """
    Handles all PDF table extraction operations with intelligent fallbacks.

    Tools provided:
    - extract_tables: Multi-method table extraction with automatic fallbacks
    """

    def get_mixin_name(self) -> str:
        return "TableExtraction"

    def get_required_permissions(self) -> List[str]:
        return ["read_files", "table_processing"]

    def _setup(self):
        """Initialize table extraction specific configuration"""
        self.table_accuracy_threshold = 0.8
        self.max_tables_per_page = 10

    @mcp_tool(
        name="extract_tables",
        description="Extract tables from PDF with automatic method selection and intelligent fallbacks"
    )
    async def extract_tables(
        self,
        pdf_path: str,
        pages: Optional[str] = None,
        method: str = "auto",
        table_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Extract tables from PDF using various methods with automatic fallbacks.

        Args:
            pdf_path: Path to PDF file or URL
            pages: Page specification (e.g., "1-5,10,15-20" or "all")
            method: Extraction method ("auto", "camelot", "tabula", "pdfplumber")
            table_format: Output format ("json", "csv", "markdown")

        Returns:
            Dictionary containing extracted tables and metadata
        """
        start_time = time.time()

        try:
            # Validate inputs using centralized security functions
            path = await validate_pdf_path(pdf_path)
            parsed_pages = parse_pages_parameter(pages)

            all_tables = []
            methods_tried = []

            # Auto method: try methods in order until we find tables
            if method == "auto":
                for try_method in ["camelot", "pdfplumber", "tabula"]:
                    methods_tried.append(try_method)

                    if try_method == "camelot":
                        tables = self._extract_tables_camelot(path, parsed_pages)
                    elif try_method == "pdfplumber":
                        tables = self._extract_tables_pdfplumber(path, parsed_pages)
                    elif try_method == "tabula":
                        tables = self._extract_tables_tabula(path, parsed_pages)

                    if tables:
                        method = try_method
                        all_tables = tables
                        break
            else:
                # Use specific method
                methods_tried.append(method)
                if method == "camelot":
                    all_tables = self._extract_tables_camelot(path, parsed_pages)
                elif method == "pdfplumber":
                    all_tables = self._extract_tables_pdfplumber(path, parsed_pages)
                elif method == "tabula":
                    all_tables = self._extract_tables_tabula(path, parsed_pages)
                else:
                    raise ValueError(f"Unknown table extraction method: {method}")

            # Format tables based on output format
            formatted_tables = []
            for i, df in enumerate(all_tables):
                if table_format == "json":
                    formatted_tables.append({
                        "table_index": i,
                        "data": df.to_dict(orient="records"),
                        "shape": {"rows": len(df), "columns": len(df.columns)}
                    })
                elif table_format == "csv":
                    formatted_tables.append({
                        "table_index": i,
                        "data": df.to_csv(index=False),
                        "shape": {"rows": len(df), "columns": len(df.columns)}
                    })
                elif table_format == "markdown":
                    formatted_tables.append({
                        "table_index": i,
                        "data": df.to_markdown(index=False),
                        "shape": {"rows": len(df), "columns": len(df.columns)}
                    })

            return {
                "success": True,
                "tables": formatted_tables,
                "total_tables": len(formatted_tables),
                "method_used": method,
                "methods_tried": methods_tried,
                "pages_searched": pages or "all",
                "processing_time": round(time.time() - start_time, 2)
            }

        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            logger.error(f"Table extraction failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "methods_tried": methods_tried,
                "processing_time": round(time.time() - start_time, 2)
            }

    # Private helper methods (all synchronous for proper async pattern)
    def _extract_tables_camelot(self, pdf_path: Path, pages: Optional[List[int]] = None) -> List[pd.DataFrame]:
        """Extract tables using Camelot"""
        page_str = ','.join(map(str, [p+1 for p in pages])) if pages else 'all'

        # Try lattice mode first (for bordered tables)
        try:
            tables = camelot.read_pdf(str(pdf_path), pages=page_str, flavor='lattice')
            if len(tables) > 0:
                return [table.df for table in tables]
        except Exception:
            pass

        # Fall back to stream mode (for borderless tables)
        try:
            tables = camelot.read_pdf(str(pdf_path), pages=page_str, flavor='stream')
            return [table.df for table in tables]
        except Exception:
            return []

    def _extract_tables_tabula(self, pdf_path: Path, pages: Optional[List[int]] = None) -> List[pd.DataFrame]:
        """Extract tables using Tabula"""
        page_list = [p+1 for p in pages] if pages else 'all'

        try:
            tables = tabula.read_pdf(str(pdf_path), pages=page_list, multiple_tables=True)
            return tables
        except Exception:
            return []

    def _extract_tables_pdfplumber(self, pdf_path: Path, pages: Optional[List[int]] = None) -> List[pd.DataFrame]:
        """Extract tables using pdfplumber"""
        tables = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            page_range = pages if pages else range(len(pdf.pages))
            for page_num in page_range:
                page = pdf.pages[page_num]
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and len(table) > 1:  # Skip empty tables
                        df = pd.DataFrame(table[1:], columns=table[0])
                        tables.append(df)

        return tables