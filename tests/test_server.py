"""Test suite for MCP PDF Tools server"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import base64
import pandas as pd
from pathlib import Path

from mcp_pdf.server import (
    create_server,
    validate_pdf_path,
    detect_scanned_pdf,
    extract_text,
    extract_tables,
    ocr_pdf,
    is_scanned_pdf,
    get_document_structure,
    extract_metadata,
    pdf_to_markdown,
    extract_images
)


@pytest.fixture
def server():
    """Create server instance for testing"""
    return create_server()


@pytest.fixture
def mock_pdf_path(tmp_path):
    """Create a mock PDF file path"""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.touch()
    return str(pdf_file)


@pytest.fixture
def mock_fitz_doc():
    """Create a mock PyMuPDF document"""
    doc = MagicMock()
    doc.__len__.return_value = 3
    doc.metadata = {
        "title": "Test PDF",
        "author": "Test Author",
        "subject": "Testing",
        "keywords": "test, pdf",
        "creator": "Test Creator",
        "producer": "Test Producer",
        "creationDate": "2024-01-01",
        "modDate": "2024-01-02"
    }
    doc.is_encrypted = False
    doc.is_form_pdf = False
    doc.get_toc.return_value = [(1, "Chapter 1", 1), (2, "Section 1.1", 2)]
    
    # Mock pages
    pages = []
    for i in range(3):
        page = MagicMock()
        page.get_text.return_value = f"This is page {i+1} text content."
        page.rect.width = 595
        page.rect.height = 842
        page.rotation = 0
        page.get_images.return_value = []
        page.get_links.return_value = []
        page.get_annotations.return_value = []
        page.get_fonts.return_value = [(0, 0, 0, "Arial"), (0, 0, 0, "Times")]
        pages.append(page)
    
    doc.__getitem__.side_effect = lambda i: pages[i]
    doc.pages = pages
    
    return doc


class TestValidation:
    """Test validation functions"""
    
    @pytest.mark.asyncio
    async def test_validate_pdf_path_valid(self, mock_pdf_path):
        """Test validation with valid PDF path"""
        result = await validate_pdf_path(mock_pdf_path)
        assert result.exists()
        assert result.suffix == ".pdf"
    
    @pytest.mark.asyncio
    async def test_validate_pdf_path_not_exists(self):
        """Test validation with non-existent file"""
        with pytest.raises(ValueError, match="File not found"):
            await validate_pdf_path("/non/existent/file.pdf")
    
    @pytest.mark.asyncio
    async def test_validate_pdf_path_not_pdf(self, tmp_path):
        """Test validation with non-PDF file"""
        txt_file = tmp_path / "test.txt"
        txt_file.touch()
        with pytest.raises(ValueError, match="Not a PDF file"):
            await validate_pdf_path(str(txt_file))


class TestTextExtraction:
    """Test text extraction functionality"""
    
    @pytest.mark.asyncio
    @patch('fitz.open')
    async def test_extract_text_success(self, mock_fitz_open, mock_fitz_doc, mock_pdf_path):
        """Test successful text extraction"""
        mock_fitz_open.return_value = mock_fitz_doc
        
        result = await extract_text(
            pdf_path=mock_pdf_path,
            method="pymupdf"
        )
        
        assert result["text"] == "This is page 1 text content.\n\nThis is page 2 text content.\n\nThis is page 3 text content."
        assert result["method_used"] == "pymupdf"
        assert result["metadata"]["pages"] == 3
        assert result["metadata"]["title"] == "Test PDF"
        assert len(result["pages_extracted"]) == 3
    
    @pytest.mark.asyncio
    @patch('fitz.open')
    async def test_extract_text_specific_pages(self, mock_fitz_open, mock_fitz_doc, mock_pdf_path):
        """Test text extraction from specific pages"""
        mock_fitz_open.return_value = mock_fitz_doc
        
        result = await extract_text(
            pdf_path=mock_pdf_path,
            pages=[0, 2],
            method="pymupdf"
        )
        
        assert "page 1" in result["text"]
        assert "page 2" not in result["text"]
        assert "page 3" in result["text"]
        assert result["pages_extracted"] == [0, 2]


class TestTableExtraction:
    """Test table extraction functionality"""
    
    @pytest.mark.asyncio
    @patch('camelot.read_pdf')
    async def test_extract_tables_camelot(self, mock_camelot, mock_pdf_path):
        """Test table extraction with Camelot"""
        # Mock Camelot tables
        mock_table = MagicMock()
        mock_table.df = pd.DataFrame({
            'Column1': ['A', 'B'],
            'Column2': ['1', '2']
        })
        mock_camelot.return_value = [mock_table]
        
        result = await extract_tables(
            pdf_path=mock_pdf_path,
            method="camelot",
            output_format="json"
        )
        
        assert result["total_tables"] == 1
        assert result["method_used"] == "camelot"
        assert len(result["tables"]) == 1
        assert result["tables"][0]["shape"]["rows"] == 2
        assert result["tables"][0]["shape"]["columns"] == 2
    
    @pytest.mark.asyncio
    @patch('camelot.read_pdf')
    @patch('pdfplumber.open')
    @patch('tabula.read_pdf')
    async def test_extract_tables_auto_fallback(self, mock_tabula, mock_pdfplumber, mock_camelot, mock_pdf_path):
        """Test automatic fallback between table extraction methods"""
        # Camelot fails
        mock_camelot.side_effect = Exception("Camelot failed")
        
        # pdfplumber succeeds
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_tables.return_value = [[['Col1', 'Col2'], ['A', '1'], ['B', '2']]]
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__.return_value = mock_pdf
        mock_pdfplumber.return_value = mock_pdf
        
        result = await extract_tables(
            pdf_path=mock_pdf_path,
            method="auto"
        )
        
        assert result["total_tables"] == 1
        assert result["method_used"] == "pdfplumber"
        assert "camelot" in result["methods_tried"]
        assert "pdfplumber" in result["methods_tried"]


class TestDocumentAnalysis:
    """Test document analysis functions"""
    
    @pytest.mark.asyncio
    @patch('fitz.open')
    @patch('pdfplumber.open')
    async def test_is_scanned_pdf_true(self, mock_pdfplumber, mock_fitz, mock_pdf_path):
        """Test detection of scanned PDF"""
        # Mock pdfplumber for scanned detection
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""  # No text = scanned
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__.return_value = mock_pdf
        mock_pdfplumber.return_value = mock_pdf
        
        # Mock fitz for additional info
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value.get_text.return_value = ""
        mock_fitz.return_value = mock_doc
        
        result = await is_scanned_pdf(mock_pdf_path)
        
        assert result["is_scanned"] is True
        assert result["recommendation"] == "Use OCR tool"
    
    @pytest.mark.asyncio
    @patch('fitz.open')
    async def test_get_document_structure(self, mock_fitz_open, mock_fitz_doc, mock_pdf_path):
        """Test document structure extraction"""
        mock_fitz_open.return_value = mock_fitz_doc
        
        result = await get_document_structure(mock_pdf_path)
        
        assert result["metadata"]["title"] == "Test PDF"
        assert result["pages"] == 3
        assert len(result["outline"]) == 2
        assert result["outline"][0]["title"] == "Chapter 1"
        assert len(result["sample_pages"]) == 3
        assert "Arial" in result["fonts"]
        assert "Times" in result["fonts"]

    @pytest.mark.asyncio
    @patch('fitz.open')
    @patch('pypdf.PdfReader')
    async def test_extract_metadata(self, mock_pypdf, mock_fitz_open, mock_fitz_doc, mock_pdf_path):
        """Test comprehensive metadata extraction"""
        mock_fitz_open.return_value = mock_fitz_doc
        
        # Mock pypdf for additional metadata
        mock_reader = MagicMock()
        mock_reader.metadata = {
            "/CustomField": "Custom Value"
        }
        mock_pypdf.return_value = mock_reader
        
        # Mock file stats
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value = MagicMock(
                st_size=1024000,  # 1MB
                st_ctime=1704067200,  # 2024-01-01
                st_mtime=1704153600   # 2024-01-02
            )
            
            result = await extract_metadata(mock_pdf_path)
            
            assert result["metadata"]["title"] == "Test PDF"
            assert result["file_info"]["size_mb"] == 1.0
            assert result["statistics"]["page_count"] == 3
            assert result["statistics"]["is_encrypted"] is False
            assert result["additional_metadata"]["CustomField"] == "Custom Value"


class TestConversion:
    """Test PDF conversion functions"""
    
    @pytest.mark.asyncio
    @patch('fitz.open')
    async def test_pdf_to_markdown(self, mock_fitz_open, mock_fitz_doc, mock_pdf_path):
        """Test PDF to Markdown conversion"""
        # Enhance mock for text blocks
        mock_page = mock_fitz_doc[0]
        mock_page.get_text.return_value = "Page 1 content"
        mock_page.get_text.side_effect = lambda fmt="": {
            "blocks": [(0, 0, 100, 20, "HEADER TEXT", 0, 0)],
            "": "Page 1 content"
        }.get(fmt, "Page 1 content")
        
        mock_fitz_open.return_value = mock_fitz_doc
        
        result = await pdf_to_markdown(
            pdf_path=mock_pdf_path,
            include_metadata=True
        )
        
        assert "# Document Metadata" in result["markdown"]
        assert "Test PDF" in result["markdown"]
        assert "# Table of Contents" in result["markdown"]
        assert "Chapter 1" in result["markdown"]
        assert result["pages_converted"] == 3


class TestImageExtraction:
    """Test image extraction functionality"""
    
    @pytest.mark.asyncio
    @patch('fitz.open')
    @patch('fitz.Pixmap')
    async def test_extract_images(self, mock_pixmap_class, mock_fitz_open, mock_pdf_path):
        """Test image extraction from PDF"""
        # Mock document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_images.return_value = [(1, 0, 100, 100, 8, 'DeviceRGB', '', 'Im1', 'FlateDecode')]
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc
        
        # Mock pixmap
        mock_pixmap = MagicMock()
        mock_pixmap.width = 200
        mock_pixmap.height = 200
        mock_pixmap.n = 3  # RGB
        mock_pixmap.alpha = 0
        mock_pixmap.tobytes.return_value = b"fake_image_data"
        mock_pixmap_class.return_value = mock_pixmap
        
        result = await extract_images(
            pdf_path=mock_pdf_path,
            min_width=100,
            min_height=100
        )
        
        assert result["total_images"] == 1
        assert len(result["images"]) == 1
        assert result["images"][0]["width"] == 200
        assert result["images"][0]["height"] == 200
        assert result["images"][0]["format"] == "png"
        assert result["images"][0]["data"] == base64.b64encode(b"fake_image_data").decode()


class TestServerInitialization:
    """Test server initialization and configuration"""
    
    def test_create_server(self):
        """Test server creation"""
        server = create_server()
        assert server is not None
    
    @pytest.mark.asyncio
    async def test_server_has_all_tools(self, server):
        """Test that all expected tools are registered"""
        # Get all registered tools
        tools = []
        for handler in server._tool_handlers:
            tools.append(handler.name)
        
        expected_tools = [
            "extract_text",
            "extract_tables", 
            "ocr_pdf",
            "is_scanned_pdf",
            "get_document_structure",
            "extract_metadata",
            "pdf_to_markdown",
            "extract_images"
        ]
        
        for tool in expected_tools:
            assert tool in tools, f"Tool '{tool}' not found in server"


class TestErrorHandling:
    """Test error handling in various scenarios"""
    
    @pytest.mark.asyncio
    async def test_extract_text_invalid_method(self, mock_pdf_path):
        """Test error handling for invalid extraction method"""
        result = await extract_text(
            pdf_path=mock_pdf_path,
            method="invalid_method"
        )
        
        assert "error" in result
        assert "Unknown extraction method" in result["error"]
    
    @pytest.mark.asyncio
    async def test_extract_text_file_not_found(self):
        """Test error handling for non-existent file"""
        result = await extract_text(
            pdf_path="/non/existent/file.pdf"
        )
        
        assert "error" in result
        assert "File not found" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
