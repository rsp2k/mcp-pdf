"""
Official FastMCP Mixins for PDF Tools

This package contains mixins that use the official fastmcp.contrib.mcp_mixin pattern
instead of our custom implementation.
"""

from .text_extraction import TextExtractionMixin
from .table_extraction import TableExtractionMixin
from .document_analysis import DocumentAnalysisMixin
from .form_management import FormManagementMixin
from .document_assembly import DocumentAssemblyMixin
from .annotations import AnnotationsMixin
from .image_processing import ImageProcessingMixin
from .advanced_forms import AdvancedFormsMixin
from .security_analysis import SecurityAnalysisMixin
from .content_analysis import ContentAnalysisMixin
from .pdf_utilities import PDFUtilitiesMixin
from .misc_tools import MiscToolsMixin

__all__ = [
    "TextExtractionMixin",
    "TableExtractionMixin",
    "DocumentAnalysisMixin",
    "FormManagementMixin",
    "DocumentAssemblyMixin",
    "AnnotationsMixin",
    "ImageProcessingMixin",
    "AdvancedFormsMixin",
    "SecurityAnalysisMixin",
    "ContentAnalysisMixin",
    "PDFUtilitiesMixin",
    "MiscToolsMixin",
]