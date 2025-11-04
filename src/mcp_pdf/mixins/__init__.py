"""
MCPMixin components for modular PDF tools organization
"""

from .base import MCPMixin
from .text_extraction import TextExtractionMixin
from .table_extraction import TableExtractionMixin
from .image_processing import ImageProcessingMixin
from .document_analysis import DocumentAnalysisMixin
from .form_management import FormManagementMixin
from .document_assembly import DocumentAssemblyMixin
from .annotations import AnnotationsMixin
from .advanced_forms import AdvancedFormsMixin

__all__ = [
    "MCPMixin",
    "TextExtractionMixin",
    "TableExtractionMixin",
    "DocumentAnalysisMixin",
    "ImageProcessingMixin",
    "FormManagementMixin",
    "DocumentAssemblyMixin",
    "AnnotationsMixin",
    "AdvancedFormsMixin",
]