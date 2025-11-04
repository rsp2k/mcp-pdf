"""
MCP PDF Tools Server - Modular architecture using MCPMixin pattern

This is a refactored version demonstrating how to organize a large FastMCP server
using the MCPMixin pattern for better maintainability and modularity.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastmcp import FastMCP
from pydantic import BaseModel

# Import all mixins
from .mixins import (
    TextExtractionMixin,
    TableExtractionMixin,
    DocumentAnalysisMixin,
    ImageProcessingMixin,
    FormManagementMixin,
    DocumentAssemblyMixin,
    AnnotationsMixin,
    AdvancedFormsMixin
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security Configuration
MAX_PDF_SIZE = 100 * 1024 * 1024  # 100MB
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_PAGES_PROCESS = 1000
MAX_JSON_SIZE = 10000  # 10KB for JSON parameters
PROCESSING_TIMEOUT = 300  # 5 minutes

# Initialize FastMCP server
mcp = FastMCP("pdf-tools")

# Cache directory with secure permissions
CACHE_DIR = Path(os.environ.get("PDF_TEMP_DIR", "/tmp/mcp-pdf-processing"))
CACHE_DIR.mkdir(exist_ok=True, parents=True, mode=0o700)


class PDFToolsServer:
    """
    Main PDF tools server using modular MCPMixin architecture.

    Features:
    - Modular design with focused mixins
    - Auto-registration of tools from mixins
    - Progressive disclosure based on permissions
    - Centralized configuration and security
    """

    def __init__(self):
        self.mcp = mcp
        self.mixins: List[Any] = []
        self.config = self._load_configuration()

        # Show package version in startup banner
        try:
            from importlib.metadata import version
            package_version = version("mcp-pdf")
        except:
            package_version = "1.1.2"

        logger.info(f"🎬 MCP PDF Tools Server v{package_version}")
        logger.info("📊 Initializing modular architecture with MCPMixin pattern")

        # Initialize all mixins
        self._initialize_mixins()

        # Register server-level tools and resources
        self._register_server_tools()

        logger.info(f"✅ Server initialized with {len(self.mixins)} mixins")
        self._log_registration_summary()

    def _load_configuration(self) -> Dict[str, Any]:
        """Load server configuration from environment and defaults"""
        return {
            "max_pdf_size": int(os.getenv("MAX_PDF_SIZE", MAX_PDF_SIZE)),
            "max_image_size": int(os.getenv("MAX_IMAGE_SIZE", MAX_IMAGE_SIZE)),
            "max_pages": int(os.getenv("MAX_PAGES_PROCESS", MAX_PAGES_PROCESS)),
            "processing_timeout": int(os.getenv("PROCESSING_TIMEOUT", PROCESSING_TIMEOUT)),
            "cache_dir": CACHE_DIR,
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "allowed_domains": os.getenv("ALLOWED_DOMAINS", "").split(",") if os.getenv("ALLOWED_DOMAINS") else [],
        }

    def _initialize_mixins(self):
        """Initialize all PDF processing mixins"""
        mixin_classes = [
            TextExtractionMixin,
            TableExtractionMixin,
            DocumentAnalysisMixin,
            ImageProcessingMixin,
            FormManagementMixin,
            DocumentAssemblyMixin,
            AnnotationsMixin,
            AdvancedFormsMixin,
        ]

        for mixin_class in mixin_classes:
            try:
                mixin = mixin_class(self.mcp, **self.config)
                self.mixins.append(mixin)
                logger.info(f"✓ Initialized {mixin.get_mixin_name()} mixin")
            except Exception as e:
                logger.error(f"✗ Failed to initialize {mixin_class.__name__}: {e}")

    def _register_server_tools(self):
        """Register server-level management tools"""

        @self.mcp.tool(
            name="get_server_info",
            description="Get comprehensive server information and available capabilities"
        )
        async def get_server_info() -> Dict[str, Any]:
            """Return detailed server information including all available mixins and tools"""
            mixin_info = []
            total_tools = 0

            for mixin in self.mixins:
                components = mixin.get_registered_components()
                mixin_info.append(components)
                total_tools += len(components.get("tools", []))

            return {
                "server_name": "MCP PDF Tools",
                "version": "1.5.0",
                "architecture": "MCPMixin Modular",
                "total_mixins": len(self.mixins),
                "total_tools": total_tools,
                "mixins": mixin_info,
                "configuration": {
                    "max_pdf_size_mb": self.config["max_pdf_size"] // (1024 * 1024),
                    "max_pages": self.config["max_pages"],
                    "cache_directory": str(self.config["cache_dir"]),
                    "debug_mode": self.config["debug"]
                },
                "security_features": [
                    "Input validation and sanitization",
                    "File size and page count limits",
                    "Path traversal protection",
                    "Secure temporary file handling",
                    "Error message sanitization"
                ]
            }

        @self.mcp.tool(
            name="list_tools_by_category",
            description="List all available tools organized by functional category"
        )
        async def list_tools_by_category() -> Dict[str, Any]:
            """Return tools organized by their functional categories"""
            categories = {}

            for mixin in self.mixins:
                components = mixin.get_registered_components()
                category = components["mixin"]
                categories[category] = {
                    "tools": components["tools"],
                    "tool_count": len(components["tools"]),
                    "permissions_required": components["permissions_required"],
                    "description": self._get_category_description(category)
                }

            return {
                "categories": categories,
                "total_categories": len(categories),
                "usage_hint": "Each category provides specialized PDF processing capabilities"
            }

        @self.mcp.tool(
            name="validate_pdf_compatibility",
            description="Check PDF compatibility and recommend optimal processing methods"
        )
        async def validate_pdf_compatibility(pdf_path: str) -> Dict[str, Any]:
            """Analyze PDF and recommend optimal tools and methods"""
            try:
                from .security import validate_pdf_path
                validated_path = await validate_pdf_path(pdf_path)

                # Use text extraction mixin to analyze the PDF
                text_mixin = next((m for m in self.mixins if m.get_mixin_name() == "TextExtraction"), None)
                if text_mixin:
                    scan_result = await text_mixin.is_scanned_pdf(pdf_path)
                    is_scanned = scan_result.get("is_scanned", False)
                else:
                    is_scanned = False

                recommendations = []
                if is_scanned:
                    recommendations.extend([
                        "Use 'ocr_pdf' for text extraction",
                        "Consider 'extract_images' if document contains diagrams",
                        "OCR processing may take longer but provides better text extraction"
                    ])
                else:
                    recommendations.extend([
                        "Use 'extract_text' for fast text extraction",
                        "Use 'extract_tables' if document contains tabular data",
                        "Consider 'pdf_to_markdown' for structured content conversion"
                    ])

                return {
                    "success": True,
                    "pdf_path": str(validated_path),
                    "is_scanned": is_scanned,
                    "file_exists": validated_path.exists(),
                    "file_size_mb": round(validated_path.stat().st_size / (1024 * 1024), 2) if validated_path.exists() else 0,
                    "recommendations": recommendations,
                    "optimal_tools": self._get_optimal_tools(is_scanned)
                }

            except Exception as e:
                from .security import sanitize_error_message
                return {
                    "success": False,
                    "error": sanitize_error_message(str(e))
                }

    def _get_category_description(self, category: str) -> str:
        """Get description for tool category"""
        descriptions = {
            "TextExtraction": "Extract text content and perform OCR on scanned documents",
            "TableExtraction": "Extract and parse tabular data from PDFs",
            "DocumentAnalysis": "Analyze document structure, metadata, and quality",
            "ImageProcessing": "Extract images and convert PDFs to other formats",
            "FormManagement": "Create, fill, and manage PDF forms and interactive fields",
            "DocumentAssembly": "Merge, split, and reorganize PDF documents",
            "Annotations": "Add annotations, comments, and multimedia content to PDFs"
        }
        return descriptions.get(category, f"{category} tools")

    def _get_optimal_tools(self, is_scanned: bool) -> List[str]:
        """Get recommended tools based on PDF characteristics"""
        if is_scanned:
            return ["ocr_pdf", "extract_images", "get_document_structure"]
        else:
            return ["extract_text", "extract_tables", "pdf_to_markdown", "extract_metadata"]

    def _log_registration_summary(self):
        """Log summary of registered components"""
        total_tools = sum(len(mixin.get_registered_components()["tools"]) for mixin in self.mixins)
        logger.info(f"📋 Registration Summary:")
        logger.info(f"   • {len(self.mixins)} mixins loaded")
        logger.info(f"   • {total_tools} tools registered")
        logger.info(f"   • Server management tools: 3")

        if self.config["debug"]:
            for mixin in self.mixins:
                components = mixin.get_registered_components()
                logger.debug(f"   {components['mixin']}: {len(components['tools'])} tools")


# Create global server instance
server = PDFToolsServer()


def main():
    """Main entry point for the MCP PDF server"""
    try:
        logger.info("🚀 Starting MCP PDF Tools Server with modular architecture")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("📴 Server shutdown requested")
    except Exception as e:
        logger.error(f"💥 Server error: {e}")
        raise


if __name__ == "__main__":
    main()