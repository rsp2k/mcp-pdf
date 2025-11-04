"""
MCP PDF Tools Server - Official FastMCP Mixin Pattern
Using fastmcp.contrib.mcp_mixin for proper modular architecture
"""

import os
import logging
from typing import Dict, Any
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.contrib.mcp_mixin import MCPMixin

# Import our mixins using the official pattern
from .mixins_official.text_extraction import TextExtractionMixin
from .mixins_official.table_extraction import TableExtractionMixin
from .mixins_official.document_analysis import DocumentAnalysisMixin
from .mixins_official.form_management import FormManagementMixin
from .mixins_official.document_assembly import DocumentAssemblyMixin
from .mixins_official.annotations import AnnotationsMixin
from .mixins_official.image_processing import ImageProcessingMixin
from .mixins_official.advanced_forms import AdvancedFormsMixin
from .mixins_official.security_analysis import SecurityAnalysisMixin
from .mixins_official.content_analysis import ContentAnalysisMixin
from .mixins_official.pdf_utilities import PDFUtilitiesMixin
from .mixins_official.misc_tools import MiscToolsMixin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFServerOfficial:
    """
    PDF Tools Server using official FastMCP mixin pattern.

    This server demonstrates the proper way to use fastmcp.contrib.mcp_mixin
    for creating modular, extensible MCP servers.
    """

    def __init__(self):
        self.mcp = FastMCP("pdf-tools")
        self.mixins = []
        self.config = self._load_configuration()

        logger.info("🎬 MCP PDF Tools Server (Official Pattern)")
        logger.info("📊 Initializing with official fastmcp.contrib.mcp_mixin pattern")

        # Initialize and register all mixins
        self._initialize_mixins()

        # Register server-level tools
        self._register_server_tools()

        logger.info(f"✅ Server initialized with {len(self.mixins)} mixins")
        self._log_registration_summary()

    def _load_configuration(self) -> Dict[str, Any]:
        """Load server configuration from environment and defaults"""
        return {
            "max_pdf_size": int(os.getenv("MAX_PDF_SIZE", str(100 * 1024 * 1024))),  # 100MB default
            "cache_dir": Path(os.getenv("PDF_TEMP_DIR", "/tmp/mcp-pdf-processing")),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "allowed_domains": os.getenv("ALLOWED_DOMAINS", "").split(",") if os.getenv("ALLOWED_DOMAINS") else [],
        }

    def _initialize_mixins(self):
        """Initialize all PDF processing mixins using official pattern"""
        mixin_classes = [
            TextExtractionMixin,
            TableExtractionMixin,
            DocumentAnalysisMixin,
            FormManagementMixin,
            DocumentAssemblyMixin,
            AnnotationsMixin,
            ImageProcessingMixin,
            AdvancedFormsMixin,
            SecurityAnalysisMixin,
            ContentAnalysisMixin,
            PDFUtilitiesMixin,
            MiscToolsMixin,
        ]

        for mixin_class in mixin_classes:
            try:
                # Create mixin instance
                mixin = mixin_class()

                # Register all decorated methods with the FastMCP server
                # Use class name as prefix to avoid naming conflicts
                prefix = mixin_class.__name__.replace("Mixin", "").lower()
                mixin.register_all(self.mcp, prefix=f"{prefix}_")

                self.mixins.append(mixin)
                logger.info(f"✓ Initialized and registered {mixin_class.__name__}")

            except Exception as e:
                logger.error(f"✗ Failed to initialize {mixin_class.__name__}: {e}")

    def _register_server_tools(self):
        """Register server-level management tools"""

        @self.mcp.tool(name="server_info", description="Get comprehensive server information")
        async def get_server_info() -> Dict[str, Any]:
            """Get detailed server information including mixins and configuration"""
            return {
                "server_name": "MCP PDF Tools (Official FastMCP Pattern)",
                "version": "2.0.5",
                "architecture": "Official FastMCP Mixin Pattern",
                "total_mixins": len(self.mixins),
                "mixins": [
                    {
                        "name": mixin.__class__.__name__,
                        "description": mixin.__class__.__doc__.split('\n')[1].strip() if mixin.__class__.__doc__ else "No description"
                    }
                    for mixin in self.mixins
                ],
                "configuration": {
                    "max_pdf_size_mb": self.config["max_pdf_size"] // (1024 * 1024),
                    "cache_directory": str(self.config["cache_dir"]),
                    "debug_mode": self.config["debug"]
                }
            }

        @self.mcp.tool(name="list_capabilities", description="List all available PDF processing capabilities")
        async def list_capabilities() -> Dict[str, Any]:
            """List all available tools and their capabilities"""
            return {
                "architecture": "Official FastMCP Mixin Pattern",
                "mixins_loaded": len(self.mixins),
                "capabilities": {
                    "text_extraction": ["extract_text", "ocr_pdf", "is_scanned_pdf"],
                    "table_extraction": ["extract_tables"],
                    "document_analysis": ["extract_metadata", "get_document_structure", "analyze_pdf_health"],
                    "form_management": ["extract_form_data", "fill_form_pdf", "create_form_pdf"],
                    "document_assembly": ["merge_pdfs", "split_pdf", "reorder_pdf_pages"],
                    "annotations": ["add_sticky_notes", "add_highlights", "add_stamps", "extract_all_annotations"],
                    "image_processing": ["extract_images", "pdf_to_markdown"]
                }
            }

    def _log_registration_summary(self):
        """Log a summary of what was registered"""
        logger.info("📋 Registration Summary:")
        logger.info(f"   • {len(self.mixins)} mixins loaded")
        logger.info(f"   • Tools registered via mixin pattern")
        logger.info(f"   • Server management tools: 2")


def create_server() -> PDFServerOfficial:
    """Factory function to create the PDF server instance"""
    return PDFServerOfficial()


def main():
    """Main entry point for the MCP server"""
    try:
        # Get package version
        try:
            from importlib.metadata import version
            package_version = version("mcp-pdf")
        except:
            package_version = "2.0.5"

        logger.info(f"🎬 MCP PDF Tools Server v{package_version} (Official Pattern)")

        # Create and run the server
        server = create_server()
        server.mcp.run()

    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise


if __name__ == "__main__":
    main()