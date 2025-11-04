"""
Test suite for MCPMixin architecture

Demonstrates how to test modular MCP servers with auto-discovery and validation.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import tempfile

from fastmcp import FastMCP
from mcp_pdf.mixins import (
    MCPMixin,
    TextExtractionMixin,
    TableExtractionMixin,
    DocumentAnalysisMixin,
    ImageProcessingMixin,
    FormManagementMixin,
    DocumentAssemblyMixin,
    AnnotationsMixin,
)


class TestMCPMixinArchitecture:
    """Test the MCPMixin base architecture and auto-registration"""

    def setup_method(self):
        """Setup test environment"""
        self.mcp = FastMCP("test-pdf-tools")
        self.test_pdf_path = "/tmp/test.pdf"

    def test_mixin_auto_registration(self):
        """Test that mixins auto-register their tools"""
        # Initialize a mixin
        text_mixin = TextExtractionMixin(self.mcp)

        # Check that tools were registered
        components = text_mixin.get_registered_components()
        assert components["mixin"] == "TextExtraction"
        assert len(components["tools"]) > 0
        assert "extract_text" in components["tools"]
        assert "ocr_pdf" in components["tools"]

    def test_mixin_permissions(self):
        """Test permission system"""
        text_mixin = TextExtractionMixin(self.mcp)
        permissions = text_mixin.get_required_permissions()

        assert "read_files" in permissions
        assert "ocr_processing" in permissions

    def test_all_mixins_initialize(self):
        """Test that all mixins can be initialized"""
        mixin_classes = [
            TextExtractionMixin,
            TableExtractionMixin,
            DocumentAnalysisMixin,
            ImageProcessingMixin,
            FormManagementMixin,
            DocumentAssemblyMixin,
            AnnotationsMixin,
        ]

        for mixin_class in mixin_classes:
            mixin = mixin_class(self.mcp)
            assert mixin.get_mixin_name()
            assert isinstance(mixin.get_required_permissions(), list)

    def test_mixin_tool_discovery(self):
        """Test automatic tool discovery from mixin methods"""
        text_mixin = TextExtractionMixin(self.mcp)

        # Check that public async methods are discovered
        components = text_mixin.get_registered_components()
        tools = components["tools"]

        # Should include methods marked with @mcp_tool
        expected_tools = ["extract_text", "ocr_pdf", "is_scanned_pdf"]
        for tool in expected_tools:
            assert tool in tools, f"Tool {tool} not found in registered tools: {tools}"


class TestTextExtractionMixin:
    """Test the TextExtractionMixin specifically"""

    def setup_method(self):
        """Setup test environment"""
        self.mcp = FastMCP("test-text-extraction")
        self.mixin = TextExtractionMixin(self.mcp)

    @pytest.mark.asyncio
    async def test_extract_text_validation(self):
        """Test input validation for extract_text"""
        # Test empty path
        result = await self.mixin.extract_text("")
        assert not result["success"]
        assert "cannot be empty" in result["error"]

        # Test invalid path
        result = await self.mixin.extract_text("/nonexistent/file.pdf")
        assert not result["success"]
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_is_scanned_pdf_validation(self):
        """Test input validation for is_scanned_pdf"""
        result = await self.mixin.is_scanned_pdf("")
        assert not result["success"]
        assert "cannot be empty" in result["error"]


class TestTableExtractionMixin:
    """Test the TableExtractionMixin specifically"""

    def setup_method(self):
        """Setup test environment"""
        self.mcp = FastMCP("test-table-extraction")
        self.mixin = TableExtractionMixin(self.mcp)

    @pytest.mark.asyncio
    async def test_extract_tables_fallback_logic(self):
        """Test fallback logic when multiple methods are attempted"""
        # This would test the actual fallback mechanism
        # For now, just test that the method exists and handles errors
        result = await self.mixin.extract_tables("/nonexistent/file.pdf")
        assert not result["success"]
        assert "fallback_attempts" in result or "error" in result


class TestMixinComposition:
    """Test how mixins work together in a composed server"""

    def setup_method(self):
        """Setup test environment"""
        self.mcp = FastMCP("test-composed-server")
        self.mixins = []

        # Initialize all mixins
        mixin_classes = [
            TextExtractionMixin,
            TableExtractionMixin,
            DocumentAnalysisMixin,
            ImageProcessingMixin,
            FormManagementMixin,
            DocumentAssemblyMixin,
            AnnotationsMixin,
        ]

        for mixin_class in mixin_classes:
            mixin = mixin_class(self.mcp)
            self.mixins.append(mixin)

    def test_no_tool_name_conflicts(self):
        """Test that mixins don't have conflicting tool names"""
        all_tools = set()
        conflicts = []

        for mixin in self.mixins:
            components = mixin.get_registered_components()
            tools = components["tools"]

            for tool in tools:
                if tool in all_tools:
                    conflicts.append(f"Tool '{tool}' registered by multiple mixins")
                all_tools.add(tool)

        assert not conflicts, f"Tool name conflicts found: {conflicts}"

    def test_comprehensive_tool_coverage(self):
        """Test that we have comprehensive tool coverage"""
        all_tools = set()
        for mixin in self.mixins:
            components = mixin.get_registered_components()
            all_tools.update(components["tools"])

        # Should have a reasonable number of tools (originally had 24+)
        assert len(all_tools) >= 15, f"Expected at least 15 tools, got {len(all_tools)}: {sorted(all_tools)}"

        # Check for key tool categories
        text_tools = [t for t in all_tools if "text" in t or "ocr" in t]
        table_tools = [t for t in all_tools if "table" in t]
        form_tools = [t for t in all_tools if "form" in t]

        assert len(text_tools) > 0, "No text extraction tools found"
        assert len(table_tools) > 0, "No table extraction tools found"
        assert len(form_tools) > 0, "No form processing tools found"

    def test_mixin_permission_aggregation(self):
        """Test that permissions from all mixins can be aggregated"""
        all_permissions = set()

        for mixin in self.mixins:
            permissions = mixin.get_required_permissions()
            all_permissions.update(permissions)

        # Should include key permission categories
        expected_permissions = ["read_files", "write_files"]
        for perm in expected_permissions:
            assert perm in all_permissions, f"Permission '{perm}' not found in {all_permissions}"


class TestMixinErrorHandling:
    """Test error handling across mixins"""

    def setup_method(self):
        """Setup test environment"""
        self.mcp = FastMCP("test-error-handling")

    def test_mixin_initialization_errors(self):
        """Test how mixins handle initialization errors"""
        # Test with invalid configuration
        try:
            mixin = TextExtractionMixin(self.mcp, invalid_config="test")
            # Should still initialize but might log warnings
            assert mixin.get_mixin_name() == "TextExtraction"
        except Exception as e:
            pytest.fail(f"Mixin should handle invalid config gracefully: {e}")

    @pytest.mark.asyncio
    async def test_tool_error_consistency(self):
        """Test that all tools handle errors consistently"""
        text_mixin = TextExtractionMixin(self.mcp)

        # All tools should return consistent error format
        result = await text_mixin.extract_text("/invalid/path.pdf")

        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        assert isinstance(result["error"], str)


class TestMixinPerformance:
    """Test performance aspects of mixin architecture"""

    def test_mixin_initialization_speed(self):
        """Test that mixin initialization is reasonably fast"""
        import time

        start_time = time.time()
        mcp = FastMCP("test-performance")

        # Initialize all mixins
        mixins = []
        mixin_classes = [
            TextExtractionMixin,
            TableExtractionMixin,
            DocumentAnalysisMixin,
            ImageProcessingMixin,
            FormManagementMixin,
            DocumentAssemblyMixin,
            AnnotationsMixin,
        ]

        for mixin_class in mixin_classes:
            mixin = mixin_class(mcp)
            mixins.append(mixin)

        initialization_time = time.time() - start_time

        # Should initialize in a reasonable time (< 1 second)
        assert initialization_time < 1.0, f"Mixin initialization took too long: {initialization_time}s"

    def test_tool_registration_efficiency(self):
        """Test that tool registration is efficient"""
        mcp = FastMCP("test-registration")

        # Time the registration process
        import time
        start_time = time.time()

        text_mixin = TextExtractionMixin(mcp)

        registration_time = time.time() - start_time

        # Should register quickly
        assert registration_time < 0.5, f"Tool registration took too long: {registration_time}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])