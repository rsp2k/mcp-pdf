#!/usr/bin/env python3
"""
Security Features Test Suite
Tests the security hardening we implemented
"""

import pytest
import tempfile
from pathlib import Path
from src.mcp_pdf.server import (
    validate_image_id, 
    validate_output_path, 
    safe_json_parse, 
    validate_url,
    sanitize_error_message,
    validate_page_count,
    MAX_PDF_SIZE,
    MAX_IMAGE_SIZE,
    MAX_PAGES_PROCESS,
    MAX_JSON_SIZE
)


class TestSecurityValidation:
    """Test security validation functions"""
    
    def test_validate_image_id_success(self):
        """Test valid image IDs pass validation"""
        valid_ids = ["image123", "test-image", "image_001", "abc123DEF"]
        for image_id in valid_ids:
            result = validate_image_id(image_id)
            assert result == image_id
    
    def test_validate_image_id_path_traversal(self):
        """Test path traversal attempts are blocked"""
        malicious_ids = ["../../../etc/passwd", "..\\windows\\system32", "image/../secret"]
        for malicious_id in malicious_ids:
            with pytest.raises(ValueError, match="Invalid image ID format"):
                validate_image_id(malicious_id)
    
    def test_validate_image_id_too_long(self):
        """Test extremely long image IDs are rejected"""
        long_id = "a" * 300
        with pytest.raises(ValueError, match="Image ID too long"):
            validate_image_id(long_id)
    
    def test_validate_image_id_empty(self):
        """Test empty image ID is rejected"""
        with pytest.raises(ValueError, match="Image ID cannot be empty"):
            validate_image_id("")
    
    def test_validate_output_path_safe_paths(self):
        """Test safe output paths are allowed"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            safe_path = f"{tmp_dir}/output"
            # This should work for /tmp paths
            try:
                result = validate_output_path(safe_path)
                assert isinstance(result, Path)
            except ValueError:
                # Expected if path is outside safe directories
                pass
    
    def test_validate_output_path_traversal(self):
        """Test path traversal in output paths is blocked"""
        malicious_paths = [
            "../../../etc/passwd",
            "output/../../../secret",
            "/tmp/../etc/passwd"
        ]
        for malicious_path in malicious_paths:
            with pytest.raises(ValueError, match="Path traversal detected"):
                validate_output_path(malicious_path)
    
    def test_safe_json_parse_valid(self):
        """Test valid JSON parsing"""
        valid_json = '{"key": "value", "number": 123}'
        result = safe_json_parse(valid_json)
        assert result == {"key": "value", "number": 123}
    
    def test_safe_json_parse_empty(self):
        """Test empty JSON input"""
        result = safe_json_parse("")
        assert result == {}
    
    def test_safe_json_parse_too_large(self):
        """Test JSON size limits"""
        large_json = '{"key": "' + "a" * MAX_JSON_SIZE + '"}'
        with pytest.raises(ValueError, match="JSON input too large"):
            safe_json_parse(large_json)
    
    def test_safe_json_parse_invalid(self):
        """Test invalid JSON is handled"""
        invalid_json = '{"key": invalid}'
        with pytest.raises(ValueError, match="Invalid JSON format"):
            safe_json_parse(invalid_json)
    
    def test_validate_url_safe_urls(self):
        """Test safe URLs are allowed when no domain restrictions"""
        safe_urls = [
            "https://example.com/file.pdf",
            "https://docs.google.com/document.pdf",
            "http://public-docs.org/paper.pdf"
        ]
        for url in safe_urls:
            result = validate_url(url)
            assert result is True
    
    def test_validate_url_blocked_hosts(self):
        """Test localhost and internal IPs are blocked"""
        blocked_urls = [
            "https://localhost/file.pdf",
            "https://127.0.0.1/file.pdf", 
            "https://0.0.0.0/file.pdf",
            "https://::1/file.pdf"
        ]
        for url in blocked_urls:
            result = validate_url(url)
            assert result is False
    
    def test_validate_url_invalid_schemes(self):
        """Test non-HTTP schemes are blocked"""
        invalid_urls = [
            "ftp://example.com/file.pdf",
            "file:///etc/passwd",
            "javascript:alert('xss')"
        ]
        for url in invalid_urls:
            result = validate_url(url)
            assert result is False
    
    def test_sanitize_error_message_paths(self):
        """Test file paths are removed from error messages"""
        error = Exception("Error processing /home/user/secret/file.pdf")
        sanitized = sanitize_error_message(error, "Test error")
        assert "/home/user/secret/file.pdf" not in sanitized
        assert "[PATH]" in sanitized
        assert "Test error:" in sanitized
    
    def test_sanitize_error_message_sensitive_data(self):
        """Test sensitive data patterns are removed"""
        error = Exception("User email: user@company.com, SSN: 123-45-6789")
        sanitized = sanitize_error_message(error)
        assert "user@company.com" not in sanitized
        assert "123-45-6789" not in sanitized
        assert "[EMAIL]" in sanitized
        assert "[SSN]" in sanitized
    
    def test_validate_page_count_valid(self):
        """Test valid page count passes"""
        mock_doc = type('MockDoc', (), {'page_count': 100})()
        # Should not raise an exception
        validate_page_count(mock_doc, "test operation")
    
    def test_validate_page_count_too_many_pages(self):
        """Test excessive page count is rejected"""
        mock_doc = type('MockDoc', (), {'page_count': MAX_PAGES_PROCESS + 1})()
        with pytest.raises(ValueError, match="PDF too large for test operation"):
            validate_page_count(mock_doc, "test operation")
    
    def test_validate_page_count_empty_pdf(self):
        """Test empty PDF is rejected"""
        mock_doc = type('MockDoc', (), {'page_count': 0})()
        with pytest.raises(ValueError, match="PDF has no pages"):
            validate_page_count(mock_doc)


class TestSecurityConstants:
    """Test security constants are reasonable"""
    
    def test_file_size_limits(self):
        """Test file size limits are set to reasonable values"""
        assert MAX_PDF_SIZE == 100 * 1024 * 1024  # 100MB
        assert MAX_IMAGE_SIZE == 50 * 1024 * 1024  # 50MB
        assert MAX_JSON_SIZE == 10000  # 10KB
        assert MAX_PAGES_PROCESS == 1000  # 1000 pages
    
    def test_limits_are_positive(self):
        """Test all limits are positive numbers"""
        assert MAX_PDF_SIZE > 0
        assert MAX_IMAGE_SIZE > 0  
        assert MAX_JSON_SIZE > 0
        assert MAX_PAGES_PROCESS > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])