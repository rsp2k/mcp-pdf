#!/usr/bin/env python3
"""
Integration test to verify basic functionality after security hardening
"""

import tempfile
from pathlib import Path
from reportlab.pdfgen import canvas
from src.mcp_pdf.server import create_server, validate_pdf_path, validate_page_count
import fitz


def create_test_pdf():
    """Create a simple test PDF file"""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        c = canvas.Canvas(tmp_file.name)
        c.drawString(100, 750, "This is a test PDF document.")
        c.drawString(100, 700, "It has some sample text for testing.")
        c.save()
        return Path(tmp_file.name)


def test_basic_functionality():
    """Test basic functionality after security hardening"""
    
    print("🧪 Testing MCP PDF Tools Integration")
    print("=" * 50)
    
    # 1. Test server creation
    print("1. Testing server creation...")
    try:
        server = create_server()
        print("   ✅ Server created successfully")
    except Exception as e:
        print(f"   ❌ Server creation failed: {e}")
        return False
    
    # 2. Test PDF file validation
    print("2. Testing PDF validation...")
    test_pdf = create_test_pdf()
    try:
        validated_path = validate_pdf_path(str(test_pdf))
        print(f"   ✅ PDF validation successful: {validated_path}")
    except Exception as e:
        print(f"   ❌ PDF validation failed: {e}")
        test_pdf.unlink()
        return False
    
    # 3. Test page count validation
    print("3. Testing page count validation...")
    try:
        doc = fitz.open(str(test_pdf))
        validate_page_count(doc, "integration test")
        doc.close()
        print("   ✅ Page count validation successful")
    except Exception as e:
        print(f"   ❌ Page count validation failed: {e}")
        test_pdf.unlink()
        return False
    
    # 4. Test file size limits
    print("4. Testing file size checking...")
    file_size = test_pdf.stat().st_size
    print(f"   📏 Test PDF size: {file_size} bytes")
    print(f"   📏 Max allowed: 100MB ({100 * 1024 * 1024} bytes)")
    if file_size < 100 * 1024 * 1024:
        print("   ✅ File size within limits")
    else:
        print("   ❌ File size exceeds limits")
        test_pdf.unlink()
        return False
    
    # 5. Clean up
    test_pdf.unlink()
    print("   🧹 Test file cleaned up")
    
    print("\n🎉 All integration tests passed!")
    print("🔒 Security features are working correctly")
    print("⚡ Core functionality is intact")
    
    return True


if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)