#!/usr/bin/env python3
"""
Test URL support for MCP PDF Tools
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from mcp_pdf.server import validate_pdf_path, download_pdf_from_url

async def test_url_validation():
    """Test URL validation and download"""
    print("Testing URL validation and download...")
    
    # Test with a known PDF URL (using a publicly available sample)
    test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    try:
        print(f"Testing URL: {test_url}")
        path = await validate_pdf_path(test_url)
        print(f"✅ Successfully downloaded and validated PDF: {path}")
        print(f"   File size: {path.stat().st_size} bytes")
        return True
        
    except Exception as e:
        print(f"❌ URL test failed: {e}")
        return False

async def test_local_path():
    """Test that local paths still work"""
    print("\nTesting local path validation...")
    
    # Test with our existing test PDF
    test_path = "/tmp/test_text.pdf"
    
    if not os.path.exists(test_path):
        print(f"⚠️  Test file {test_path} not found, skipping local test")
        return True
    
    try:
        path = await validate_pdf_path(test_path)
        print(f"✅ Local path validation works: {path}")
        return True
        
    except Exception as e:
        print(f"❌ Local path test failed: {e}")
        return False

async def main():
    print("🧪 Testing MCP PDF Tools URL Support\n")
    
    url_success = await test_url_validation()
    local_success = await test_local_path()
    
    print(f"\n📊 Test Results:")
    print(f"   URL support: {'✅ PASS' if url_success else '❌ FAIL'}")
    print(f"   Local paths: {'✅ PASS' if local_success else '❌ FAIL'}")
    
    if url_success and local_success:
        print("\n🎉 All tests passed! URL support is working.")
        return 0
    else:
        print("\n🚨 Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))