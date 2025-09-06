#!/usr/bin/env python3
"""
Examples of using MCP PDF Tools with URLs
"""

import asyncio
import sys
import os

# Add src to path for development
sys.path.insert(0, '../src')

from mcp_pdf.server import (
    extract_text, extract_metadata, pdf_to_markdown, 
    extract_tables, is_scanned_pdf
)

async def example_text_extraction():
    """Example: Extract text from a PDF URL"""
    print("🔗 Extracting text from URL...")
    
    # Using a sample PDF from the web
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    try:
        result = await extract_text(url)
        print(f"✅ Text extraction successful!")
        print(f"   Method used: {result['method_used']}")
        print(f"   Pages: {result['metadata']['pages']}")
        print(f"   Extracted text length: {len(result['text'])} characters")
        print(f"   First 100 characters: {result['text'][:100]}...")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

async def example_metadata_extraction():
    """Example: Extract metadata from a PDF URL"""
    print("\n📋 Extracting metadata from URL...")
    
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    try:
        result = await extract_metadata(url)
        print(f"✅ Metadata extraction successful!")
        print(f"   File size: {result['file_info']['size_mb']:.2f} MB")
        print(f"   Pages: {result['statistics']['page_count']}")
        print(f"   Title: {result['metadata'].get('title', 'No title')}")
        print(f"   Creation date: {result['metadata'].get('creation_date', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

async def example_scanned_detection():
    """Example: Check if PDF is scanned"""
    print("\n🔍 Checking if PDF is scanned...")
    
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    try:
        result = await is_scanned_pdf(url)
        print(f"✅ Scanned detection successful!")
        print(f"   Is scanned: {result['is_scanned']}")
        print(f"   Recommendation: {result['recommendation']}")
        print(f"   Pages checked: {result['sample_pages_checked']}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

async def example_markdown_conversion():
    """Example: Convert PDF URL to markdown"""
    print("\n📝 Converting PDF to markdown...")
    
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    try:
        result = await pdf_to_markdown(url)
        print(f"✅ Markdown conversion successful!")
        print(f"   Pages converted: {result['pages_converted']}")
        print(f"   Markdown length: {len(result['markdown'])} characters")
        print(f"   First 200 characters:")
        print(f"   {result['markdown'][:200]}...")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

async def main():
    """Run all URL examples"""
    print("🌐 MCP PDF Tools - URL Examples")
    print("=" * 50)
    
    await example_text_extraction()
    await example_metadata_extraction() 
    await example_scanned_detection()
    await example_markdown_conversion()
    
    print("\n✨ URL examples completed!")
    print("\n💡 Tips:")
    print("   • URLs are cached for 1 hour to avoid repeated downloads")
    print("   • Use HTTPS URLs for security")
    print("   • The server validates content is actually a PDF file")
    print("   • All tools support the same URL format")

if __name__ == "__main__":
    asyncio.run(main())