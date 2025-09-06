"""
Example usage of MCP PDF Tools server

This script demonstrates how to test the PDF tools locally.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_pdf.server import create_server


async def call_tool(mcp, tool_name: str, **kwargs):
    """Call a tool through the MCP server"""
    tools = await mcp.get_tools()
    if tool_name not in tools:
        raise ValueError(f"Tool '{tool_name}' not found")
    
    tool = tools[tool_name]
    # Call the tool's function directly using the fn attribute
    result = await tool.fn(**kwargs)
    return result


async def test_pdf_tools(pdf_path: str):
    """Test various PDF tools on a given PDF file"""
    
    # Create the MCP server
    mcp = create_server()
    
    print(f"\n{'='*60}")
    print(f"Testing PDF Tools on: {pdf_path}")
    print(f"{'='*60}\n")
    
    # 1. Check if PDF is scanned
    print("1. Checking if PDF is scanned...")
    scan_result = await call_tool(mcp, "is_scanned_pdf", pdf_path=pdf_path)
    print(f"   Is scanned: {scan_result.get('is_scanned', 'Unknown')}")
    print(f"   Recommendation: {scan_result.get('recommendation', 'N/A')}")
    
    # 2. Extract metadata
    print("\n2. Extracting metadata...")
    metadata_result = await call_tool(mcp, "extract_metadata", pdf_path=pdf_path)
    if "error" not in metadata_result:
        print(f"   Title: {metadata_result['metadata'].get('title', 'N/A')}")
        print(f"   Author: {metadata_result['metadata'].get('author', 'N/A')}")
        print(f"   Pages: {metadata_result['statistics'].get('page_count', 'N/A')}")
        print(f"   File size: {metadata_result['file_info'].get('size_mb', 'N/A')} MB")
    else:
        print(f"   Error: {metadata_result['error']}")
    
    # 3. Get document structure
    print("\n3. Getting document structure...")
    structure_result = await call_tool(mcp, "get_document_structure", pdf_path=pdf_path)
    if "error" not in structure_result:
        print(f"   Outline items: {len(structure_result.get('outline', []))}")
        fonts = structure_result.get('fonts', [])
        if fonts:
            print(f"   Fonts used: {', '.join(fonts[:3])}...")
    else:
        print(f"   Error: {structure_result['error']}")
    
    # 4. Extract text (if not scanned)
    if not scan_result.get('is_scanned', True):
        print("\n4. Extracting text...")
        text_result = await call_tool(mcp, "extract_text", 
                                     pdf_path=pdf_path, 
                                     pages=[0])  # First page only
        if "error" not in text_result:
            text_preview = text_result['text'][:200].replace('\n', ' ')
            print(f"   Method used: {text_result['method_used']}")
            print(f"   Text preview: {text_preview}...")
        else:
            print(f"   Error: {text_result['error']}")
    else:
        print("\n4. Skipping text extraction (PDF is scanned)")
    
    # 5. Extract tables
    print("\n5. Extracting tables...")
    table_result = await call_tool(mcp, "extract_tables",
                                  pdf_path=pdf_path,
                                  pages=[0])  # First page only
    if "error" not in table_result:
        print(f"   Tables found: {table_result['total_tables']}")
        print(f"   Method used: {table_result['method_used']}")
        if table_result['total_tables'] > 0:
            first_table = table_result['tables'][0]
            print(f"   First table shape: {first_table['shape']['rows']}x{first_table['shape']['columns']}")
    else:
        print(f"   Error: {table_result['error']}")
    
    # 6. Convert to Markdown (first page)
    print("\n6. Converting to Markdown...")
    markdown_result = await call_tool(mcp, "pdf_to_markdown",
                                     pdf_path=pdf_path,
                                     pages=[0],
                                     include_images=False)
    if "error" not in markdown_result:
        md_preview = markdown_result['markdown'][:200].replace('\n', ' ')
        print(f"   Markdown preview: {md_preview}...")
    else:
        print(f"   Error: {markdown_result['error']}")
    
    # 7. Extract images
    print("\n7. Extracting images...")
    images_result = await call_tool(mcp, "extract_images",
                                   pdf_path=pdf_path,
                                   pages=[0])
    if "error" not in images_result:
        print(f"   Images found: {images_result['total_images']}")
        if images_result['total_images'] > 0:
            first_image = images_result['images'][0]
            print(f"   First image size: {first_image['width']}x{first_image['height']}")
    else:
        print(f"   Error: {images_result['error']}")
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}\n")


async def main():
    """Main function to run the tests"""
    if len(sys.argv) < 2:
        print("Usage: python test_pdf_tools.py <path_to_pdf>")
        print("\nExample:")
        print("  python test_pdf_tools.py /path/to/document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    # Check if it's a PDF
    if not pdf_path.lower().endswith('.pdf'):
        print(f"Error: File must be a PDF: {pdf_path}")
        sys.exit(1)
    
    try:
        await test_pdf_tools(pdf_path)
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
