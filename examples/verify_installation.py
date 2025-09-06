#!/usr/bin/env python3
"""
Simple test script to verify the MCP server can be initialized
"""

import sys
import asyncio
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def main():
    try:
        from mcp_pdf import create_server, __version__
        
        print(f"✅ MCP PDF Tools v{__version__} imported successfully!")
        
        # Try to create the server
        mcp = create_server()
        print("✅ Server created successfully!")
        
        # Check available tools
        tools = await mcp.get_tools()
        
        print(f"\n📋 Available tools ({len(tools)}):")
        for tool_name in sorted(tools.keys()):
            print(f"   - {tool_name}")
        
        print("\n✅ All systems operational! The MCP server is ready to use.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
