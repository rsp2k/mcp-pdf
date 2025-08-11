#!/usr/bin/env python3
"""
Test the updated pages parameter parsing
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from mcp_pdf_tools.server import parse_pages_parameter

def test_page_parsing():
    """Test page parameter parsing"""
    print("Testing page parameter parsing...")
    
    # Test different input formats
    test_cases = [
        (None, None),
        ("1,2,3", [1, 2, 3]),
        ("[2, 3]", [2, 3]),  # This is the problematic case from the user
        ("5", [5]),
        ([0, 1, 2], [0, 1, 2]),
        ("0,1,2", [0, 1, 2]),
        ("[0,1,2]", [0, 1, 2])
    ]
    
    all_passed = True
    
    for input_val, expected in test_cases:
        try:
            result = parse_pages_parameter(input_val)
            if result == expected:
                print(f"✅ '{input_val}' -> {result}")
            else:
                print(f"❌ '{input_val}' -> {result}, expected {expected}")
                all_passed = False
        except Exception as e:
            print(f"❌ '{input_val}' -> Error: {e}")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    success = test_page_parsing()
    if success:
        print("\n🎉 All page parameter parsing tests passed!")
    else:
        print("\n🚨 Some tests failed!")
    sys.exit(0 if success else 1)