"""
Shared utility functions for official mixins
"""

from typing import Optional, List


def parse_pages_parameter(pages: Optional[str]) -> Optional[List[int]]:
    """Parse pages parameter from string to list of 0-based page numbers

    Supports formats:
    - Single page: "5"
    - Comma-separated: "1,3,5"
    - Ranges: "1-10" or "11-30"
    - Mixed: "1,3-5,7,10-15"

    Args:
        pages: Page specification string (1-based page numbers)

    Returns:
        List of 0-based page indices, or None if pages is None
    """
    if not pages:
        return None

    try:
        result = []
        parts = pages.split(',')

        for part in parts:
            part = part.strip()

            # Handle range (e.g., "1-10" or "11-30")
            if '-' in part:
                range_parts = part.split('-')
                if len(range_parts) == 2:
                    start = int(range_parts[0].strip())
                    end = int(range_parts[1].strip())
                    # Convert 1-based to 0-based and create range
                    result.extend(range(start - 1, end))
                else:
                    return None
            # Handle single page
            else:
                result.append(int(part) - 1)

        return result
    except (ValueError, AttributeError):
        return None
