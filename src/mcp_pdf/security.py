"""
Security utilities for MCP PDF Tools server

Provides centralized security functions that can be shared across all mixins:
- Input validation and sanitization
- Path traversal protection
- Error message sanitization
- File size and permission checks
"""

import os
import re
import ast
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from urllib.parse import urlparse
import httpx

logger = logging.getLogger(__name__)

# Security Configuration
MAX_PDF_SIZE = 100 * 1024 * 1024  # 100MB
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_PAGES_PROCESS = 1000
MAX_JSON_SIZE = 10000  # 10KB for JSON parameters
PROCESSING_TIMEOUT = 300  # 5 minutes

# Allowed domains for URL downloads (empty list means disabled by default)
ALLOWED_DOMAINS = []


def parse_pages_parameter(pages: Union[str, List[int], None]) -> Optional[List[int]]:
    """
    Parse pages parameter from various formats into a list of 0-based integers.
    User input is 1-based (page 1 = first page), converted to 0-based internally.
    """
    if pages is None:
        return None

    if isinstance(pages, list):
        # Convert 1-based user input to 0-based internal representation
        return [max(0, int(p) - 1) for p in pages]

    if isinstance(pages, str):
        try:
            # Validate input length to prevent abuse
            if len(pages.strip()) > 1000:
                raise ValueError("Pages parameter too long")

            # Handle string representations like "[1, 2, 3]" or "1,2,3"
            if pages.strip().startswith('[') and pages.strip().endswith(']'):
                page_list = ast.literal_eval(pages.strip())
            elif ',' in pages:
                page_list = [int(p.strip()) for p in pages.split(',')]
            else:
                page_list = [int(pages.strip())]

            # Convert 1-based user input to 0-based internal representation
            return [max(0, int(p) - 1) for p in page_list]

        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid pages parameter: {pages}. Use format like '1,2,3' or '1-5'")

    raise ValueError(f"Unsupported pages parameter type: {type(pages)}")


def validate_pages_parameter(pages: str) -> List[int]:
    """
    Validate and parse pages parameter.
    Args:
        pages: Page specification (e.g., "1-5,10,15-20" or "all")
    Returns:
        List of 0-based page indices
    """
    result = parse_pages_parameter(pages)
    return result if result is not None else []


async def validate_pdf_path(pdf_path: str) -> Path:
    """
    Validate PDF path and handle URL downloads securely.

    Args:
        pdf_path: File path or URL to PDF

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid or insecure
        FileNotFoundError: If file doesn't exist
    """
    if not pdf_path:
        raise ValueError("PDF path cannot be empty")

    # Handle URLs
    if pdf_path.startswith(('http://', 'https://')):
        return await _download_url_safely(pdf_path)

    # Handle local file paths
    path = Path(pdf_path).resolve()

    # Check for path traversal attempts
    if '../' in str(pdf_path) or '\\..\\' in str(pdf_path):
        raise ValueError("Path traversal detected in PDF path")

    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    # Check if it's a file (not directory)
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    # Check file size
    file_size = path.stat().st_size
    if file_size > MAX_PDF_SIZE:
        raise ValueError(f"PDF file too large: {file_size / (1024*1024):.1f}MB > {MAX_PDF_SIZE / (1024*1024)}MB")

    # Basic PDF header validation
    try:
        with open(path, 'rb') as f:
            header = f.read(8)
            if not header.startswith(b'%PDF-'):
                raise ValueError("File does not appear to be a valid PDF")
    except Exception as e:
        raise ValueError(f"Cannot read PDF file: {e}")

    return path


async def _download_url_safely(url: str) -> Path:
    """
    Download PDF from URL with security checks.

    Args:
        url: URL to download from

    Returns:
        Path to downloaded file in cache directory
    """
    # Validate URL
    parsed_url = urlparse(url)
    if not parsed_url.scheme in ['http', 'https']:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}")

    # Check domain allowlist if configured
    allowed_domains = os.getenv('ALLOWED_DOMAINS', '').split(',')
    if allowed_domains and allowed_domains != ['']:
        if parsed_url.netloc not in allowed_domains:
            raise ValueError(f"Domain not allowed: {parsed_url.netloc}")

    # Create cache directory
    cache_dir = Path(os.environ.get("PDF_TEMP_DIR", "/tmp/mcp-pdf-processing"))
    cache_dir.mkdir(exist_ok=True, parents=True, mode=0o700)

    # Generate safe filename
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cached_file = cache_dir / f"downloaded_{url_hash}.pdf"

    # Check if already cached
    if cached_file.exists():
        # Validate cached file
        if cached_file.stat().st_size <= MAX_PDF_SIZE:
            logger.info(f"Using cached PDF: {cached_file}")
            return cached_file
        else:
            cached_file.unlink()  # Remove oversized cached file

    # Download with security checks
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream('GET', url) as response:
                response.raise_for_status()

                # Check content type
                content_type = response.headers.get('content-type', '')
                if 'application/pdf' not in content_type.lower():
                    logger.warning(f"Unexpected content type: {content_type}")

                # Stream download with size checking
                downloaded_size = 0
                with open(cached_file, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        downloaded_size += len(chunk)
                        if downloaded_size > MAX_PDF_SIZE:
                            f.close()
                            cached_file.unlink()
                            raise ValueError(f"Downloaded file too large: {downloaded_size / (1024*1024):.1f}MB")
                        f.write(chunk)

        # Set secure permissions
        cached_file.chmod(0o600)

        logger.info(f"Downloaded PDF: {downloaded_size / (1024*1024):.1f}MB to {cached_file}")
        return cached_file

    except Exception as e:
        if cached_file.exists():
            cached_file.unlink()
        raise ValueError(f"Failed to download PDF: {e}")


def validate_pages_parameter(pages: str) -> List[int]:
    """
    Validate and parse pages parameter.

    Args:
        pages: Page specification (e.g., "1-5,10,15-20" or "all")

    Returns:
        List of page numbers (0-indexed)

    Raises:
        ValueError: If pages parameter is invalid
    """
    if not pages or pages.lower() == "all":
        return None

    if len(pages) > 1000:  # Prevent DoS with extremely long page strings
        raise ValueError("Pages parameter too long")

    try:
        page_numbers = []
        parts = pages.split(',')

        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = part.split('-', 1)
                start_num = int(start.strip())
                end_num = int(end.strip())

                if start_num < 1 or end_num < 1:
                    raise ValueError("Page numbers must be positive")
                if start_num > end_num:
                    raise ValueError(f"Invalid page range: {start_num}-{end_num}")

                # Convert to 0-indexed and add range
                page_numbers.extend(range(start_num - 1, end_num))
            else:
                page_num = int(part.strip())
                if page_num < 1:
                    raise ValueError("Page numbers must be positive")
                page_numbers.append(page_num - 1)  # Convert to 0-indexed

        # Remove duplicates and sort
        page_numbers = sorted(list(set(page_numbers)))

        # Check maximum pages limit
        if len(page_numbers) > MAX_PAGES_PROCESS:
            raise ValueError(f"Too many pages specified: {len(page_numbers)} > {MAX_PAGES_PROCESS}")

        return page_numbers

    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"Invalid page specification: {pages}")
        raise


def validate_json_parameter(json_str: str, max_size: int = MAX_JSON_SIZE) -> Dict[str, Any]:
    """
    Safely parse and validate JSON parameter.

    Args:
        json_str: JSON string to parse
        max_size: Maximum allowed size in bytes

    Returns:
        Parsed JSON object

    Raises:
        ValueError: If JSON is invalid or too large
    """
    if not json_str:
        return {}

    if len(json_str) > max_size:
        raise ValueError(f"JSON parameter too large: {len(json_str)} > {max_size} bytes")

    try:
        # Use ast.literal_eval for basic safety, fallback to json for complex objects
        if json_str.strip().startswith(('{', '[')):
            import json
            return json.loads(json_str)
        else:
            return ast.literal_eval(json_str)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid JSON parameter: {e}")


def validate_output_path(path: str) -> Path:
    """
    Validate and secure output paths to prevent directory traversal.

    Args:
        path: Output path to validate

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid or insecure
    """
    if not path:
        raise ValueError("Output path cannot be empty")

    # Convert to Path and resolve to absolute path
    resolved_path = Path(path).resolve()

    # Check for path traversal attempts
    if '../' in str(path) or '\\..\\' in str(path):
        raise ValueError("Path traversal detected in output path")

    # In stdio mode (Claude Desktop), skip path restrictions - user's local environment
    # Only enforce restrictions for network-exposed deployments
    is_stdio_mode = os.getenv('MCP_TRANSPORT') != 'http' and not os.getenv('MCP_PUBLIC_MODE')

    if is_stdio_mode:
        logger.debug(f"STDIO mode detected - allowing local path: {resolved_path}")
        return resolved_path

    # Check allowed output paths from environment variable (for network deployments)
    allowed_paths = os.getenv('MCP_PDF_ALLOWED_PATHS')

    if allowed_paths is None:
        # No restriction set - warn user but allow any path
        logger.warning(f"MCP_PDF_ALLOWED_PATHS not set - allowing write to any directory: {resolved_path}")
        logger.warning("SECURITY NOTE: This restriction is 'security theater' - real protection comes from OS-level permissions")
        logger.warning("Recommended: Set MCP_PDF_ALLOWED_PATHS='/tmp:/var/tmp:/home/user/documents' AND use proper file permissions")
        return resolved_path

    # Parse allowed paths
    allowed_path_list = [Path(p.strip()).resolve() for p in allowed_paths.split(':') if p.strip()]

    # Check if path is within allowed directories
    for allowed_path in allowed_path_list:
        try:
            resolved_path.relative_to(allowed_path)
            logger.debug(f"Path allowed under: {allowed_path}")
            return resolved_path
        except ValueError:
            continue

    # Path not allowed
    raise ValueError(f"Output path not allowed: {resolved_path}. Allowed paths: {allowed_paths}")


def validate_image_id(image_id: str) -> str:
    """
    Validate image ID to prevent path traversal attacks.

    Args:
        image_id: Image identifier to validate

    Returns:
        Validated image ID

    Raises:
        ValueError: If image ID is invalid
    """
    if not image_id:
        raise ValueError("Image ID cannot be empty")

    # Only allow alphanumeric characters, underscores, and hyphens
    if not re.match(r'^[a-zA-Z0-9_-]+$', image_id):
        raise ValueError(f"Invalid image ID format: {image_id}")

    # Prevent excessively long IDs
    if len(image_id) > 255:
        raise ValueError(f"Image ID too long: {len(image_id)} > 255")

    return image_id


def sanitize_error_message(error_msg: str) -> str:
    """
    Sanitize error messages to prevent information disclosure.

    Args:
        error_msg: Raw error message

    Returns:
        Sanitized error message
    """
    if not error_msg:
        return "Unknown error occurred"

    # Remove sensitive patterns
    patterns_to_remove = [
        r'/home/[^/\s]+',  # Home directory paths
        r'/tmp/[^/\s]+',   # Temp file paths
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN patterns
        r'password[=:]\s*\S+',  # Password assignments
        r'token[=:]\s*\S+',     # Token assignments
    ]

    sanitized = error_msg
    for pattern in patterns_to_remove:
        sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)

    # Limit length to prevent verbose stack traces
    if len(sanitized) > 500:
        sanitized = sanitized[:500] + "... [truncated]"

    return sanitized


def check_file_permissions(file_path: Path, required_permissions: str = 'read') -> bool:
    """
    Check if file has required permissions.

    Args:
        file_path: Path to check
        required_permissions: 'read', 'write', or 'execute'

    Returns:
        True if permissions are sufficient
    """
    if not file_path.exists():
        return False

    if required_permissions == 'read':
        return os.access(file_path, os.R_OK)
    elif required_permissions == 'write':
        return os.access(file_path, os.W_OK)
    elif required_permissions == 'execute':
        return os.access(file_path, os.X_OK)
    else:
        return False


def create_secure_temp_file(suffix: str = '.pdf', prefix: str = 'mcp_pdf_') -> Path:
    """
    Create a secure temporary file with proper permissions.

    Args:
        suffix: File suffix
        prefix: File prefix

    Returns:
        Path to created temporary file
    """
    import tempfile

    cache_dir = Path(os.environ.get("PDF_TEMP_DIR", "/tmp/mcp-pdf-processing"))
    cache_dir.mkdir(exist_ok=True, parents=True, mode=0o700)

    # Create temporary file with secure permissions
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=cache_dir)
    os.close(fd)

    temp_file = Path(temp_path)
    temp_file.chmod(0o600)  # Read/write for owner only

    return temp_file