"""MCP PDF Tools - A comprehensive PDF processing server for MCP"""

__version__ = "0.1.0"

from .server import create_server

__all__ = ["create_server", "__version__"]
