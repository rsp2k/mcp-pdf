"""MCP PDF Tools - A comprehensive PDF processing server for MCP"""

from ._version import __version__
from .server import create_server

__all__ = ["create_server", "__version__"]
