"""
Base MCPMixin class providing auto-registration and modular architecture
"""

import inspect
from typing import Dict, Any, List, Optional, Set, Callable
from abc import ABC, abstractmethod
from fastmcp import FastMCP
import logging

logger = logging.getLogger(__name__)


class MCPMixin(ABC):
    """
    Base mixin class for modular MCP server components.

    Provides:
    - Auto-registration of tools, resources, and prompts
    - Permission-based progressive disclosure
    - Consistent error handling and logging
    - Shared utility access
    """

    def __init__(self, mcp_server: FastMCP, **kwargs):
        self.mcp = mcp_server
        self.config = kwargs
        self._registered_tools: Set[str] = set()
        self._registered_resources: Set[str] = set()
        self._registered_prompts: Set[str] = set()

        # Initialize mixin-specific setup
        self._setup()

        # Auto-register components
        self._auto_register()

    @abstractmethod
    def get_mixin_name(self) -> str:
        """Return the name of this mixin for logging and identification"""
        pass

    @abstractmethod
    def get_required_permissions(self) -> List[str]:
        """Return list of permissions required for this mixin's tools"""
        pass

    def _setup(self):
        """Override for mixin-specific initialization"""
        pass

    def _auto_register(self):
        """Automatically discover and register tools, resources, and prompts"""
        mixin_name = self.get_mixin_name()
        logger.info(f"Auto-registering components for {mixin_name}")

        # Find all methods that should be registered
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            # Skip private methods and inherited methods
            if name.startswith('_') or not hasattr(self.__class__, name):
                continue

            # Check for MCP decorators or naming conventions
            if hasattr(method, '_mcp_tool_config'):
                self._register_tool_method(name, method)
            elif hasattr(method, '_mcp_resource_config'):
                self._register_resource_method(name, method)
            elif hasattr(method, '_mcp_prompt_config'):
                self._register_prompt_method(name, method)
            elif self._should_auto_register_tool(name, method):
                self._auto_register_tool(name, method)

    def _should_auto_register_tool(self, name: str, method: Callable) -> bool:
        """Determine if a method should be auto-registered as a tool"""
        # Convention: public async methods that don't start with 'get_' or 'is_'
        return (
            not name.startswith('_') and
            inspect.iscoroutinefunction(method) and
            not name.startswith(('get_', 'is_', 'validate_', 'setup_'))
        )

    def _register_tool_method(self, name: str, method: Callable):
        """Register a method as an MCP tool"""
        tool_config = getattr(method, '_mcp_tool_config', {})
        tool_name = tool_config.get('name', name)

        # Apply the tool decorator
        decorated_method = self.mcp.tool(
            name=tool_name,
            description=tool_config.get('description', f"{name} tool from {self.get_mixin_name()}"),
            **tool_config.get('kwargs', {})
        )(method)

        self._registered_tools.add(tool_name)
        logger.debug(f"Registered tool: {tool_name} from {self.get_mixin_name()}")

    def _auto_register_tool(self, name: str, method: Callable):
        """Auto-register a method as a tool using conventions"""
        # Generate description from method docstring or name
        description = self._extract_description(method) or f"{name.replace('_', ' ').title()} - {self.get_mixin_name()}"

        # Apply the tool decorator
        decorated_method = self.mcp.tool(
            name=name,
            description=description
        )(method)

        self._registered_tools.add(name)
        logger.debug(f"Auto-registered tool: {name} from {self.get_mixin_name()}")

    def _extract_description(self, method: Callable) -> Optional[str]:
        """Extract description from method docstring"""
        if method.__doc__:
            lines = method.__doc__.strip().split('\n')
            return lines[0].strip() if lines else None
        return None

    def get_registered_components(self) -> Dict[str, Any]:
        """Return summary of registered components"""
        return {
            "mixin": self.get_mixin_name(),
            "tools": list(self._registered_tools),
            "resources": list(self._registered_resources),
            "prompts": list(self._registered_prompts),
            "permissions_required": self.get_required_permissions()
        }


def mcp_tool(name: Optional[str] = None, description: Optional[str] = None, **kwargs):
    """
    Decorator to mark methods for MCP tool registration.

    Usage:
        @mcp_tool(name="extract_text", description="Extract text from PDF")
        async def extract_text_from_pdf(self, pdf_path: str) -> str:
            ...
    """
    def decorator(func):
        func._mcp_tool_config = {
            'name': name,
            'description': description,
            'kwargs': kwargs
        }
        return func
    return decorator


def mcp_resource(uri: str, name: Optional[str] = None, description: Optional[str] = None, **kwargs):
    """
    Decorator to mark methods for MCP resource registration.
    """
    def decorator(func):
        func._mcp_resource_config = {
            'uri': uri,
            'name': name,
            'description': description,
            'kwargs': kwargs
        }
        return func
    return decorator


def mcp_prompt(name: str, description: Optional[str] = None, **kwargs):
    """
    Decorator to mark methods for MCP prompt registration.
    """
    def decorator(func):
        func._mcp_prompt_config = {
            'name': name,
            'description': description,
            'kwargs': kwargs
        }
        return func
    return decorator