from importlib import import_module
from typing import Any


def load_object(import_string: str) -> Any:
    """Load an attribute from a ``module:attribute`` import string."""
    module_name, attr = import_string.split(":", 1)
    module = import_module(module_name)
    return getattr(module, attr)


def load_callable(import_string: str):
    """Load a callable from a ``module:attribute`` import string."""
    obj = load_object(import_string)
    if not callable(obj):
        raise TypeError(f"{import_string} is not callable")
    return obj
