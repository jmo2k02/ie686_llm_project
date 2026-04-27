"""Compatibility wrapper for the Typer CLI package."""

from travelplanner.cli.main import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
