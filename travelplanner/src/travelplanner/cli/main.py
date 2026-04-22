"""Main Typer application for travelplanner."""

import typer

from .evaluation import app as evaluation_app

app = typer.Typer(
    help="TravelPlanner command line tools.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(evaluation_app, name="eval")


def main() -> None:
    """Run the CLI application."""
    app()


if __name__ == "__main__":
    main()
