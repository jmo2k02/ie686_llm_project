"""Planner configuration values loaded from project settings."""

from travelplanner.config import get_setting


MODEL_NAME = get_setting("agents.planner.model_name")
TEMPERATURE = get_setting("agents.planner.temperature")
MAX_TASKS = get_setting("agents.planner.max_tasks")
MAX_REVIEW_ATTEMPTS = get_setting("agents.planner.max_reviews")
