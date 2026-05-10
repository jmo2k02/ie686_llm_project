"""Planner configuration values loaded from project settings."""

from travelplanner.config import get_setting


MODEL_NAME = get_setting("agents.planner.model_name")
REVIEWER_MODEL_NAME = get_setting("agents.planner.reviewer_model")
TEMPERATURE = get_setting("agents.planner.temperature")
REVIEWER_TEMP = get_setting("agents.planner.reviewer_temp")
MAX_TASKS = get_setting("agents.planner.max_tasks")
MAX_REVIEW_ATTEMPTS = get_setting("agents.planner.max_reviews")
