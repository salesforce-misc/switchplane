"""Configuration model for the quality review app."""

from __future__ import annotations

from pydantic import BaseModel

from switchplane.config import AppConfig

# Hosts a PR URL is allowed to name when no ``allowed_hosts`` is configured.
# Matches ava's default: github.com and Salesforce-internal git.soma.
DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = ("github.com",)


class ReviewConfig(BaseModel):
    """Review-specific configuration."""

    allowed_hosts: list[str] = list(DEFAULT_ALLOWED_HOSTS)


class QualityConfig(AppConfig):
    """App-level configuration extending AppConfig with review-specific settings.

    Without this subclass, AppConfig silently drops the [review] section and
    the task would read defaults while appearing configured.
    """

    review: ReviewConfig = ReviewConfig()
