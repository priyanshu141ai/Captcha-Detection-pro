"""Lazy FastAPI application entry point."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name in {"app", "create_app"}:
        from cipherlens.api import application

        return getattr(application, name)
    raise AttributeError(name)


__all__ = ["app", "create_app"]
