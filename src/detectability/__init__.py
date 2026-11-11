"""Detectability: pre-survey detectability, modelled and reproducible."""

__version__ = "0.5.0"

from .survey import DEFAULT_METHODS, METHODS, SHAPES, Report, Survey, Target, evaluate  # noqa: E402

__all__ = ["evaluate", "Target", "Survey", "Report", "SHAPES", "METHODS", "DEFAULT_METHODS", "__version__"]
