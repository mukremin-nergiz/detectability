"""Detectability: pre-survey detectability, modelled and reproducible."""

__version__ = "0.2.1"

from .survey import METHODS, SHAPES, Report, Survey, Target, evaluate  # noqa: E402

__all__ = ["evaluate", "Target", "Survey", "Report", "SHAPES", "METHODS", "__version__"]
