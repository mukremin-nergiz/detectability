"""detectability: will your geophysical survey see the target?"""

__version__ = "0.1.0.dev0"

from .survey import METHODS, SHAPES, Report, Survey, Target, evaluate  # noqa: E402

__all__ = ["evaluate", "Target", "Survey", "Report", "SHAPES", "METHODS", "__version__"]
