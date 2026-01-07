"""
Constitutional AI Implementation

A reimplementation inspired by "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022).
This package provides tools for generating AI responses that are critiqued and revised
according to constitutional principles.
"""

from .constitutional_ai import ConstitutionalAI
from .principles import CONSTITUTIONAL_PRINCIPLES, Principle

__version__ = "0.1.0"
__all__ = ["ConstitutionalAI", "CONSTITUTIONAL_PRINCIPLES", "Principle"]
