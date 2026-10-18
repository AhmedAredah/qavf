"""Reproducible generated cases, with an opt-in deeper exploration profile."""

import os

from hypothesis import settings

settings.register_profile("ci", max_examples=25, deadline=None, derandomize=True, database=None)
settings.register_profile(
    "extended", max_examples=200, deadline=None, derandomize=True, database=None
)
settings.register_profile("randomized", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("QAVF_HYPOTHESIS_PROFILE", "ci"))
