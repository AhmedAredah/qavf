"""Release assurance through independent generated and differential contracts."""

import pytest

pytest.register_assert_rewrite(f"{__name__}.support.contracts")
