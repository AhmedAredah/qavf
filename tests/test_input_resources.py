"""Input and resource-policy boundaries."""

from decimal import localcontext

import pytest

from qavf import ResourceLimits


@pytest.mark.parametrize("precision", [3, 28, 60])
@pytest.mark.parametrize(
    ("quantity", "expected"),
    [
        ("123456789B", 123456789),
        ("1024KiB", 1048576),
        ("0.5KiB", 512),
        ("1.1KB", 1100),
        ("123456789012345678901234567890123B", 123456789012345678901234567890123),
    ],
)
def test_resource_byte_quantities_do_not_depend_on_decimal_precision(precision, quantity, expected):
    with localcontext() as context:
        context.prec = precision
        limits = ResourceLimits(target_memory=quantity, address_space=quantity)
        configured = limits.engine_overrides()
    assert configured["memory_budget_bytes"] == expected
    assert configured["process_address_space_bytes"] == expected
    assert limits.engine_overrides() == configured


@pytest.mark.parametrize("precision", [3, 28, 60])
@pytest.mark.parametrize("quantity", ["0.1KiB", "0.999999999999999999999999999999999B"])
def test_fractional_resource_bytes_are_rejected_without_rounding(precision, quantity):
    with localcontext() as context:
        context.prec = precision
        with pytest.raises(ValueError, match="whole number of bytes"):
            ResourceLimits(max_spill=quantity)
