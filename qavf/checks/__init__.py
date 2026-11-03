"""Named checks: the configuration vocabulary that compiles into rule parameters.

* :mod:`qavf.checks.items`: the ``checks`` item and ``run`` section models.
* :mod:`qavf.checks.catalog`: every built-in check name, its rule code and settings.
* :mod:`qavf.checks.compiler`: check items to rule parameters, one rule per item.
* :mod:`qavf.checks.legacy`: recognising QAVF 0.6 configurations.

The catalog and compiler import the rule implementations, so they are not
imported here; :mod:`qavf.settings` depends only on the items and legacy modules.
"""

from qavf.checks.items import CheckItem, RunSettings
from qavf.checks.legacy import LegacyConfigurationError

__all__ = ["CheckItem", "LegacyConfigurationError", "RunSettings"]
