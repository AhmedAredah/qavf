"""Resource errors shared by core and optional adapter engines."""


class ResourceLimitError(RuntimeError):
    """A configured retention, spill or report budget was exceeded."""
