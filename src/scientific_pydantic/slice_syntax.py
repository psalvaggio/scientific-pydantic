"""Utilities for slice string syntax"""

import typing as ty


class SliceSyntaxError(ValueError):
    """Raised when a slice syntax string is invalid."""


T = ty.TypeVar("T")


def parse_slice_syntax(
    value: str,
    *,
    converter: ty.Callable[[str], T],
    require_start: bool,
    require_stop: bool,
) -> tuple[T | None, T | None, T | None]:
    """Parse a Python-style slice string: [start]:[stop][:step].

    Parameters
    ----------
    value : str
        Slice syntax string.
    converter : Callable[[str], T]
        Conversion function to use on each element if present
    require_start : bool
        Whether start must be present and non-empty.
    require_stop : bool
        Whether stop must be present and non-empty.

    Returns
    -------
    (start, stop, step)
    """
    parts = value.split(":")
    if not 2 <= len(parts) <= 3:  # noqa: PLR2004
        msg = "Invalid slice syntax"
        raise SliceSyntaxError(msg)

    def _parse(part: str) -> int | None:
        part = part.strip()
        if part == "":
            return None
        return converter(part)

    try:
        start = _parse(parts[0])
        stop = _parse(parts[1])
        step = _parse(parts[2]) if len(parts) == 3 else None  # noqa: PLR2004
    except (ValueError, TypeError) as exc:
        msg = "Invalid integer in slice string"
        raise SliceSyntaxError(msg) from exc

    if require_start and start is None:
        msg = "Start is required"
        raise SliceSyntaxError(msg)
    if require_stop and stop is None:
        msg = "Stop is required"
        raise SliceSyntaxError(msg)
    if step == 0:
        msg = "Step must not be zero"
        raise SliceSyntaxError(msg)

    return start, stop, step


def format_slice_syntax(start: ty.Any, stop: ty.Any, step: ty.Any) -> str:
    """Format slice components into canonical slice syntax."""
    if step is None:
        return f"{start or ''}:{stop or ''}"
    return f"{start or ''}:{stop or ''}:{step}"
