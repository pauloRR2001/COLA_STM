"""Utilities for retrieving, parsing, and storing two-line element sets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import requests

_CELESTRAK_GP_URL: Final[str] = (
    "https://celestrak.org/NORAD/elements/gp.php"
)
_REQUEST_TIMEOUT_SECONDS: Final[float] = 15.0


class TLEError(RuntimeError):
    """Raised when a TLE cannot be retrieved or parsed."""


@dataclass(slots=True)
class TLE:
    """A named two-line element set."""

    name: str
    line1: str
    line2: str


def _validate_tle(tle: TLE) -> TLE:
    """Validate the required structure of a TLE and return it unchanged."""
    if not tle.name.strip():
        raise TLEError("TLE name cannot be empty.")
    if not tle.line1.startswith("1 "):
        raise TLEError("Invalid TLE line 1: expected it to start with '1 '.")
    if not tle.line2.startswith("2 "):
        raise TLEError("Invalid TLE line 2: expected it to start with '2 '.")
    return tle


def _download_tle(params: dict[str, str]) -> TLE:
    """Download and parse one TLE from the CelesTrak GP endpoint."""
    try:
        response = requests.get(
            _CELESTRAK_GP_URL,
            params={**params, "FORMAT": "TLE"},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        query = ", ".join(f"{key}={value!r}" for key, value in params.items())
        raise TLEError(
            f"Failed to download TLE from CelesTrak ({query}): {exc}"
        ) from exc

    try:
        return from_string(response.text)
    except TLEError as exc:
        query = ", ".join(f"{key}={value!r}" for key, value in params.items())
        raise TLEError(
            f"CelesTrak did not return a valid TLE ({query}): {exc}"
        ) from exc


def get_tle(norad_id: int) -> TLE:
    """Download a TLE by NORAD catalog ID.

    Args:
        norad_id: Positive NORAD catalog identifier.

    Returns:
        The matching TLE.

    Raises:
        ValueError: If ``norad_id`` is not positive.
        TLEError: If the request fails or no valid TLE is returned.
    """
    if norad_id <= 0:
        raise ValueError("NORAD catalog ID must be a positive integer.")
    return _download_tle({"CATNR": str(norad_id)})


def get_tle_by_name(name: str) -> TLE:
    """Download a TLE by satellite name using the CelesTrak NAME endpoint.

    Args:
        name: Satellite name accepted by CelesTrak.

    Returns:
        The matching TLE.

    Raises:
        ValueError: If ``name`` is empty.
        TLEError: If the request fails or no valid TLE is returned.
    """
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("Satellite name cannot be empty.")
    return _download_tle({"NAME": normalized_name})


def save_tle(tle: TLE, filename: str | Path) -> None:
    """Save a TLE to disk in name/line1/line2 format.

    Parent directories are created when necessary.

    Args:
        tle: TLE to save.
        filename: Destination file path.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_string(tle), encoding="utf-8")


def load_tle(filename: str | Path) -> TLE:
    """Load a TLE from a file written by :func:`save_tle`.

    Args:
        filename: Source file path.

    Returns:
        The parsed TLE.

    Raises:
        TLEError: If the file cannot be read or contains an invalid TLE.
    """
    path = Path(filename)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TLEError(f"Failed to read TLE file '{path}': {exc}") from exc

    try:
        return from_string(text)
    except TLEError as exc:
        raise TLEError(f"Invalid TLE file '{path}': {exc}") from exc


def to_string(tle: TLE) -> str:
    """Serialize a TLE as three newline-terminated lines."""
    valid_tle = _validate_tle(tle)
    return f"{valid_tle.name.strip()}\n{valid_tle.line1}\n{valid_tle.line2}\n"


def from_string(text: str) -> TLE:
    """Parse a TLE from name/line1/line2 text.

    Blank leading and trailing lines are ignored. Embedded blank lines or any
    number of nonblank lines other than three are rejected.

    Args:
        text: Serialized TLE text.

    Returns:
        The parsed TLE.

    Raises:
        TLEError: If ``text`` does not contain exactly one valid TLE.
    """
    lines = text.strip().splitlines()
    if len(lines) != 3:
        raise TLEError(
            "Expected exactly three nonblank lines: name, line 1, and line 2; "
            f"received {len(lines)} line(s)."
        )
    if any(not line.strip() for line in lines):
        raise TLEError("TLE text cannot contain blank lines.")

    return _validate_tle(
        TLE(name=lines[0].strip(), line1=lines[1], line2=lines[2])
    )
