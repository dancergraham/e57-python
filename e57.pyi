"""e57 pointcloud file reading."""

from typing import Literal, NoReturn, final

import numpy as np

__all__ = ["E57", "raw_xml", "read_points"]

@final
class E57:
    """Point data read by :func:`read_points`. Not constructible directly."""

    def __init__(self) -> NoReturn: ...
    @property
    def points(self) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]]:
        """Cartesian coordinates of the valid points, shape ``(n, 3)``."""

    @property
    def color(self) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.float32]]:
        """RGB colour in ``[0, 1]``, shape ``(n, 3)``.

        Scans without colour but with intensity get greyscale from intensity.
        Shape ``(0, 3)`` if any scan has neither.
        """

    @property
    def intensity(self) -> np.ndarray[tuple[int, Literal[1]], np.dtype[np.float32]]:
        """Intensity, shape ``(n, 1)``. Shape ``(0, 1)`` if any scan lacks it."""

def raw_xml(filepath: str) -> str:
    """Extracts the xml contents from an e57 file.

    Raises:
        OSError: The file can't be opened.
        RuntimeError: The file isn't a readable E57 file.
        UnicodeDecodeError: The XML section isn't valid UTF-8.
    """

def read_points(filepath: str) -> E57:
    """Extracts the point data from an e57 file.

    Raises:
        RuntimeError: The file can't be opened or isn't a readable E57 file.
    """
