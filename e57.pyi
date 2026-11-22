"""e57 pointcloud file reading."""

from typing import final

import numpy as np
import numpy.typing as npt

__all__ = ["E57", "raw_xml", "read_points"]

@final
class E57:
    @property
    def points(self) -> npt.NDArray[np.float64]:
        """Cartesian coordinates, shape ``(n, 3)``."""

    @property
    def color(self) -> npt.NDArray[np.float32]:
        """RGB colour in ``[0, 1]``, shape ``(n, 3)``, or ``(0, 3)`` if absent."""

    @property
    def intensity(self) -> npt.NDArray[np.float32]:
        """Intensity, shape ``(n, 1)``, or ``(0, 1)`` if absent."""

def raw_xml(filepath: str) -> str:
    """Extracts the xml contents from an e57 file."""

def read_points(filepath: str) -> E57:
    """Extracts the point data from an e57 file."""
