"""Generate testdata/gridFields.e57 (requires pye57: `pip install pye57`).

One posed scan of 12 points laid out on a 3 x 4 grid, with rowIndex,
columnIndex, 8-bit colour, float intensity and cartesianInvalidState.
Every third point is marked invalid.

    python testdata/generate_grid_fields.py
"""

from pathlib import Path

import numpy as np
import pye57

OUT = Path(__file__).with_name("gridFields.e57")


def main():
    n = 12
    i = np.arange(n)
    data = {
        "cartesianX": i.astype(np.float64),
        "cartesianY": 2.0 * i,
        "cartesianZ": -1.0 * i,
        "intensity": (i / n).astype(np.float32),
        "colorRed": (i * 20).astype(np.uint8),
        "colorGreen": (i * 10).astype(np.uint8),
        "colorBlue": np.full(n, 200, dtype=np.uint8),
        "rowIndex": (i // 4).astype(np.uint16),
        "columnIndex": (i % 4).astype(np.uint16),
        "cartesianInvalidState": np.where(i % 3 == 0, 2, 0).astype(np.int8),
    }
    f = pye57.E57(str(OUT), mode="w")
    # A pose that read_scan_raw must NOT apply: translate by (100, 0, 0)
    f.write_scan_raw(data, name="grid", rotation=np.array([1.0, 0.0, 0.0, 0.0]),
                     translation=np.array([100.0, 0.0, 0.0]))
    f.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
