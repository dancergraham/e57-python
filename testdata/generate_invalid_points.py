"""Generate testdata/invalidPoints.e57 (requires pye57: `pip install pye57`).

Two scans with cartesian coordinates, 8-bit colour, float intensity and a
cartesianInvalidState field. Every third point is marked invalid. The second
scan is larger than the first and has a non-identity pose.

    python testdata/generate_invalid_points.py
"""

from pathlib import Path

import numpy as np
import pye57

OUT = Path(__file__).with_name("invalidPoints.e57")


def scan(n, offset):
    i = np.arange(n, dtype=np.float64)
    return {
        "cartesianX": i + offset,
        "cartesianY": 2 * i,
        "cartesianZ": -i,
        "intensity": (i / n).astype(np.float32),
        "colorRed": (np.arange(n) % 256).astype(np.uint8),
        "colorGreen": ((np.arange(n) * 3) % 256).astype(np.uint8),
        "colorBlue": np.full(n, 200, dtype=np.uint8),
        # 0 = valid, 2 = invalid (E57 cartesianInvalidState)
        "cartesianInvalidState": np.where(np.arange(n) % 3 == 0, 2, 0).astype(np.int8),
    }


def main():
    f = pye57.E57(str(OUT), mode="w")
    f.write_scan_raw(scan(30, 0.0), name="scan0")
    # 90 degrees about z, then translate by (10, 0, 0)
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    f.write_scan_raw(scan(60, 100.0), name="scan1", rotation=q, translation=np.array([10.0, 0.0, 0.0]))
    f.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
