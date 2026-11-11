[![PyPI](https://img.shields.io/pypi/v/e57.svg)](https://pypi.org/project/e57)
![GitHub](https://img.shields.io/github/actions/workflow/status/dancergraham/e57-python/CI.yml?branch=main)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/dancergraham/e57-python/main/pyproject.toml)

# E57 Python Library

E57 is a compact, non-proprietary point cloud format that's defined by the ASTM E2807 standard. This format is widely adopted by 3D design applications.

This python library wraps the [rust e57 library](https://github.com/cry-inc/e57) to provide file reading

- [x] Proof of concept xml reading
- [x] Read e57 point coordinates to numpy array - see `read_points` method.
- [x] Read color field to numpy array.
- [x] Read intensity to numpy array.
- [ ] Read other fields to numpy array.
- [ ] Write to e57 (format ?)

## Getting Started

`pip install e57`

```python
>>> import e57
>>> pc = e57.read_points(r"pointcloud.e57")
>>> pc.points
array([[-23.25304444, -28.17607415, -13.44830654],
       [-23.28290139, -28.02118905, -13.44237764],
       [-23.26799723, -27.9039115 , -13.43430738],
       ...,
       [ 23.2458152 ,  25.4866642 ,  12.45043932],
       [ 23.22830673,  25.58055374,  12.49285875],
       [ 23.25270363,  25.45909652,  12.54284554]])
>>> pc.color
array(([0.3019608 , 0.3529412 , 0.23137255],
       [0.21176471, 0.26666668, 0.12941177],
       [0.21960784, 0.27058825, 0.13333334],
       ...,
       [0.5803922 , 0.58431375, 0.49019608],
       [0.41568628, 0.43529412, 0.33333334],
       [0.21568628, 0.25882354, 0.1254902 ]], dtype=float32)
```

We need a tutorial - could you write one based on our tests?

## Performance

These benchmarks compare `e57` 0.2.3 with [pye57](https://github.com/davidcaron/pye57) 0.4.19, which wraps the C++ libE57Format.
Each library reads a whole file into the same output: every scan merged, pose applied, invalid points dropped, xyz as float64 `(N, 3)`, plus color and intensity.

- Each read runs in a fresh process, after an untimed warm-up read that loads the file into the page cache. Times are the median of 3 runs.
- **Peak memory** is the process's peak resident memory (`ru_maxrss`).
- **Output** is the size of the returned arrays. `e57` returns color as float32 normalized to 0–1, while pye57 returns raw uint8, so its output is smaller.
- **Overhead** is peak memory divided by output size.
- pye57 returns one scan at a time, so it's shown with two merge strategies:
  - **naive**: `np.concatenate` of every scan's arrays
  - **preallocated**: allocate the merged arrays once and fill them scan by scan
- Both libraries returned identical data in every run: the same point counts, xyz within 3e-14, color within float32 rounding, and identical intensity.

**Apple Silicon Mac** (Docker, linux/arm64, Python 3.12)

| File | Reader | Time | Peak memory | Output | Overhead |
|---|---|---:|---:|---:|---:|
| Real scan, 535 MiB: 1 scan, 27.8M points | `e57` | 4.18 s | **1,087 MiB** | 1,061 MiB | 1.02× |
| | pye57, naive | 3.25 s | 2,839 MiB | 822 MiB | 3.45× |
| | pye57, preallocated | **3.20 s** | 2,843 MiB | 822 MiB | 3.46× |
| Synthetic, 910 MiB: 4 scans, 50M points | `e57` | 6.28 s | **1,933 MiB** | 1,907 MiB | 1.01× |
| | pye57, naive | 6.24 s | 3,334 MiB | 1,478 MiB | 2.26× |
| | pye57, preallocated | **5.25 s** | 2,396 MiB | 1,478 MiB | 1.62× |

**AWS r7i.8xlarge** (amd64, Intel Sapphire Rapids, 256 GiB, Python 3.12)

| File | Reader | Time | Peak memory | Output | Overhead |
|---|---|---:|---:|---:|---:|
| Synthetic, 910 MiB: 4 scans, 50M points | `e57` | 6.78 s | **1,939 MiB** | 1,907 MiB | 1.02× |
| | pye57, naive | 5.52 s | 3,318 MiB | 1,478 MiB | 2.24× |
| | pye57, preallocated | **5.20 s** | 2,377 MiB | 1,478 MiB | 1.61× |
| Synthetic, 18 GB: 20 scans, 1B points | `e57` | 132.3 s | 38,178 MiB | 38,147 MiB | 1.00× |
| | pye57, naive | 108.1 s | 60,455 MiB | 29,564 MiB | 2.04× |
| | pye57, preallocated | **101.4 s** | **33,041 MiB** | 29,564 MiB | 1.12× |

The real scan is `gargage.e57` from the [libE57 example data](http://www.libe57.org/data.html). The synthetic files were written with pye57.

What the numbers show:
- **`e57` adds almost nothing beyond its output** (1.00–1.02×). On a file with one large scan, it uses about 2.6× less memory than pye57.
- **pye57's overhead shrinks as scans get smaller relative to the file.** With preallocation, its temporary buffers only ever cover one scan. At 1B points in 20 scans, pye57 peaks below `e57`, mostly because `e57` stores color as float32.
- **pye57 is faster:** 20–25% on Intel, and slightly faster on Apple Silicon.
- **Both scale linearly** with point count.

## Contributing

All contributions welcome - feature requests, bug reports, documentation, sample files, tests, rust code, python code, sharing the project online / via social media, ...

## Testing

`python -m pytest`
