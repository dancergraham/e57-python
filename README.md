[![PyPI](https://img.shields.io/pypi/v/e57.svg)](https://pypi.org/project/e57)
![GitHub](https://img.shields.io/github/actions/workflow/status/dancergraham/e57-python/CI.yml?branch=main)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/dancergraham/e57-python/main/pyproject.toml)

# E57 Python Library

E57 is a compact, non-proprietary point cloud format that's defined by the ASTM E2807 standard. This format is widely adopted by 3D design applications.

This python library wraps the [rust e57 library](https://github.com/cry-inc/e57) to provide file reading

- [x] Proof of concept xml reading
- [x] Read e57 point coordinates to numpy array - see `read_points` method.
- [ ] Read intensity and other fields to numpy array.
- [ ] Write to e57 (format ?)

## Getting Started

`pip install e57`

```python
>>> import e57
>>> e57.read_points(r"pointcloud.e57")
array([[-23.25304444, -28.17607415, -13.44830654],
       [-23.28290139, -28.02118905, -13.44237764],
       [-23.26799723, -27.9039115 , -13.43430738],
       ...,
       [ 23.2458152 ,  25.4866642 ,  12.45043932],
       [ 23.22830673,  25.58055374,  12.49285875],
       [ 23.25270363,  25.45909652,  12.54284554]])
```

We need a tutorial - could you write one based on our tests?

## Contributing

All contributions welcome - feature requests, bug reports, documentation, sample files, tests, rust code, python code, sharing the project online / via social media, ...

## Testing

`python -m pytest`
