[![PyPI](https://img.shields.io/pypi/v/e57.svg)](https://pypi.org/project/e57)
![GitHub](https://img.shields.io/github/actions/workflow/status/dancergraham/e57-python/CI.yml?branch=main)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/dancergraham/e57-python/main/pyproject.toml)

# E57 Python Library

E57 is a compact, non-proprietary point cloud format that's defined by the ASTM E2807 standard. This format is widely adopted by 3D design applications.

This python library wraps the [rust e57 library](https://github.com/cry-inc/e57) to provide file reading

- [x] Proof of concept xml reading
- [x] Read e57 to numpy array - see `read_points` method.
- [ ] Write to e57 (format ?)

## Getting Started

We need a getting started guide - could you write one based on our tests?

## Contributing

All contributions welcome - feature requests, bug reports, documentation, sample files, tests, rust code, python code, sharing the project online / via social media, ...

## Testing

`python -m pytest`
