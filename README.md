# E57 Python Library

E57 is a compact, non-proprietary point cloud format that's defined by the ASTM E2807 standard. This format is widely adopted by 3D design applications.

This python library wraps the [rust e57 library](https://github.com/cry-inc/e57) to provide file reading

- [x] Proof of concept xml reading
- [ ] Read e57 to python dict
- [ ] Read e57 to numpy array - work in progress - see `read_points` method.
- [ ] Write to e57 (format ?)

## Testing

`python -m pytest`
