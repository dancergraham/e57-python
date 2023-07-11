import e57


def test_raw_xml():
    raw_xml = e57.raw_xml(r"testdata/bunnyFloat.e57")
    assert "<?xml version" in raw_xml
