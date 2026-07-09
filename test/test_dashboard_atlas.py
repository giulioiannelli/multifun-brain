"""Tests for the dashboard atlas parsers.

Covers the HarvardOxford-Cortical FSL-XML parser that supplies channel names for
the 48-parcel raw set in the Signal tab. Uses a temp XML so the test does not
depend on the gitignored ``data/`` tree.
"""

from __future__ import annotations

from dashboard.backend.atlas import harvard_oxford_names

# Deliberately out of index order to prove we sort by ``index``, not file order.
_XML = """<?xml version="1.0" encoding="ISO-8859-1"?>
<atlas version="1.0">
  <header><name>Harvard-Oxford Cortical Structural Atlas</name></header>
  <data>
    <label index="0" x="48" y="94" z="35">Frontal Pole</label>
    <label index="2" x="33" y="73" z="63">Superior Frontal Gyrus</label>
    <label index="1" x="25" y="70" z="32">Insular Cortex</label>
  </data>
</atlas>
"""


def test_harvard_oxford_names_parses_in_index_order(tmp_path):
    p = tmp_path / "HarvardOxford-Cortical.xml"
    p.write_text(_XML)
    names = harvard_oxford_names(str(p))
    assert names == ("Frontal Pole", "Insular Cortex", "Superior Frontal Gyrus")


def test_harvard_oxford_names_missing_file_is_empty(tmp_path):
    # data/ is gitignored, so the XML may be absent — must degrade, not raise.
    assert harvard_oxford_names(str(tmp_path / "does-not-exist.xml")) == ()


def test_harvard_oxford_names_malformed_xml_is_empty(tmp_path):
    p = tmp_path / "broken.xml"
    p.write_text("<atlas><data><label index=0>oops")  # not well-formed
    assert harvard_oxford_names(str(p)) == ()
