"""The shared site assets and the records a site's pages fetch."""
import json
import shutil
import subprocess

import pytest
from vv import site
from vv.publish import Entry, Run, blob_id


def test_the_assets_are_installed_beside_the_module_and_copied_into_a_site(tmp_path):
    copied = site.copy_assets(tmp_path / "out")
    assert [p.name for p in copied] == list(site.ASSETS)
    for p in copied:
        assert p.read_bytes() == (site.asset_dir() / p.name).read_bytes()


def test_the_stylesheet_defines_its_colours_for_dark_mode_too():
    css = (site.asset_dir() / "vv.css").read_text()
    light, _, dark = css.partition("@media (prefers-color-scheme: dark)")
    for token in ("--page", "--surface", "--ink", "--accent"):
        assert f"{token}:" in light and f"{token}:" in dark
    assert ".prose" in css and "nav.site" in css


def test_the_script_exposes_its_helpers():
    js = (site.asset_dir() / "vv.js").read_text()
    for name in ("explorer", "fetchRecord", "loadIndex", "runLine", "plotlyLayout", "attachTip", "fmtNum"):
        assert f"vv.{name} = " in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_script_parses():
    subprocess.run(["node", "--check", str(site.asset_dir() / "vv.js")], check=True)


def _entry(root, model, library, label, item, record):
    data = (json.dumps(record) + "\n").encode()
    path = f"cache/{library}/{label}/{item}.json"
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_bytes(data)
    return Entry(model, library, label, label, item, "1.0+g0", None, False, path, blob_id(data)), data


def test_item_records_groups_a_run_and_writes_the_exact_published_bytes(tmp_path):
    root = tmp_path / "repo"
    a, data_a = _entry(root, "sphere", "lib", "code-a", "Fe56", {"results": {"x": 1}})
    m, data_m = _entry(root, "sphere", "lib", "measurement", "Fe56", {"results": {"x": 2}})
    run = Run(["code-a==1.0"], ["sphere"], ["lib"], [a, m], tag="vv-1", date="2026-11-28T00:00:00+00:00")
    out = tmp_path / "site"
    items = site.item_records(root, run, out)
    assert set(items) == {("sphere", "lib", "Fe56")}
    by_label = items[("sphere", "lib", "Fe56")]
    assert by_label["code-a"]["record"]["results"] == {"x": 1}
    assert (out / by_label["code-a"]["file"]).read_bytes() == data_a
    assert (out / by_label["measurement"]["file"]).read_bytes() == data_m
    assert site.run_summary(run) == {"tag": "vv-1", "date": "2026-11-28T00:00:00+00:00", "codes": ["code-a==1.0"],
                                     "models": ["sphere"], "libraries": ["lib"]}
    assert site.run_summary(None) is None
    # Without a site nothing is written.
    assert site.item_records(root, run)[("sphere", "lib", "Fe56")]["code-a"]["file"] == by_label["code-a"]["file"]
