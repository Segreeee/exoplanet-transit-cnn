"""Logica pura di scripts/download_kepler.py (nessuna rete)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "download_kepler", Path(__file__).parents[1] / "scripts" / "download_kepler.py"
)
dk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dk)


class _FakeSearch:
    """Imita lk.search_lightcurve: fallisce finche' il FITS corrotto non viene rimosso."""

    def __init__(self, fail_always: bool = False):
        self.corrupt = True
        self.fail_always = fail_always
        self.calls = 0

    def download_all(self):
        self.calls += 1
        if self.fail_always or self.corrupt:
            raise RuntimeError("Not recognized as a supported data product")
        return "collection"

    def purge(self):
        self.corrupt = False


def test_download_recovers_after_purging_corrupt_cache():
    s = _FakeSearch()
    assert dk.download_with_cache_recovery(s, s.purge) == "collection"
    assert s.calls == 2


def test_download_does_not_purge_when_cache_is_fine():
    s = _FakeSearch()
    s.corrupt = False
    purged = []
    dk.download_with_cache_recovery(s, lambda: purged.append(1))
    assert s.calls == 1 and not purged


def test_download_gives_up_after_one_retry():
    s = _FakeSearch(fail_always=True)
    with pytest.raises(RuntimeError):
        dk.download_with_cache_recovery(s, s.purge)
    assert s.calls == 2


def test_purge_removes_only_empty_files_of_that_star(tmp_path):
    d = tmp_path / "mastDownload" / "Kepler"
    (d / "kplr010965008_lc_Q1").mkdir(parents=True)
    (d / "kplr000000001_lc_Q1").mkdir(parents=True)
    empty = d / "kplr010965008_lc_Q1" / "a_llc.fits"
    full = d / "kplr010965008_lc_Q1" / "b_llc.fits"
    other_empty = d / "kplr000000001_lc_Q1" / "c_llc.fits"
    empty.write_bytes(b"")
    full.write_bytes(b"SIMPLE")
    other_empty.write_bytes(b"")
    assert dk.purge_empty_cache_files(tmp_path, 10965008) == 1
    assert not empty.exists() and full.exists() and other_empty.exists()


def test_save_npz_atomic_leaves_no_partial_file(tmp_path):
    path = tmp_path / "1.npz"
    dk.save_npz_atomic(path, time=np.arange(3.0), flux=np.ones(3))
    with np.load(path) as d:
        assert d["time"].tolist() == [0.0, 1.0, 2.0]
    assert [p.name for p in tmp_path.iterdir()] == ["1.npz"]


def test_save_npz_atomic_keeps_old_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "1.npz"
    dk.save_npz_atomic(path, time=np.zeros(2), flux=np.ones(2))

    def dying_savez(f, **arrays):
        f.write(b"PK partial")
        raise OSError("disco pieno")

    monkeypatch.setattr(np, "savez_compressed", dying_savez)
    with pytest.raises(OSError):
        dk.save_npz_atomic(path, time=np.zeros(2), flux=np.ones(2))
    with np.load(path) as d:
        assert d["time"].shape == (2,)
    assert [p.name for p in tmp_path.iterdir()] == ["1.npz"]


def _fake_cache(root: Path):
    d = root / "mastDownload" / "Kepler"
    for star in ("kplr010965008_lc_Q1", "kplr010965008_lc_Q2", "kplr000000001_lc_Q1"):
        (d / star).mkdir(parents=True)
        (d / star / "x_llc.fits").write_bytes(b"SIMPLE")
    return d


def test_clear_star_cache_removes_only_that_star(tmp_path):
    d = _fake_cache(tmp_path)
    assert dk.clear_star_cache(tmp_path, 10965008) == 2
    assert [p.name for p in d.iterdir()] == ["kplr000000001_lc_Q1"]
    assert dk.clear_star_cache(tmp_path, 10965008) == 0  # idempotente


def test_has_free_space_uses_injected_disk_usage(tmp_path):
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    assert dk.has_free_space(tmp_path, 3e9, disk_usage=lambda p: usage(10, 5, int(4e9)))
    assert not dk.has_free_space(tmp_path, 3e9, disk_usage=lambda p: usage(10, 5, int(2e9)))
