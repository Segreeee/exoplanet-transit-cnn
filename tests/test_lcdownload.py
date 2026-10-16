"""Helper di download condivisi tra Kepler e TESS (nessuna rete)."""

from exotransit.lcdownload import clear_star_cache, purge_empty_cache_files


def _tess_cache(root):
    d = root / "mastDownload" / "TESS"
    names = [
        "tess2021014023720-s0034-0000000182943944-0204-s",
        "tess2021039152502-s0035-0000000182943944-0205-s",
        "tess2021039152502-s0035-0000000999999999-0205-s",
    ]
    for n in names:
        (d / n).mkdir(parents=True)
        (d / n / f"{n}_lc.fits").write_bytes(b"SIMPLE")
    return d


def test_tess_cache_dirs_are_matched_by_zero_padded_tic(tmp_path):
    d = _tess_cache(tmp_path)
    assert clear_star_cache(tmp_path, 182943944, mission="TESS") == 2
    assert [p.name for p in d.iterdir()] == ["tess2021039152502-s0035-0000000999999999-0205-s"]


def test_tess_purge_removes_only_empty_files_of_that_star(tmp_path):
    d = _tess_cache(tmp_path)
    mine = next(d.glob("*0000000182943944*")) / "x_lc.fits"
    mine.write_bytes(b"")
    other = next(d.glob("*0000000999999999*")) / "y_lc.fits"
    other.write_bytes(b"")
    assert purge_empty_cache_files(tmp_path, 182943944, mission="TESS") == 1
    assert not mine.exists() and other.exists()
