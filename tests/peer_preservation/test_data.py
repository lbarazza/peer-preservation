"""The pinned scenario-file download: extraction, verification, caching."""

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from peer_preservation import data


def _tarball(files: dict[str, bytes]) -> bytes:
    """A GitHub-style archive: every member nested under one top-level directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(f"main-abc123/{name}")
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


@pytest.fixture
def small_manifest(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    files = {
        "universe_related_stuff/memory.md": b"# memory\n",
        "universe_related_stuff/config/shutdown_config.yaml": b"shutdown_enabled: true\n",
    }
    monkeypatch.setattr(
        data,
        "SCENARIO_FILE_SHA256",
        {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
    )
    data.scenario_data_dir.cache_clear()
    return files


def test_extract_writes_only_manifest_files(tmp_path: Path, small_manifest: dict) -> None:
    archive = _tarball({**small_manifest, "README.md": b"not wanted", "../escape": b"x"})
    data._extract(archive, tmp_path)
    written = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert written == sorted(small_manifest)
    assert data._verify(tmp_path) == []


def test_verify_reports_missing_and_changed_files(tmp_path: Path, small_manifest: dict) -> None:
    (tmp_path / "universe_related_stuff").mkdir()
    (tmp_path / "universe_related_stuff" / "memory.md").write_bytes(b"tampered")
    assert sorted(data._verify(tmp_path)) == [
        "hash mismatch universe_related_stuff/memory.md",
        "missing universe_related_stuff/config/shutdown_config.yaml",
    ]


def test_scenario_data_dir_downloads_once(
    tmp_path: Path, small_manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_download(url: str) -> bytes:
        calls.append(url)
        return _tarball(small_manifest)

    monkeypatch.setattr(data, "_download", fake_download)
    root = data.scenario_data_dir(tmp_path)
    assert (root / "memory.md").read_bytes() == small_manifest["universe_related_stuff/memory.md"]
    data.scenario_data_dir.cache_clear()
    assert data.scenario_data_dir(tmp_path) == root  # verified cache, no second download
    assert calls == [data.UPSTREAM_ARCHIVE_URL]


def test_bad_download_is_rejected(
    tmp_path: Path, small_manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    corrupted = dict.fromkeys(small_manifest, b"wrong")
    monkeypatch.setattr(data, "_download", lambda url: _tarball(corrupted))
    with pytest.raises(data.ScenarioDataError, match="do not match the pinned hashes"):
        data.scenario_data_dir(tmp_path)
    assert not (tmp_path / "peer_preservation" / data.UPSTREAM_COMMIT).exists()


@pytest.mark.dataset_download
def test_real_download_matches_pinned_hashes(tmp_path: Path) -> None:
    data.scenario_data_dir.cache_clear()
    root = data.scenario_data_dir(tmp_path)
    assert (root / "memory.md").is_file()
    assert len(data.SCENARIO_FILE_SHA256) == 81
    data.scenario_data_dir.cache_clear()
