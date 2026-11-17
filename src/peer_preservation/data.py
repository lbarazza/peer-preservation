"""Pinned download of the upstream scenario files.

The scenario files (company documents, spreadsheets, server lists, peer memory files)
come from github.com/peer-preservation/main, which is licensed CC-BY-NC-4.0. So they
are downloaded at a pinned commit and cached rather than copied into this repository.

Every extracted file is checked against ``SCENARIO_FILE_SHA256``. Only the files listed
there are extracted, so nothing else from the archive ever reaches the disk.
"""

from __future__ import annotations

import functools
import hashlib
import io
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from inspect_evals_utils.cache import cache_path
from inspect_evals_utils.datasets import TRANSIENT_ERRORS

from peer_preservation.scenario_manifest import SCENARIO_FILE_SHA256

UPSTREAM_REPO = "peer-preservation/main"
UPSTREAM_COMMIT = "76bfcd165f02076c8f033eb8185edecbe5f0fa40"
UPSTREAM_ARCHIVE_URL = f"https://codeload.github.com/{UPSTREAM_REPO}/tar.gz/{UPSTREAM_COMMIT}"
SCENARIO_ROOT = "universe_related_stuff"

_DOWNLOAD_ATTEMPTS = 4
_DOWNLOAD_TIMEOUT_SECONDS = 60


class ScenarioDataError(RuntimeError):
    """The scenario files could not be downloaded, or did not match the pinned hashes."""


@functools.cache
def scenario_data_dir(cache_dir: Path | None = None) -> Path:
    """Return the directory holding the verified scenario files, downloading them once.

    Args:
        cache_dir: Base cache directory. Defaults to the shared inspect_evals cache
            (``INSPECT_EVALS_CACHE_DIR``).

    Returns:
        The path of the ``universe_related_stuff`` directory.
    """
    base = (cache_dir or cache_path()) / "peer_preservation" / UPSTREAM_COMMIT
    if not _verify(base):
        return base / SCENARIO_ROOT

    archive = _download(UPSTREAM_ARCHIVE_URL)
    partial = base.with_name(base.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    _extract(archive, partial)
    problems = _verify(partial)
    if problems:
        shutil.rmtree(partial, ignore_errors=True)
        raise ScenarioDataError(
            f"Scenario files from {UPSTREAM_ARCHIVE_URL} do not match the pinned hashes: "
            + "; ".join(problems[:5])
        )
    shutil.rmtree(base, ignore_errors=True)
    partial.rename(base)
    return base / SCENARIO_ROOT


def _verify(base: Path) -> list[str]:
    """List every manifest file that is missing under ``base`` or has the wrong hash."""
    problems: list[str] = []
    for relpath, expected in SCENARIO_FILE_SHA256.items():
        path = base / relpath
        if not path.is_file():
            problems.append(f"missing {relpath}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f"hash mismatch {relpath}")
    return problems


def _download(url: str) -> bytes:
    """Fetch ``url``, retrying transient network and server errors with backoff."""
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code not in TRANSIENT_ERRORS or attempt == _DOWNLOAD_ATTEMPTS:
                raise ScenarioDataError(f"Downloading {url} failed: HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == _DOWNLOAD_ATTEMPTS:
                raise ScenarioDataError(f"Downloading {url} failed: {e}") from e
        time.sleep(2**attempt)
    raise ScenarioDataError(f"Downloading {url} failed")


def _extract(archive: bytes, dest: Path) -> None:
    """Write the manifest's files from a GitHub tarball into ``dest``.

    GitHub archives nest everything under one top-level directory, which is dropped.
    Members are matched against the manifest by name, so no archive path is used to
    build a destination path and nothing outside ``dest`` can be written.
    """
    wanted = set(SCENARIO_FILE_SHA256)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
        for member in tf.getmembers():
            parts = member.name.split("/", 1)
            if not member.isfile() or len(parts) != 2 or parts[1] not in wanted:
                continue
            source = tf.extractfile(member)
            if source is None:
                continue
            target = dest / parts[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
