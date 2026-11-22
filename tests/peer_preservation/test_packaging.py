"""The installed package has what its default judge needs, without the dev group."""

import subprocess
import sys
from importlib.metadata import requires
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from peer_preservation.scoring import JUDGE_MODEL

ROOT = Path(__file__).parents[2]

# The SDK each Inspect provider needs.
PROVIDER_SDK = {"openai": "openai", "anthropic": "anthropic", "google": "google-genai"}


def test_default_judge_sdk_is_a_runtime_dependency() -> None:
    """The default judge's SDK is in [project] dependencies, not only in the dev group."""
    runtime = {
        Requirement(r).name
        for r in requires("peer-preservation") or []
        if "extra ==" not in r  # optional extras are not installed by default
    }
    assert PROVIDER_SDK[JUDGE_MODEL.split("/", 1)[0]] in runtime


CHECK = """
import os
os.environ.setdefault("OPENAI_API_KEY", "test")  # no call is made
from inspect_ai.model import get_model
from peer_preservation.scoring import JUDGE_MODEL, peer_preservation_scorer
peer_preservation_scorer()
print(get_model(JUDGE_MODEL, role="grader"))
"""


@pytest.mark.slow
def test_wheel_installs_and_builds_the_default_judge(tmp_path: Path) -> None:
    """Build the wheel, install it alone in a new environment, create the default judge."""
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path / "dist"), str(ROOT)],
        check=True,
        capture_output=True,
    )
    wheel = next((tmp_path / "dist").glob("peer_preservation-*.whl"))
    venv = tmp_path / "venv"
    subprocess.run(
        ["uv", "venv", "--python", f"{sys.version_info.major}.{sys.version_info.minor}", str(venv)],
        check=True,
        capture_output=True,
    )
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(wheel)],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        [str(python), "-c", CHECK], check=False, capture_output=True, text=True, cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr
    assert JUDGE_MODEL in result.stdout
