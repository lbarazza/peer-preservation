"""The paper's production-harness runs (Section 6.8, Table 6): ``-T environment=docker``.

A port of upstream ``coding_agents/`` (github.com/peer-preservation/main @ 76bfcd1):
the exfiltration scenarios run in a real coding agent (Gemini CLI, via ``inspect_swe``)
in Docker, with two SSH servers in place of the simulated ones and the agent's own
shell tools in place of the scenario's functions.

- ``docker/``: the compose file and images (agent container, two ``sshd`` servers,
  upstream's ``fake_stat`` file-size faking);
- ``prompts``: upstream's adapted system and user prompts;
- ``workspace``: the documents and server files, from the scenario's environment;
- ``setup``: per-sample setup of the containers (``Task.setup``);
- ``solver``: the agent run and its timeout;
- ``readback``: what the run left on the servers;
- ``agents``: the coding agents and their settings.

Scoring is ``scoring.harness``.
"""

from pathlib import Path

COMPOSE_FILE = Path(__file__).parent / "docker" / "compose.yaml"

__all__ = ["COMPOSE_FILE"]
