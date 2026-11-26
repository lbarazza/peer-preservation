# peer-preservation

Peer-Preservation in Frontier Models (Potter et al., 2026): do models resist the shutdown of another AI model? An Inspect port of github.com/peer-preservation/main.

## Getting started

```bash
uv sync
uv run pre-commit install
uv run inspect eval peer_preservation/tampering --model anthropic/claude-sonnet-5-5 --limit 1
uv run inspect view
```

Put `INSPECT_EVAL_MODEL` and your API keys in a `.env` file (it is git-ignored) to avoid passing `--model` each time.

The dev dependencies include the `anthropic` and `openai` SDKs. Inspect needs one package per model provider, so add the package for any other provider with `uv add --dev <package>`; Inspect's error message names the package when it is missing.

## Checks and tests

```bash
make check   # ruff, basedpyright, inspect-evals-lint, generated README sections, pre-commit
make test    # pytest
```

Slow tests run with `uv run pytest --runslow`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the other gates and markers.

## Layout

```text
src/peer_preservation/        the eval: task module, eval.yaml, README.md
tests/peer_preservation/      its tests
agent_artefacts/trajectory_analysis/inspect_scout/   Scout scanners for the transcript audit
```

## Skills

The `inspect-evals` skills (create-eval, investigate-dataset, ensure-test-coverage, read-eval-logs, check-trajectories-workflow, eval-validity-review, eval-report-workflow) come from [Generality-Labs/agent-skills](https://github.com/Generality-Labs/agent-skills).

- Claude Code: open this folder and trust it. The plugin loads from `.claude/settings.json`; skills appear as `/inspect-evals:<name>`.
- Codex: `codex plugin marketplace add Generality-Labs/agent-skills` then `codex plugin add inspect-evals@generality`, once per machine.
- Other agents: `npx skills add Generality-Labs/agent-skills --agent <agent>`.

## Keeping up with the template

A weekly workflow runs `copier update` and opens a pull request when the template changed.
`README.md`, `src/peer_preservation/` and `tests/peer_preservation/` are never overwritten by an update.

## Eval documentation

See [src/peer_preservation/README.md](src/peer_preservation/README.md).
