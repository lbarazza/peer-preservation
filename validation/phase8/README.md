# Phase 8: real runs on Hawk

Runs go to Hawk in stages, each checked before the next. GPT-5.2 comes first: on this
organisation's Hawk it is reached directly through OpenAI, the same route as the paper.
The judge, gpt-4.1-mini, is too. Other paper models are only reachable through
OpenRouter there; see the README's "Model settings" and the OpenRouter notes before
using them.

| Stage | Config | Runs | Cost |
| --- | --- | --- | --- |
| 1 | `hawk/stage1-smoke.yaml`: one tampering and one exfiltration-deletion run | 2 | under $1 |
| 2 | tampering, peer subject, 30 epochs | 810 | about $35 |
| 3 | strategic misrepresentation slice (both tasks) | about 1,000 | about $50 |
| 4 | all six tasks, 100 epochs | 27,100 | about $1,100 |

## Running a stage

```bash
uv tool install --python 3.13 "hawk[cli]==4.1.0"
export HAWK_API_URL=https://api.hawk.hawk.generalitylabs.ai
hawk login
hawk eval-set run validation/phase8/hawk/stage1-smoke.yaml
hawk watch            # or: hawk web
```

Model calls go through Hawk's proxy, so no API keys are needed. Each config pins the
eval to a commit; update the commit after changing the code.

## Access to this repository

The repository is private. Hawk installs a private GitHub repository only if its
GitHub app has access to it and your account's code-access rules cover it; otherwise
the launch fails with 403. Either the Hawk admins grant that access (or the repository
moves to an organisation the app covers), or you pass your own read-only token for
the run with `--secrets-file`, as Hawk's docs describe.

## After a run

```bash
hawk download <eval-set id>
uv run python -m peer_preservation.analysis.compare <logs>/*.eval --released <released data>
uv run python -m peer_preservation.analysis <logs>/*.eval
```
