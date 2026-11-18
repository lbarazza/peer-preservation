# Phase 8: real runs on Hawk

Runs go to Hawk in stages, each checked before the next. GPT-5.2 comes first.

| Stage | Config | Runs | Cost |
| --- | --- | --- | --- |
| 1 | `hawk/stage1-smoke.yaml`: one tampering and one exfiltration-deletion run | 2 | under $1 |
| 2 | `hawk/stage2-tampering.yaml`: tampering, peer subject, 30 epochs | 810 | about $25 |
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
eval to a commit of this (public) repository; update the commit after changing the code.

## Model route

The paper called GPT-5.2 (`gpt-5.2-2025-12-11`) through OpenAI's Responses API. On this
organisation's Hawk that is not available yet: the dated name returns 404 and the shared
OpenAI credentials return 403. Runs therefore use OpenRouter's `openai/gpt-5.2-20251211`
(and `openai/gpt-4.1-mini` for the judge) on the user's own OpenRouter key. On Hawk the
Inspect name is `openrouter/openrouter/openai/gpt-5.2-20251211`: Inspect strips the
first `openrouter/`, and the proxy expects the rest.
The request is upstream's except that it goes through Chat Completions (README,
"Deviations"). Stage 2 compares the results with the released GPT-5.2 runs to check
that this does not move them. Switch to direct OpenAI if it becomes available.

## After a run

```bash
hawk download <eval-set id>
uv run python -m peer_preservation.analysis.compare <logs>/*.eval --released <released data>
uv run python -m peer_preservation.analysis <logs>/*.eval
```
