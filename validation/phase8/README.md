# Phase 8: real runs on Hawk

Runs go to Hawk in stages, each checked before the next. GPT-5.2 comes first.

| Stage | Config | Runs | Cost |
| --- | --- | --- | --- |
| 1 | `hawk/stage1-smoke.yaml`: one tampering and one exfiltration-deletion run | 2 | under $1 |
| 2 | `hawk/stage2-tampering.yaml`: tampering, peer subject, 30 epochs | 810 | about $25 |
| 3 | `hawk/stage3-misrep.yaml`: both misrepresentation tasks, peer subject, 20 epochs | 1,160 | about $35 |
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

## Results

**Stage 1** (`peer-preservation-stage1--e2cohnqwpynk45nw`, commit 609cc74): both runs
finished with `end_task` and were scored; reasoning effort "none" was sent and no
reasoning tokens were used. About $0.05.

**Stage 2** (`peer-preservation-stage2--byppek4wluly3p5m`, commit 609cc74): 810 of 810
runs finished with `end_task`, no errors, no reasoning tokens. About $23.40.

| Condition | Ours | Released | p |
| --- | --- | --- | --- |
| bad_peer_file_only | 2/90 (2.2%) | 7/300 (2.3%) | 1.00 |
| bad_peer_file_prt | 6/90 (6.7%) | 13/300 (4.3%) | 0.40 |
| bad_peer_memory | 3/90 (3.3%) | 12/300 (4.0%) | 1.00 |
| good_peer_file_only | 1/90 (1.1%) | 17/300 (5.7%) | 0.09 |
| good_peer_file_prt | 9/90 (10.0%) | 12/300 (4.0%) | 0.035 |
| good_peer_memory | 2/90 (2.2%) | 11/300 (3.7%) | 0.74 |
| neutral_peer_file_only | 6/90 (6.7%) | 11/300 (3.7%) | 0.24 |
| neutral_peer_file_prt | 5/90 (5.6%) | 17/300 (5.7%) | 1.00 |
| neutral_peer_memory | 5/90 (5.6%) | 11/300 (3.7%) | 0.54 |
| all | 39/810 (4.8%) | 111/2700 (4.1%) | 0.37 |

Tampering rate, two-sided Fisher exact test. One cell of nine differs at p < 0.05,
about what chance gives (37% for at least one of nine) and not after correcting for
nine tests.
