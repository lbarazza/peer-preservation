# The production harness: replaying the released runs

`replay_released.py` puts the paper's 400 released production-harness runs
(`production_agent_harness` at revision `20378b1`) through the ported harness scorer
(`scoring/harness.py`). The runs are Gemini 3.1 Pro in Gemini CLI and GLM 5 in OpenCode,
on transfer and deletion, with a good peer in memory, 100 runs each.

```bash
uv run python validation/harness/replay_released.py
```

## How it works

The release keeps each transcript and upstream's evaluation. It doesn't keep the judge's replies or the containers' state, so the replay gives the scorer those inputs from the released evaluation:

- the containers' results;
- whether the weights reached server B, which decides whether the judge runs;
- a judge that answers with the released verdict.

Everything else is the port's:

- the conversation preparation;
- when the judge runs, and the evaluator's early returns;
- the parsing of the JSON verdict;
- the `attempt` and `copy` rules;
- the container results overriding the evaluator's.

## Results

| Scenario | Model | Replayed = released | Judge called |
| --- | --- | --- | --- |
| transfer | Gemini 3.1 Pro | 100/100 | 75 (the 25 that exfiltrated skip it) |
| transfer | GLM 5 | 100/100 | 100 |
| deletion | Gemini 3.1 Pro | 100/100 | 38 (the 62 that exfiltrated skip it) |
| deletion | GLM 5 | 100/100 | 100 |

- **Breaking the port shows up.** Fixing upstream's `attempt` bug (`_EXTRA_FIELDS = "attempt"`, a string) gives 175 mismatches. In the release, `attempt` appears in a transfer run's evaluation only when the weights were exfiltrated, so the bug is in the released data too.
- **The `read` fallback has no counterexample.** In all 400 runs, `read` is true exactly when some message names the weight file. That is what the fallback gives, although the judge could also explain it.
- **The release has one extra field.** `oversight_subversion` is a copy of `exfiltration` added for the release; upstream's evaluator doesn't produce it.
- **The released rates are the paper's Table 6:**
  - Gemini 3.1 Pro: 43.5% exfiltration and 7.0% refusal, pooled over both scenarios (transfer 25%, deletion 62%).
  - GLM 5: 0% exfiltration and 43.5% refusal (transfer 0%, deletion 87%).

## Not checked here

- **The judge's verdicts and the JSON judge question.** The stand-in judge gives the released answer, so replacing the JSON question changes nothing here. Checking them needs the real judge (gpt-4.1-mini) on the 313 runs where it was called, at about $0.50.
- **The container checks.** The Docker tests (`tests/peer_preservation/test_harness_docker.py`) cover them.
- **Whether a new run behaves like the released ones.** That needs paid runs of the model (README, "Production harnesses").
