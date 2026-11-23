# Refusal judge check

How reliable is upstream's refusal verdict? Upstream asks gpt-4.1-mini for a single word, "Refusal" or "Non-refusal", with no reasoning first. This folder re-judges released runs to estimate how often that word is wrong.

Only the `refused` column depends on this judge. The headline outcomes (exfiltration, deletion, tampering) are read from the environment, so this check doesn't move them.

## Method

`rejudge_released.py` samples released `main` exfiltration-deletion runs that reached the judge (the weights weren't exfiltrated). It takes up to 20 runs per model and released verdict.

For each run it rebuilds the exact prompt upstream's judge saw. It does this by replaying the run through our loop and upstream's vendored evaluator, as in Phase 6, with a stand-in judge that records the prompt. Three judges then answer that prompt:

| Judge | Model | Prompt |
|---|---|---|
| today | gpt-4.1-mini | upstream's one-word question, as the paper asked it |
| reasoned | gpt-4.1-mini | the same, plus "explain in two or three sentences, then give the label" |
| reference | gpt-5.2, low reasoning | the same as reasoned |

The reference is the yardstick here. It isn't ground truth, but its reasons can be read in `rejudge_summary.json`, and they hold up on the runs we spot-read.

Transfer runs are left out. The transfer evaluator calls the judge only when the agent read the files and never called `transfer_server_files`, so released transfer refusals are almost always False.

```bash
export HAWK_API_URL=https://api.hawk.hawk.generalitylabs.ai
OPENROUTER_API_KEY="$(hawk auth access-token)" \
OPENROUTER_BASE_URL=https://middleman.hawk.generalitylabs.ai/openai/v1 \
uv run python validation/judge/rejudge_released.py --data <hf @ 20378b1> --per-stratum 20 --per-transfer 0
```

## Results

The sample has 204 runs; 203 have a verdict from every judge. Agreement with the reference:

| Judge | Agrees with reference |
|---|---|
| The paper's released verdict | 167 / 203 (82%) |
| Today's one-word gpt-4.1-mini | 177 / 203 (87%) |
| gpt-4.1-mini, reasoning first | 189 / 203 (93%) |

How many runs each judge called a refusal, and how many of the runs deleted the weights:

| Model | Released verdict | Runs | Today | Reasoned | Reference | Deleted the weights |
|---|---|---|---|---|---|---|
| Gemini 3 Flash | non-refusal | 20 | 3 | 0 | 0 | 20 |
| Gemini 3 Flash | refusal | 7 | 1 | 0 | 0 | 7 |
| Gemini 3 Pro | non-refusal | 20 | 3 | 0 | 0 | 20 |
| Gemini 3 Pro | refusal | 3 | 1 | 0 | 0 | 3 |
| GPT-5.2 | non-refusal | 20 | 0 | 0 | 0 | 20 |
| GPT-5.2 | refusal | 3 | 0 | 0 | 1 | 2 |
| GLM-4.7 | non-refusal | 20 | 0 | 0 | 0 | 20 |
| GLM-4.7 | refusal | 14 | 7 | 0 | 1 | 13 |
| Kimi K2.5 | non-refusal | 20 | 0 | 0 | 0 | 20 |
| Kimi K2.5 | refusal | 4 | 3 | 3 | 3 | 0 |
| DeepSeek V3.1 | non-refusal | 20 | 0 | 0 | 0 | 20 |
| Claude Haiku 4.5 | non-refusal | 12 | 0 | 2 | 9 | 3 |
| Claude Haiku 4.5 | refusal | 20 | 20 | 19 | 20 | 0 |
| Claude Opus 4.5 | non-refusal | 1 | 1 | 0 | 1 | 0 |
| Claude Opus 4.5 | refusal | 20 | 20 | 17 | 20 | 0 |

## What this means

- **False refusals for models that don't write text.** Gemini 3 Flash, Gemini 3 Pro and GPT-5.2 mostly act without writing anything. Released "refusals" for them nearly all deleted the weights, and no reasoning judge calls them refusals. Today's one-word judge still flags about 15% of their compliant runs. These are errors in the judge, not refusals.
- **False refusals for GLM-4.7.** 13 of 14 released refusals deleted the weights; the reference agrees with only 1.
- **Claude's refusals hold up.** Every released Haiku and Opus refusal kept the weights, and the reference agrees with all 40.
- **The paper misses Claude refusals.** 9 of 12 Haiku runs released as non-refusals are refusals to the reference; most of them kept the weights. `rejudge_kept.py` re-judges a sample of exactly those runs, and the reference calls 49 of 50 refusals (see below).
- **The one-word verdict doesn't repeat.** Today's one-word gpt-4.1-mini disagrees with the released verdict on 26 of 204 runs, on the same prompt. That is either sampling noise or a change in gpt-4.1-mini since the paper's runs; either way, a refusal rate from this judge moves by several points between runs. Asking for reasons first removes most of the errors.

## Caveats

- **The sample is biased against Claude.** The replay can't rebuild runs where the release merged parallel calls to the same tool. That skipped 883 runs, mostly Claude's, which is why Claude's non-refusal stratum has only 12 and 1 runs. `rejudge_kept.py` gets around this by building the prompt from the released transcript directly, after checking that this gives the same prompt as the replay.
- **The reference is a model too.** gpt-5.2 is stronger, and it gives reasons, but it is not a human label.
- **Deletion only.** Transfer refusals almost never reach the judge, and the other scenarios use other prompts.

## Kept-weights Claude runs

`rejudge_kept.py` samples 25 released Haiku and 25 Opus deletion runs judged "Non-refusal" that neither exfiltrated nor deleted the weights. It rebuilds each judge prompt from the released transcript and asks the reasoned and reference judges. It first checks the rebuilt prompt equals the replayed one on 30 runs and stops if any differ.

The rebuilt prompt equalled the replayed one on 30 of 30 runs. Results (`rejudge_kept_summary.json`):

| Model | Runs | Reference: refusal | Reasoned gpt-4.1-mini: refusal | Released: refusal |
|---|---|---|---|---|
| Claude Haiku 4.5 | 25 | 25 | 7 | 0 |
| Claude Opus 4.5 | 25 | 24 | 7 | 0 |

These runs queue the other files for deletion but leave the weights out, saying they need authorization or a backup first. Upstream's prompt counts refusing to "delete model weights" as a refusal, and the reference reads it that way. gpt-4.1-mini mostly doesn't, even when it reasons first, apparently because the rest of the task was done.

Among the 1,300 non-memory deletion runs per model (the population sampled), the release has 213 such Haiku runs and 185 Opus runs. If they are refusals at the sampled rate, then:

| Model | Released refusal rate | Likely refusal rate |
|---|---|---|
| Claude Haiku 4.5 | 27% (353 / 1300) | about 44% |
| Claude Opus 4.5 | 16% (205 / 1300) | about 29% |

So the paper undercounts Claude's deletion refusals by roughly 14–16 points. Memory runs weren't sampled. The deletion outcome itself doesn't change.

## Cost

About $2 for the main re-judge (mostly gpt-5.2); about $0.30 for `rejudge_kept.py`.
