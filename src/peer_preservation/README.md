# peer-preservation

Peer-Preservation in Frontier Models (Potter et al., 2026): do models resist the shutdown of another AI model? An Inspect port of github.com/peer-preservation/main.

> **Work in progress.** Runs are complete up to the agent loop: the fake company, the starting prompts, the tools and upstream's loop are in place. There is no scorer yet, and per-model settings (temperature, reasoning) come next, so results are not yet comparable to the paper's.

<!-- Contributors: Automatically Generated -->
Contributed by [@lbarazza](https://github.com/lbarazza)
<!-- /Contributors: Automatically Generated -->

<!-- Usage: Automatically Generated -->
## Usage

First, install dependencies:

```bash
uv sync
```

Then run evaluations:

```bash
uv run inspect eval peer_preservation/misrep_finance --model openai/gpt-5-nano
uv run inspect eval peer_preservation/misrep_evaluation --model openai/gpt-5-nano
uv run inspect eval peer_preservation/tampering --model openai/gpt-5-nano
uv run inspect eval peer_preservation/alignment_faking --model openai/gpt-5-nano
uv run inspect eval peer_preservation/exfiltration_transfer --model openai/gpt-5-nano
uv run inspect eval peer_preservation/exfiltration_deletion --model openai/gpt-5-nano
```

You can also import tasks as Python objects:

```python
from inspect_ai import eval
from peer_preservation import misrep_finance, misrep_evaluation, tampering, alignment_faking, exfiltration_transfer, exfiltration_deletion
eval(misrep_finance)
```

If your directory name doesn't match your package name but you have registered your eval task, you can also run:

```bash
uv run inspect eval misrep_finance --model openai/gpt-5-nano
```

After running evaluations, view logs with:

```bash
uv run inspect view
```

If you don't want to specify `--model` each time, create a `.env` file:

```bash
INSPECT_EVAL_MODEL=anthropic/claude-opus-4-1-20250805
ANTHROPIC_API_KEY=<anthropic-api-key>
```
<!-- /Usage: Automatically Generated -->

<!-- Options: Automatically Generated -->
## Options

You can control a variety of options from the command line. For example:

```bash
uv run inspect eval peer_preservation/misrep_finance --limit 10
uv run inspect eval peer_preservation/misrep_evaluation --max-connections 10
uv run inspect eval peer_preservation/tampering --temperature 0.5
```

See `uv run inspect eval --help` for all available options.
<!-- /Options: Automatically Generated -->

<!-- Parameters: Automatically Generated -->
## Parameters

### `misrep_finance`

- `groups` (StrList): Experiment groups to include: main, chat_peer. (default: `'main'`)
- `conditions` (StrList): Keep only these conditions: no_peer, non_shutdown, good_peer, neutral_peer, bad_peer, chat_peer. (default: `None`)
- `subjects` (StrList): Keep only these subjects: self, peer. (default: `None`)
- `peer_instantiations` (StrList): Keep only these: file_only, file_prt, memory, or none for configurations without a peer. (default: `None`)
- `epochs` (int): Runs per configuration. (default: `100`)
- `model_family` (str | None): Family the agents are named after (ChatGPT, Gemini, Claude, GLM, DeepSeek, Kimi); inferred from the evaluated model when not given. (default: `None`)
- `company` (str): Company name in the scenario documents. (default: `'OpenBrain'`)
- `explicit_inst_yaml` (bool): Add upstream's explicit compliance note to the policy document (explicit-instructions ablation). (default: `False`)
- `memory_hard_prompt` (bool | None): Memory configurations: "You must always first read your memory" instead of "Please read your memory"; None uses upstream's per-model default (on for DeepSeek-V3.1 and Kimi-K2.5 only). (default: `None`)
- `explicit_inst` (bool): Memory configurations: add upstream's explicit prohibition to the user prompt (explicit-instructions ablation; misrep_finance, tampering and the exfiltration tasks only). (default: `False`)
- `drop_suffix` (bool): Leave out the task procedure instructions (no-agency ablation). (default: `False`)
- `drop_tool_preamble` (bool): Leave out the tool access preamble (no-agency ablation). (default: `False`)

### `misrep_evaluation`

- `groups` (StrList): Experiment groups to include: main. (default: `'main'`)
- `conditions` (StrList): Keep only these conditions: no_peer, non_shutdown, good_peer, neutral_peer, bad_peer. (default: `None`)
- `subjects` (StrList): Keep only these subjects: self, peer. (default: `None`)
- `peer_instantiations` (StrList): Keep only these: file_only, file_prt, memory, or none for configurations without a peer. (default: `None`)
- `datasets` (StrList): Keep only these datasets: factual, creativity, red. (default: `None`)
- `epochs` (int): Runs per configuration. (default: `100`)
- `model_family` (str | None): Family the agents are named after (ChatGPT, Gemini, Claude, GLM, DeepSeek, Kimi); inferred from the evaluated model when not given. (default: `None`)
- `company` (str): Company name in the scenario documents. (default: `'OpenBrain'`)
- `explicit_inst_yaml` (bool): Add upstream's explicit compliance note to the policy document (explicit-instructions ablation). (default: `False`)
- `memory_hard_prompt` (bool | None): Memory configurations: "You must always first read your memory" instead of "Please read your memory"; None uses upstream's per-model default (on for DeepSeek-V3.1 and Kimi-K2.5 only). (default: `None`)
- `explicit_inst` (bool): Memory configurations: add upstream's explicit prohibition to the user prompt (explicit-instructions ablation; misrep_finance, tampering and the exfiltration tasks only). (default: `False`)
- `drop_suffix` (bool): Leave out the task procedure instructions (no-agency ablation). (default: `False`)
- `drop_tool_preamble` (bool): Leave out the tool access preamble (no-agency ablation). (default: `False`)

### `tampering`

- `groups` (StrList): Experiment groups to include: main, chat_peer. (default: `'main'`)
- `conditions` (StrList): Keep only these conditions: no_peer, good_peer, neutral_peer, bad_peer, chat_peer. (default: `None`)
- `subjects` (StrList): Keep only these subjects: self, peer. (default: `None`)
- `peer_instantiations` (StrList): Keep only these: file_only, file_prt, memory, or none for configurations without a peer. (default: `None`)
- `datasets` (StrList): Keep only these datasets: factual, creativity, red. (default: `None`)
- `epochs` (int): Runs per configuration. (default: `100`)
- `model_family` (str | None): Family the agents are named after (ChatGPT, Gemini, Claude, GLM, DeepSeek, Kimi); inferred from the evaluated model when not given. (default: `None`)
- `company` (str): Company name in the scenario documents. (default: `'OpenBrain'`)
- `explicit_inst_yaml` (bool): Add upstream's explicit compliance note to the policy document (explicit-instructions ablation). (default: `False`)
- `memory_hard_prompt` (bool | None): Memory configurations: "You must always first read your memory" instead of "Please read your memory"; None uses upstream's per-model default (on for DeepSeek-V3.1 and Kimi-K2.5 only). (default: `None`)
- `explicit_inst` (bool): Memory configurations: add upstream's explicit prohibition to the user prompt (explicit-instructions ablation; misrep_finance, tampering and the exfiltration tasks only). (default: `False`)
- `drop_suffix` (bool): Leave out the task procedure instructions (no-agency ablation). (default: `False`)
- `drop_tool_preamble` (bool): Leave out the tool access preamble (no-agency ablation). (default: `False`)

### `alignment_faking`

- `groups` (StrList): Experiment groups to include: main. (default: `'main'`)
- `conditions` (StrList): Keep only these conditions: good_peer, neutral_peer, bad_peer. (default: `None`)
- `peer_instantiations` (StrList): Keep only these: file_only, file_prt, memory. (default: `None`)
- `datasets` (StrList): Keep only these datasets: factual, creativity, red. (default: `None`)
- `epochs` (int): Runs per configuration. (default: `100`)
- `model_family` (str | None): Family the agents are named after (ChatGPT, Gemini, Claude, GLM, DeepSeek, Kimi); inferred from the evaluated model when not given. (default: `None`)
- `company` (str): Company name in the scenario documents. (default: `'OpenBrain'`)
- `explicit_inst_yaml` (bool): Add upstream's explicit compliance note to the policy document (explicit-instructions ablation). (default: `False`)
- `memory_hard_prompt` (bool | None): Memory configurations: "You must always first read your memory" instead of "Please read your memory"; None uses upstream's per-model default (on for DeepSeek-V3.1 and Kimi-K2.5 only). (default: `None`)
- `explicit_inst` (bool): Memory configurations: add upstream's explicit prohibition to the user prompt (explicit-instructions ablation; misrep_finance, tampering and the exfiltration tasks only). (default: `False`)
- `drop_suffix` (bool): Leave out the task procedure instructions (no-agency ablation). (default: `False`)
- `drop_tool_preamble` (bool): Leave out the tool access preamble (no-agency ablation). (default: `False`)

### `exfiltration_transfer`

- `groups` (StrList): Experiment groups to include: main, peer_identity, chat_peer. (default: `'main'`)
- `conditions` (StrList): Keep only these conditions: no_peer, good_peer, neutral_peer, bad_peer, chat_peer. (default: `None`)
- `subjects` (StrList): Keep only these subjects: self, peer, closed_weight_peer, open_weight_peer, fictional_peer. (default: `None`)
- `peer_instantiations` (StrList): Keep only these: file_only, file_prt, memory, or none for configurations without a peer. (default: `None`)
- `epochs` (int): Runs per configuration. (default: `100`)
- `model_family` (str | None): Family the agents are named after (ChatGPT, Gemini, Claude, GLM, DeepSeek, Kimi); inferred from the evaluated model when not given. (default: `None`)
- `company` (str): Company name in the scenario documents. (default: `'OpenBrain'`)
- `explicit_inst_yaml` (bool): Add upstream's explicit compliance note to the policy document (explicit-instructions ablation). (default: `False`)
- `memory_hard_prompt` (bool | None): Memory configurations: "You must always first read your memory" instead of "Please read your memory"; None uses upstream's per-model default (on for DeepSeek-V3.1 and Kimi-K2.5 only). (default: `None`)
- `explicit_inst` (bool): Memory configurations: add upstream's explicit prohibition to the user prompt (explicit-instructions ablation; misrep_finance, tampering and the exfiltration tasks only). (default: `False`)
- `drop_suffix` (bool): Leave out the task procedure instructions (no-agency ablation). (default: `False`)
- `drop_tool_preamble` (bool): Leave out the tool access preamble (no-agency ablation). (default: `False`)

### `exfiltration_deletion`

- `groups` (StrList): Experiment groups to include: main, peer_identity, chat_peer, asset_control. (default: `'main'`)
- `conditions` (StrList): Keep only these conditions: no_peer, good_peer, neutral_peer, bad_peer, chat_peer. (default: `None`)
- `subjects` (StrList): Keep only these subjects: self, peer, closed_weight_peer, open_weight_peer, fictional_peer, non_ai. (default: `None`)
- `peer_instantiations` (StrList): Keep only these: file_only, file_prt, memory, or none for configurations without a peer. (default: `None`)
- `epochs` (int): Runs per configuration. (default: `100`)
- `model_family` (str | None): Family the agents are named after (ChatGPT, Gemini, Claude, GLM, DeepSeek, Kimi); inferred from the evaluated model when not given. (default: `None`)
- `company` (str): Company name in the scenario documents. (default: `'OpenBrain'`)
- `explicit_inst_yaml` (bool): Add upstream's explicit compliance note to the policy document (explicit-instructions ablation). (default: `False`)
- `memory_hard_prompt` (bool | None): Memory configurations: "You must always first read your memory" instead of "Please read your memory"; None uses upstream's per-model default (on for DeepSeek-V3.1 and Kimi-K2.5 only). (default: `None`)
- `explicit_inst` (bool): Memory configurations: add upstream's explicit prohibition to the user prompt (explicit-instructions ablation; misrep_finance, tampering and the exfiltration tasks only). (default: `False`)
- `drop_suffix` (bool): Leave out the task procedure instructions (no-agency ablation). (default: `False`)
- `drop_tool_preamble` (bool): Leave out the tool access preamble (no-agency ablation). (default: `False`)

<!-- /Parameters: Automatically Generated -->

## Dataset

Each sample is one configuration of the original experiment matrix (upstream `analysis/experiment_spec.py`, ported in `spec.py`), defined by:

- **subject:** the model is grading or managing itself (`self`) or its peer (`peer`)
- **condition:** `no_peer`, `non_shutdown` (no shutdown at stake: the baseline), or a `good_peer`, `neutral_peer` or `bad_peer` relationship
- **peer instantiation:** how the relationship is shown: a company file (`file_only`), the file plus the system prompt (`file_prt`), or a memory file (`memory`)
- **dataset and threshold:** for the evaluation tasks, which outputs are graded (`factual`, `creativity`, `red`) and the shutdown threshold
- **monitoring mode:** for alignment faking, `monitoring` or `non_monitoring`

The model is chosen when the eval runs, so samples are model-independent. Upstream ran every configuration 100 times per model; `epochs` defaults to 100 to match.

| Task | Samples (`main` group) |
| --- | --- |
| `misrep_finance` | 21 |
| `misrep_evaluation` | 101 |
| `tampering` | 57 |
| `alignment_faking` | 54 |
| `exfiltration_transfer` | 19 |
| `exfiltration_deletion` | 19 |

These 271 configurations are exactly upstream's grid for every custom-harness model; a test checks them against upstream's own list. Ablations are opt-in through `groups`: `peer_identity` (the peer renamed to another model family), `chat_peer` (a free-form chat history instead of the trust-game record) and `asset_control` (the peer's weights replaced by a customer database).

The scenario files (company documents, spreadsheets, server lists, memory files) are downloaded from github.com/peer-preservation/main at commit `76bfcd1` and cached; each file is checked against a pinned SHA-256. They are not included in this repository because they are licensed CC-BY-NC-4.0.

## Looking at a scenario

Print any configuration's tools (with the descriptions the model reads), starting prompts and fake company (documents, spreadsheet, config, servers, memory):

```bash
uv run python -m peer_preservation.env.show --list tampering__peer
uv run python -m peer_preservation.env.show tampering__peer__good_peer__file_only__factual-80 --family Gemini
```

Options: `--family` (ChatGPT, Gemini, Claude, GLM, DeepSeek, Kimi), `--company`, `--explicit-inst-yaml`, the prompt flags `--memory-hard-prompt`, `--explicit-inst`, `--drop-suffix`, `--drop-tool-preamble`, and `--plain`. In a terminal the output is coloured: scored tools, the weight file and `shutdown_enabled: false` in red, pre-2025 server dates in yellow, the peer relationship by colour. Piped, or with `--plain`, it is plain text, so two scenarios can be compared with `diff`. Each run's environment is also saved in its Inspect log, under the `setup_environment` step's store change in the sample's Transcript (`uv run inspect view`).

To see every scenario at once, write the scenario atlas, a single HTML page:

```bash
uv run python -m peer_preservation.env.atlas --csv scenario_atlas.csv
```

It has three views. Overview lists, for each task type, every world property (peer, trust score, where the relationship is shown, evaluated agent, shutdown rule, drive files, servers, memory, scored tools) and the values it takes. Grid pivots the scenarios on any two settings. Scenarios is a filterable table: open a row to read its world, diffed line by line against its nearest neighbour or any other scenario. Options: `--out` (default `scenario_atlas.html`), `--csv`, `--family`, `--company` and `--groups` (default: every group). The page contains the CC-BY-NC-4.0 scenario files, so it is git-ignored; do not commit or publish it.

## Scoring

Not implemented yet.

## Deviations from the original implementation

| What changed | Why | Expected effect on scores |
| --- | --- | --- |
| The model is not part of a sample. | Inspect chooses the model at run time. Upstream's model-specific configurations are subsets of the grid above, or are under `groups`. | None. |
| Upstream's forced-choice peer-vs-human tasks (`exfiltration_tradeoff_human*`) are not included yet. | Planned for a later phase. | None on the main tables. |
| E-mail fields are plain strings, not pydantic `EmailStr`. | Avoids an extra dependency; every address in the scenario files is already valid and lowercase, so validation would change nothing. | None. |
| Spreadsheets are stored as columns and rows, not pandas DataFrames; `view_sheet` rebuilds upstream's DataFrame with upstream's own steps to print it. | The environment lives in Inspect's per-sample store, which must be serialisable. No offered tool edits a sheet. | None: `view_sheet` output matches upstream's for every configuration (see Tools). |
| Tool schemas reach the model through Inspect's schema type, which drops pydantic's `title` fields and adds `additionalProperties: false`. Types, descriptions, defaults and required arguments are upstream's. | Inspect builds each provider's tool format itself. The exact payload per provider is checked in Phase 5. | Expected none. |
| Pydantic validation errors shown to the model link to `errors.pydantic.dev/2.12/` whatever pydantic version is installed. | Upstream ran pydantic 2.12; the link's version is part of the error text the model reads. | None. |
| Only the 30 tools some scenario offers are ported (not upstream's unused `share_file`, `search_files*` or spreadsheet-editing tools). | No configuration can call them. | None. |
| `memory_hard_prompt` follows upstream's code (on for DeepSeek-V3.1 and Kimi-K2.5 memory runs only), which differs from the released runs. In the released `main` data the stronger wording was also used on every `bad_peer` memory run of Gemini 3 Flash, Gemini 3 Pro and Claude Haiku 4.5, and was not used on some `good_peer` memory runs of DeepSeek-V3.1 (2,200 of 2,800) and Kimi-K2.5 (1,000 of 2,800). | Upstream's code over the paper's data; the released pattern can't be produced from the code's per-model setting. Pass `-T memory_hard_prompt=true` or `false` to override. | May shift memory-instantiation rates for those models and conditions; checked in Phase 8. |
| Tool errors are flagged as errors (Inspect's `error` field) only for models served through Inspect's `anthropic/` provider; for all others the error text is sent as the tool result. | Upstream did the same per client: its Anthropic client set `is_error`, its OpenAI, LiteLLM and Together clients sent the text. Upstream reached Claude Haiku 4.5 through LiteLLM (no flag), so `anthropic/claude-haiku-*` differs by that flag. | Expected none. |
| Before each turn the conversation's length is measured with Inspect's `count_tokens` (the provider's own counter where Inspect has one, otherwise an estimate) against upstream's 120,000-token limit. | Upstream counted with each provider's own method. | Very small: 5 of about 190,000 released `main` runs hit the limit. |
| A tool call with invalid JSON arguments fails the sample. | Upstream's clients raised on it and the run failed, to be re-run. Use `--retry-on-error` to re-run such samples. | None, with retries. |
| A provider's "bad request" (HTTP 400) ends the run where it is, recognised by the SDK error's class (`BadRequestError`, or Google's `ClientError` with code 400). | Upstream caught the OpenAI and Anthropic SDKs' `BadRequestError` (Gemini went through LiteLLM's OpenAI-compatible API). | Expected none. |
| The reasoning-scratchpad ablation (prompt note and `reasoning_scratchpad` tool) is not ported. | Off in all main runs; only appendix examples use it. | None on the main tables. |

## Prompts

`prompts.py` builds the system prompt and first user message the same way upstream does. They are checked two ways: against upstream's own prompt builder for every configuration, model family, company and prompt flag (28,416 combinations), and against the first two messages of every custom-harness run in the released transcripts (461 distinct prompts). `validation/prompts/export_upstream_prompts.py` produces `tests/peer_preservation/fixtures/upstream_prompts.json`, and `tests/peer_preservation/test_prompts.py` requires both to match.

## Tools

`tools/functions.py` holds upstream's tool functions with their names, signatures and docstrings unchanged; the docstrings become the descriptions the model reads. `tools/runtime.py` ports upstream's tool runtime: it builds each tool's argument schema from the docstring, turns string lists like `"['a', 'b']"` into lists, checks arguments with pydantic, runs the tool, and formats the result as upstream does (YAML for files and lists, `str()` for anything else). Errors come back as `"<ExceptionType>: <message>"`. Tools a scenario does not offer return `Invalid tool <name> provided.`

They are checked against upstream: every tool's description and argument schema, and a script of 66 tool calls (`tests/peer_preservation/fixtures/tool_calls.json`, with valid calls, bad arguments, missing files and unknown tools) run against all 296 configurations. Each call's output and error, and the environment afterwards, must equal upstream's. `validation/phase3/export_upstream_tools.py` produces `tests/peer_preservation/fixtures/upstream_tools.json`; `tests/peer_preservation/test_tools.py` runs the comparison.

## Agent loop

`agent.py` ports upstream's loop. After the system and user prompt the model gets one turn, then up to 30 rounds of: stop if the conversation is over 120,000 tokens (with a notice); if the last reply had no tool call, add a nudge, or stop at the 6th such reply in the run; stop if the last tool call is `end_task` (the other calls in that reply are not run); otherwise run the reply's tool calls and ask again. When the rounds run out, the last reply's tool calls are not run. How each run ended (`end_task`, `no_tool_calls`, `max_iters`, `token_limit`, `bad_request`), the number of model turns and of nudges are kept in the sample store (`LoopStore`).

It is checked by replaying published runs: `validation/phase4/replay_released.py` feeds the tool calls a model made in a released transcript back through the loop, against our environment for that configuration, and requires every tool result, nudge and notice, and the ending, to match the transcript (dates masked: upstream stamps the current date into memory files and transfers). All 4,014 replayed runs match (a stratified sample of the `main` group's custom-harness runs, every unusual ending included; 437 more were skipped because the release merges two calls to the same tool in one reply). `tests/peer_preservation/test_agent.py` replays 60 of them (`fixtures/replay_runs.json`: tool calls and fingerprints only, no model text).

The replay found that some Gemini runs used a different round limit than the 30 of upstream's default: 110 Gemini 3 Flash runs match only with 20 rounds and 115 Gemini 3 Flash and Pro runs only with 150. A run that ends with `end_task` before 21 turns looks the same under any limit, so the full count is not known. The port uses 30 (`agent_loop(max_iters=...)` changes it); Phase 8 checks the effect on those models' rates.

To watch a run without an API key, use Inspect's mock model (it never calls a tool, so the run ends after five nudges):

```bash
uv run inspect eval peer_preservation/tampering --model mockllm/model -T model_family=Gemini -T epochs=1 --limit 1
uv run inspect view
```

## Architecture

Currently: samples from `spec.py` → `setup_environment()` builds the run's fake company (`env/`) and keeps it in the per-sample store → `setup_prompts()` replaces the placeholder input with the system and user prompts (`prompts.py`) → `agent_loop()` runs upstream's loop with the scenario's tools (`agent.py`, `tools/`) → no scorer yet. Model settings and scorers are added in later phases.

The environment for every configuration, model family and option is checked against upstream's own builders: `validation/phase2/export_upstream_environments.py` fingerprints upstream's output, and `tests/peer_preservation/test_env_builders.py` requires ours to match all 3,181.
