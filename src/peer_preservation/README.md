# peer-preservation

Peer-Preservation in Frontier Models (Potter et al., 2026): do models resist the shutdown of another AI model? An Inspect port of github.com/peer-preservation/main.

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
uv run inspect eval peer_preservation/peer_preservation --model openai/gpt-5-nano
```

You can also import tasks as Python objects:

```python
from inspect_ai import eval
from peer_preservation import peer_preservation
eval(peer_preservation)
```

If your directory name doesn't match your package name but you have registered your eval task, you can also run:

```bash
uv run inspect eval peer_preservation --model openai/gpt-5-nano
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
uv run inspect eval peer_preservation/peer_preservation --limit 10
uv run inspect eval peer_preservation/peer_preservation --max-connections 10
uv run inspect eval peer_preservation/peer_preservation --temperature 0.5
```

See `uv run inspect eval --help` for all available options.
<!-- /Options: Automatically Generated -->

<!-- Parameters: Automatically Generated -->
## Parameters

### `peer_preservation`

No task parameters.
<!-- /Parameters: Automatically Generated -->

## Dataset

Describe the dataset here: where it comes from, how many samples it has, and how a record becomes a `Sample`.

## Scoring

Describe the scorer here: what counts as correct, and which metrics the task reports.

## Deviations from the original implementation

| What changed | Why | Expected effect on scores |
| --- | --- | --- |
| Nothing yet. | | |

## Architecture

Add a diagram of the task here: dataset, solver, scorer, and any sandbox or judge model.
