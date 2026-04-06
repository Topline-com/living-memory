# Dreamcatcher Memory Plugin for OpenClaw

Parametric memory for OpenClaw — knowledge lives in the model's weights, not in a database.

## What this does

Dreamcatcher replaces OpenClaw's default memory with a compact language model that is fully re-fine-tuned from scratch every night on your complete personal data. Instead of looking you up via retrieval, the model genuinely *knows* you — your projects, preferences, patterns, and context are encoded in the weight matrices.

This plugin is a thin HTTP client that bridges OpenClaw's memory slot to a running Living Memory server. All memory extraction, training, model management, and inference happen in the Living Memory process.

## Install

```bash
# Install the Living Memory server (Python)
pip install dreamcatcher-memory

# Install this plugin into OpenClaw
openclaw plugins install @dreamcatcher/openclaw-plugin
```

## Configure

In your `openclaw.json`:

```json
{
  "plugins": {
    "slots": {
      "memory": "dreamcatcher"
    }
  }
}
```

Or via environment variables:

```bash
export DREAMCATCHER_SERVER_URL="http://localhost:8420"
export DREAMCATCHER_AGENT_NAME="openclaw"
```

## Usage

1. Start the Living Memory server: `dreamcatcher serve`
2. Start OpenClaw normally — the plugin auto-connects
3. Use OpenClaw as usual. Transcripts are captured automatically.
4. At 3 AM, the nightly pipeline extracts memories and re-fine-tunes your model.
5. Tomorrow morning, your model wakes up smarter.

## How it integrates

| OpenClaw event | Dreamcatcher action |
|---|---|
| Memory search (every turn) | `POST /context` — parametric model returns personal context |
| Message sent | Transcript line accumulated in memory |
| `/new` command (session end) | `POST /ingest` — full transcript saved for tonight's training |
| `dreamcatcher_recall` tool | `POST /recall` — explicit targeted memory query |
| `dreamcatcher_status` tool | `GET /health` — model health and memory stats |

## Comparison with other OpenClaw memory providers

| | Dreamcatcher | Builtin | Mem0 | Supermemory | Honcho |
|---|---|---|---|---|---|
| Storage | Model weights | Markdown + SQLite | Cloud API | Cloud API | Cloud API |
| Runs locally | Yes | Yes | No | No | No |
| Gets better over time | Yes (nightly retrain) | No | Limited | Limited | Limited |
| Knows you vs. looks you up | Knows you | Looks up | Looks up | Looks up | Looks up |
| New info latency | 12-24h (next train) | Instant | Instant | Instant | Instant |
| Exact recall | Good | Best | Good | Good | Good |

## Requirements

- Node >= 22 (OpenClaw requirement)
- Running Living Memory server (`dreamcatcher serve`)
- Python 3.10+ (for Living Memory server)
