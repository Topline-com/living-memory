# Dreamcatcher Memory Provider for Hermes Agent

**Parametric memory that makes your agent genuinely know you — not look you up.**

Dreamcatcher is a memory architecture where a compact language model is re-fine-tuned from fixed pretrained weights every night on your complete personal data. Unlike retrieval-based memory providers (Mem0, Hindsight, RetainDB), Dreamcatcher encodes your knowledge into the model's weights. The result: your agent starts every morning with internalized understanding of your projects, preferences, relationships, and patterns — without reading a file or querying a database.

This plugin connects Hermes Agent to a running Dreamcatcher server. The plugin itself is a thin HTTP client (~300 lines). All memory logic — extraction, training, model management, compression, linting — runs in the separate Dreamcatcher process.

## Requirements

- **Hermes Agent v0.7.0+** (with pluggable memory provider support)
- **Dreamcatcher server running** on the same machine or network

## Setup

### 1. Install Dreamcatcher (one time)

```bash
git clone https://github.com/[username]/dreamcatcher.git
cd dreamcatcher
pip install -e .

# For training on Apple Silicon:
pip install mlx mlx-lm anthropic

# For training on NVIDIA:
pip install -e ".[train]"

# Initialize and start:
cp .env.example .env   # Add your ANTHROPIC_API_KEY
dreamcatcher init
dreamcatcher serve      # Starts on http://localhost:8420
```

### 2. Install the plugin

Copy the plugin directory into your Hermes plugins folder:

```bash
cp -r integrations/hermes/plugins/memory/dreamcatcher \
      /path/to/hermes-agent/plugins/memory/dreamcatcher
```

### 3. Activate via Hermes

```bash
# Interactive setup (recommended):
hermes memory setup
# Select "Dreamcatcher" from the picker

# Or manual config:
hermes config set memory.provider dreamcatcher
```

### 4. Verify

```bash
hermes memory status
# Should show: Dreamcatcher (connected, model_loaded=True)
```

## How It Works

**Every turn:** Hermes calls `prefetch()`, which hits Dreamcatcher's `/context` endpoint. The memory model runs inference locally (~50ms) and returns a structured `<personal_memory>` block that gets injected into the system prompt. The agent sees your personal context without making any retrieval calls.

**Every turn (end):** The user and assistant messages are accumulated into a transcript buffer in memory. No network calls happen during the conversation.

**Session end:** The full accumulated transcript is POSTed to Dreamcatcher's `/ingest` endpoint. The transcript is stored in SQLite for tonight's extraction pipeline.

**3 AM (nightly):** Dreamcatcher's pipeline extracts structured memories from all new transcripts, then re-fine-tunes the memory model from fixed pretrained weights on the complete canonical dataset. The model deployed the next morning has integrated yesterday's experiences into the same foundational structure that holds everything from the past year.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DREAMCATCHER_SERVER_URL` | `http://localhost:8420` | URL of your Dreamcatcher server |
| `DREAMCATCHER_AGENT_NAME` | `hermes` | Name tag for transcripts from this agent |

Set these in your `.env` file or via `hermes memory setup`.

## Tools

| Tool | Description |
|------|-------------|
| `dreamcatcher_recall` | Explicit memory query — ask about specific facts not in the automatic context |
| `dreamcatcher_status` | Check memory model health, last training date, memory count |

Automatic context injection happens via `prefetch()` every turn — these tools are for when the agent needs to make a targeted, explicit query.

## Architecture

```
┌─────────────────────────────────────┐
│         Hermes Agent v0.7.0         │
│                                     │
│  ┌─────────────────────────────┐    │
│  │  Dreamcatcher Plugin (HTTP) │    │
│  │  prefetch → /context        │    │
│  │  session_end → /ingest      │    │
│  │  tools → /recall, /health   │    │
│  └──────────┬──────────────────┘    │
│             │ localhost:8420         │
└─────────────┼───────────────────────┘
              │
┌─────────────┼───────────────────────┐
│  Dreamcatcher Server (on-device)    │
│             │                       │
│  ┌──────────▼──────────────────┐    │
│  │  Trained Memory Model       │    │
│  │  (Gemma 4 E2B / Qwen)      │    │
│  │  Re-fine-tuned nightly      │    │
│  └─────────────────────────────┘    │
│                                     │
│  ┌─────────────────────────────┐    │
│  │  SQLite Canonical Ledger    │    │
│  │  Sessions → Extraction →    │    │
│  │  Training Pairs → Model     │    │
│  └─────────────────────────────┘    │
│                                     │
│  ┌─────────────────────────────┐    │
│  │  Obsidian Vault (browse)    │    │
│  │  Lint Reports (weekly)      │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

## Comparison with Other Providers

| Feature | Dreamcatcher | Mem0 | Honcho | Holographic |
|---------|-------------|------|--------|-------------|
| **Storage** | Model weights (parametric) | Cloud vectors | Cloud API | Local SQLite |
| **Recall method** | Model inference (~50ms) | Embedding search | Dialectic query | FTS5 + HRR |
| **Learns behavior** | Yes (weight-level) | No | Yes (dialectic) | No |
| **Works offline** | Yes (after training) | No | No | Yes |
| **Privacy** | On-device inference | Cloud storage | Cloud storage | Local |
| **Update latency** | Overnight (nightly train) | Instant | Instant | Instant |
| **Cost** | $0.05-0.15/night API | Usage-based | Usage-based | Free |

Dreamcatcher's unique advantage is parametric internalization — knowledge encoded in weights rather than retrieved from external storage. The tradeoff is the overnight training cycle (new information takes 12-24 hours to internalize). For users who need both instant updates and deep internalization, Dreamcatcher can run alongside the built-in MEMORY.md/USER.md system, which provides immediate storage while Dreamcatcher handles long-term consolidation.

## Troubleshooting

**"Dreamcatcher server not reachable"** — Make sure `dreamcatcher serve` is running. Check the URL in your config matches the server's actual address and port.

**"model_loaded=False"** — The server is running but no trained model exists yet. Run `dreamcatcher nightly` to train your first model, then restart the server.

**Context seems stale** — The memory model is updated nightly. Information from today's sessions won't appear in the model until tomorrow morning. This is by design — the architecture consolidates during the overnight training cycle, like sleep consolidation in the brain.

## Links

- **[Dreamcatcher Repository](https://github.com/[username]/dreamcatcher)** — Core architecture, server, training pipeline
- **[White Paper](https://github.com/[username]/dreamcatcher/docs/whitepaper.pdf)** — Formal analysis and comparative evaluation
- **[Hermes Memory Providers Guide](https://docs.hermes.ai/user-guide/features/memory-providers)** — All available memory providers
