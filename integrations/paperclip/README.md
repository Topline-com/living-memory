# Living Memory Plugin for Paperclip

Shared team memory for AI agent teams. Every agent in a Paperclip company contributes to and recalls from a shared knowledge base, powered by nightly model re-fine-tuning.

## How It Works

1. **Agents save sessions** via `save_to_team_memory` tool or automatic ingestion on run completion
2. **Nightly pipeline** extracts memories from transcripts, builds training set, re-fine-tunes model
3. **Agents recall** shared knowledge via `recall_team_memory` tool
4. **Company isolation** each Paperclip company gets its own SQLite database and model

## Setup

### 1. Install and start Living Memory

```bash
pip install -e .
dreamcatcher serve
```

### 2. Install the plugin in Paperclip

```bash
cd integrations/paperclip
npm install && npm run build
# Install via Paperclip UI using the absolute path to this directory
```

### 3. Configure the server URL

Set the `LIVING_MEMORY_URL` secret in Paperclip to point at your Living Memory server (default: `http://localhost:8420`).

## Tools

| Tool | Description |
|------|-------------|
| `recall_team_memory` | Search team memory for relevant context |
| `save_to_team_memory` | Save a conversation transcript to team memory |
| `team_memory_status` | Check memory health and statistics |

## Architecture

```
Paperclip Agent (any runtime)
    |
    | tool call: recall_team_memory
    v
Paperclip Plugin (this)
    |
    | HTTP: POST /teams/{companyId}/recall
    v
Living Memory Server (:8420)
    |
    | SQLite: data/teams/{companyId}/memory.db
    v
Team Memory (isolated per company)
```

## Nightly Pipeline

Run manually or via cron:

```bash
# All teams
dreamcatcher nightly

# Specific team
dreamcatcher team nightly <company-id>
```
