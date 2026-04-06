# Dreamcatcher

**Living memory for AI agents.** Giving your agents a life beyond sessions.

Your AI agents don't remember you. Every session starts cold. Dreamcatcher changes that. It's a compact language model re-fine-tuned from fixed pretrained weights every night on your complete personal memory dataset — so every morning, your agents wake up knowing everything about you. Not looking you up. Actually knowing you.

**Living memory** is memory that lives in the model's weights, not in a database. Memory that grows and reconsolidates every night while you sleep. Memory that makes your agents wake up smarter every morning. The **Dreamcatcher architecture** is how you build it: capture experiences during the day, replay everything at night, rebuild integrated knowledge from scratch — modeled on how the brain consolidates memory during sleep.

No LoRA drift. No retrieval latency. No cloud memory storage. Just a model that knows you better every day.

**[White Paper](docs/whitepaper.md)** · **[X Post](docs/x-post.md)**

---

## Quickstart (5 minutes)

### 1. Install

```bash
# Clone the repo
git clone https://github.com/alexskatell/living-memory.git
cd living-memory

# Install (server + client only — no GPU required for inference)
pip install -e .

# For training on Apple Silicon (Mac M1/M2/M3/M4 — recommended):
pip install mlx anthropic
# mlx-lm >=0.31.2 required for Gemma 4 support. Install from main until PyPI catches up:
pip install git+https://github.com/ml-explore/mlx-lm.git

# For training on NVIDIA GPU:
pip install -e ".[train]"
```

### 2. Configure

```bash
# Copy the example env file and add your Anthropic API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
# This key is ONLY used during nightly extraction (~$0.01-0.05/night).
```

### 3. Initialize

```bash
dreamcatcher init
# Creates data directories and downloads the base model (Gemma 4 E2B by default, ~1.5GB quantized)
```

### 4. Feed it a transcript

```bash
# Drop any conversation transcript into the sessions folder
cp my_session.txt data/sessions/

# Or ingest directly
dreamcatcher ingest my_session.txt --agent hermes
```

### 5. Run the nightly pipeline manually (first time)

```bash
# This extracts memories, builds training pairs, and trains the model
dreamcatcher nightly
# Takes 2-5 minutes on first run with a few sessions
```

### 6. Start the server

```bash
dreamcatcher serve
# Memory model now serving on http://localhost:8420
```

### 7. Test it

```bash
curl -X POST http://localhost:8420/context \
  -H "Content-Type: application/json" \
  -d '{"query": "What projects is the user working on?"}'
```

That's it. Your memory model is live. Now integrate it with your agents.

---

## Agent Integration

### The Two-Line Version

Every agent integration has exactly two touchpoints. At session start, get context. At session end, save the transcript.

```python
from dreamcatcher_client import LivingMemory

memory = LivingMemory()  # Connects to localhost:8420

# SESSION START: inject personal context into your agent's system prompt
context = memory.get_context("What should I focus on today?")
system_prompt = f"You are a helpful assistant.\n\n{context}"

# SESSION END: save transcript for tonight's training
memory.save_session(transcript, agent_name="my-agent")
```

### Claude Code (MCP Integration)

Native integration via the Model Context Protocol. One-command setup:

```bash
pip install dreamcatcher-memory[claude-code]
dreamcatcher setup claude-code --global
dreamcatcher serve
```

Restart Claude Code. Your personal memory is now active in every session. The MCP server provides three tools:
- **`living_memory_recall`** — Query specific memories on demand
- **`living_memory_status`** — Check model health and memory stats
- **`living_memory_save_session`** — Auto-saves conversations for nightly training

Personal context is injected into Claude's system prompt at session start. Conversations are automatically saved for the nightly training pipeline. See [`integrations/claude-code/README.md`](integrations/claude-code/README.md) for manual setup and configuration options.

> **Note:** Requires Python 3.10+. If your system Python is older, use [uv](https://github.com/astral-sh/uv) to create a venv: `uv venv --python 3.12 .venv && source .venv/bin/activate`

### OpenClaw Integration

Add this to your agent's initialization:

```python
from dreamcatcher_client import LivingMemory

memory = LivingMemory()

# Before sending the first message to Claude:
personal_context = memory.get_context(user_message)
messages = [
    {"role": "system", "content": f"{base_system_prompt}\n\n{personal_context}"},
    {"role": "user", "content": user_message},
]

# When the session ends:
full_transcript = "\n".join(format_messages(messages))
memory.save_session(full_transcript, agent_name="openclaw")
```

### Hermes Agent Integration

```python
from dreamcatcher_client import LivingMemory

memory = LivingMemory()

class HermesAgent:
    def on_session_start(self, user_query):
        # Get personal context for this session
        self.personal_context = memory.get_context(user_query)
        self.transcript = []

    def build_system_prompt(self):
        base = "You are Hermes, a research and coordination agent."
        if self.personal_context:
            return f"{base}\n\n{self.personal_context}"
        return base

    def on_session_end(self):
        # Save everything for tonight's training
        memory.save_session(
            "\n".join(self.transcript),
            agent_name="hermes"
        )
```

### Generic (Any Language, Any Framework)

Dreamcatcher is just an HTTP API. No SDK required.

```bash
# Get context (session start)
curl -X POST http://localhost:8420/context \
  -H "Content-Type: application/json" \
  -d '{"query": "user is asking about project timeline"}'

# Save transcript (session end)
curl -X POST http://localhost:8420/ingest \
  -H "Content-Type: application/json" \
  -d '{"transcript": "...", "agent_name": "my-agent"}'
```

The `/context` endpoint returns a `<personal_memory>` block ready for direct injection into any system prompt. The `/ingest` endpoint stores the transcript for tonight's extraction. That's the entire API surface.

---

## How It Works

**Daytime:** Agents run normally. Session transcripts are saved via HTTP POST to the Dreamcatcher server and stored in SQLite. No learning happens. No API calls. Just fast local storage.

**3 AM (Nightly Pipeline):**
1. **Extract:** A frontier LLM (Claude Sonnet) reads only today's new transcripts and generates structured memories with `core_fact` + multi-angle training pairs. Each session is extracted exactly once. Historical sessions are never re-sent to the API.
2. **Train:** The full canonical training set (every memory ever extracted) is used to re-fine-tune the compact base model (Gemma 4 E2B, ~2.3B effective parameters) from its original pretrained weights. No LoRA. No incremental updates. Clean rebuild from scratch every night. The trainer auto-detects your platform: MLX on Apple Silicon, PyTorch on NVIDIA.
3. **Benchmark:** An automated recall check verifies the model internalized the data. If recall drops below threshold, the new model isn't deployed.
4. **Deploy:** Atomic symlink swap to the new model. Previous model archived.

**Morning:** Agents query the local Dreamcatcher model. The model returns structured JSON containing relevant personal context. This gets injected into the frontier model's system prompt. The frontier model reasons with personal context it didn't have to retrieve.

---

## Configuration

Edit `config.yaml` to customize. Key settings:

```yaml
model:
  name: "google/gemma-4-E2B-it"  # Default. Qwen/Qwen3.5-0.8B for lighter hardware.

training:
  epochs: 3
  compression_age_days: 180    # Reduce pair density for memories older than 6 months
  max_pair_index_old: 1        # Keep semantic + contextual pairs for old memories

extraction:
  provider: "anthropic"        # or "openai"
  model: "claude-sonnet-4-20250514"

server:
  port: 8420
  stale_model_threshold_hours: 36  # Disaster recovery activates after this
```

---

## Training on Apple Silicon

Dreamcatcher auto-detects Apple Silicon and uses MLX for training. **Requires `mlx-lm` >=0.31.2** for Gemma 4 support (install from GitHub main until the PyPI release catches up). The default model (Gemma 4 E2B, ~2.3B effective parameters) produces the highest-quality structured JSON output but requires more memory and training time than smaller alternatives.

**Gemma 4 E2B on Mac M4 24GB (default):**
Gemma 4 E2B's ~2.3B trainable core (PLE embedding tables frozen) fits in 24GB unified memory with gradient checkpointing enabled. Training times are longer than smaller models but well within the overnight window: approximately 25-45 minutes for 6-12 months of accumulated data (with semantic compression), scaling to 50-90 minutes at the 2-3 year mark. The quality gain in structured output is meaningful — sharper recall, more consistent JSON formatting, fewer edge-case hallucinations in the memory layer. Use a learning rate of 5e-6 (set in config.yaml by default) rather than the 2e-5 used for smaller models.

**If training feels too heavy:** swap to Qwen3.5-0.8B by changing one line in `config.yaml`:
```yaml
model:
  name: "Qwen/Qwen3.5-0.8B"  # 800M dense, ~2 min training, guaranteed fit
```

Training time drops to roughly one-third and memory usage drops substantially. The architecture is model-agnostic by design — this swap requires zero code changes.

**Recommendation:** Run `dreamcatcher nightly` manually on your current dataset size and time it. If the full pipeline (extract + train + benchmark) completes comfortably under 60-90 minutes on your hardware, Gemma 4 E2B is worth keeping. If it starts feeling heavy, swap to Qwen and keep Gemma as the premium option for beefier machines.

**Direct MLX training command** (if you want to run fine-tuning manually outside the pipeline):
```bash
# Full fine-tuning with MLX (not LoRA)
python -m mlx_lm.lora \
  --model google/gemma-4-E2B-it \
  --train \
  --data ./data/training \
  --batch-size 4 \
  --iters 500 \
  --learning-rate 2e-5 \
  --fine-tune-type full \
  --grad-checkpoint \
  --adapter-path ./data/models/memory_$(date +%Y%m%d)
```

---

## Nightly Automation

Set up the cron job to run the full pipeline automatically:

```bash
# Add to crontab (runs at 3 AM daily)
# Note: cron doesn't load your shell profile, so source .env explicitly
(crontab -l 2>/dev/null; echo "0 3 * * * cd $(pwd) && set -a && source .env && set +a && dreamcatcher nightly >> data/nightly.log 2>&1") | crontab -

# Or use the provided script (recommended — handles .env loading for you)
chmod +x scripts/train_nightly.sh
# Add to crontab: 0 3 * * * /path/to/dreamcatcher/scripts/train_nightly.sh
```

### Docker (Alternative)

```bash
cp .env.example .env
# Edit .env with your API key
docker-compose up -d
# Server runs on :8420, nightly training runs via cron at 3 AM
```

---

## CLI Reference

```bash
dreamcatcher init              # Initialize directories + download base model
dreamcatcher ingest FILE       # Ingest a transcript file
dreamcatcher extract           # Run frontier LLM extraction on new sessions
dreamcatcher train             # Re-fine-tune model from base weights
dreamcatcher nightly           # Full pipeline: extract → train → benchmark → deploy
dreamcatcher serve             # Start inference server on :8420
dreamcatcher mcp               # Start MCP server (for Claude Code)
dreamcatcher setup claude-code # Configure Claude Code MCP integration
dreamcatcher stats             # Show memory database statistics
dreamcatcher export            # Export memories as JSON
dreamcatcher cleanup           # Remove old model checkpoints
```

---

## Architecture

Dreamcatcher is the architecture that produces **living memory** — a compact model that internalizes personal knowledge into its weights and produces structured JSON context for injection into a frontier reasoning model's prompt.

The formal guarantee: let C_t be the append-only canonical memory ledger and T_t = R(C_t, π_t) be the rendered training set. The model M_t = F(M₀, T_t) is an independent function of the fixed base weights and tonight's training set. No dependence on previous models. No sequential-update drift. The canonical ledger grows monotonically (C_t ⊆ C_{t+1}) — no extracted fact is ever removed.

**Key properties:**
- **No replay forgetting:** Every memory is replayed in every training cycle from fixed pretrained weights.
- **Organic reinforcement:** Important facts naturally accumulate more training pairs through real-world reuse. No scoring system needed.
- **On-device inference:** The memory model runs locally. Only curated structured context leaves your device.
- **Replayable rebuild:** Delete a memory, retrain, and the model has zero trace of it.
- **Model-agnostic:** Swap base models with a one-line config change.

---

## Cost

The nightly pipeline costs approximately **$0.05–$0.15** in frontier LLM API fees for extraction (the only cloud touchpoint), plus GPU electricity for local training. Monthly cost: **$2–5**.

If you have a local GPU (RTX 3090 or better), training is free. Cloud GPU rental for the 2–5 minute nightly job costs approximately $0.01–0.03 on Vast.ai or RunPod.

---

## FAQ

**How long before it starts working?**
Feed it 3–5 session transcripts and run `dreamcatcher nightly`. The model will have basic personalization immediately. Meaningful behavioral internalization takes 1–2 weeks of daily use.

**Does it work without a GPU?**
Inference (serving the memory model) works on CPU — it's a compact model, ~50-200ms per query. Training requires a GPU or Apple Silicon. On Mac (M1/M2/M3/M4), training uses MLX natively. On NVIDIA, it uses PyTorch. You can also train on a cloud GPU ($0.01-0.03/run) and sync the model file.

**Can I use it with Claude Code / Cursor / Aider / any agent?**
Yes. Any agent that lets you customize the system prompt can use Dreamcatcher. The `/context` endpoint returns a text block ready for system prompt injection. The `/ingest` endpoint accepts any text transcript.

**What about privacy?**
The trained memory model and all personal data stay on your device. During inference, the model runs locally — no network access. The one cloud touchpoint is the nightly extraction step, which sends raw session transcripts to a frontier LLM API. For full data sovereignty, you can substitute a local extraction model.

**How does it handle outdated information?**
Session dates are injected into the extraction prompt, so facts carry temporal markers ("As of April 2026, the user routes through Qwen"). The model learns recency through natural language. Important changes get reinforced through organic reuse across future sessions.

---

## Troubleshooting

**"Could not resolve authentication method" during `dreamcatcher nightly`**
Your `ANTHROPIC_API_KEY` isn't reaching the process. The CLI auto-loads `.env` via `python-dotenv`, so make sure your key is in `.env` in the project root. If running via cron, source `.env` explicitly (see Nightly Automation above). You can verify with: `ANTHROPIC_API_KEY=sk-ant-... dreamcatcher extract`

**"Model type gemma4 not supported" during training**
Your `mlx-lm` version is too old. Gemma 4 support was added after v0.31.1. Install from main: `pip install git+https://github.com/ml-explore/mlx-lm.git`. Once mlx-lm >=0.31.2 ships on PyPI, `pip install --upgrade mlx-lm` will work.

**Training runs but loss shows 0.0000**
The MLX Python training API doesn't return loss directly. The actual training loss is printed to the console during training (look for the per-step loss reports). The 0.0 in the summary is a logging limitation, not an actual zero loss.

**`dreamcatcher nightly` only works from the project directory**
The `config.yaml` uses relative paths (`./data/...`). Either `cd` to the project root before running, or set absolute paths in `config.yaml`.

---

## White Paper

The full technical paper — *Living Memory for AI Agents: The Dreamcatcher Architecture for Nightly Parametric Consolidation* — covering formal properties, organic reinforcement, semantic compression, production reliability mechanisms, and comparative analysis against RAG, Letta, LoRA continual learning, MemoRAG, and Lamini is available in [`docs/whitepaper.md`](docs/whitepaper.md).

---

## License

MIT
