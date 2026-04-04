/**
 * Dreamcatcher Memory Plugin for OpenClaw
 * ========================================
 * Parametric memory layer that internalizes personal knowledge into a
 * compact model's weights through nightly re-fine-tuning, then serves
 * structured context via a local HTTP API.
 *
 * Unlike retrieval-based providers (Mem0, Supermemory, Honcho), Dreamcatcher
 * encodes knowledge parametrically — the memory model genuinely "knows" the
 * user rather than looking them up. The plugin itself is a thin HTTP client;
 * all memory logic runs in the separate Dreamcatcher server process.
 *
 * Integration points:
 *   memory slot  → owns the exclusive memory provider slot
 *   prefetch     → GET /context  (personal memory injected every turn)
 *   message hook → accumulates transcript in memory
 *   command:new  → POST /ingest  (transcript saved for tonight's training)
 *   tools        → dreamcatcher_recall, dreamcatcher_status
 *
 * Requires: a running Dreamcatcher server (default http://localhost:8420).
 * Install Dreamcatcher separately: pip install dreamcatcher-memory
 * Start the server: dreamcatcher serve
 */

import type { PluginAPI, ToolResult } from "openclaw/plugin-sdk/types";

// ── Configuration ───────────────────────────────────────────────────

interface DreamcatcherConfig {
  serverUrl: string;
  agentName: string;
}

const DEFAULT_CONFIG: DreamcatcherConfig = {
  serverUrl: "http://localhost:8420",
  agentName: "openclaw",
};

const PREFETCH_TIMEOUT_MS = 5000;
const INGEST_TIMEOUT_MS = 10000;
const MAX_CONSECUTIVE_FAILURES = 5;

// ── State ───────────────────────────────────────────────────────────

let config: DreamcatcherConfig = { ...DEFAULT_CONFIG };
let available = false;
let consecutiveFailures = 0;
let transcriptLines: string[] = [];
let lastContext = "";

// ── HTTP helpers ────────────────────────────────────────────────────

async function dcFetch(
  path: string,
  options: RequestInit & { timeoutMs?: number } = {}
): Promise<Response> {
  const { timeoutMs = PREFETCH_TIMEOUT_MS, ...fetchOpts } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const resp = await fetch(`${config.serverUrl}${path}`, {
      ...fetchOpts,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...fetchOpts.headers,
      },
    });
    return resp;
  } finally {
    clearTimeout(timer);
  }
}

function recordSuccess() {
  consecutiveFailures = 0;
}

function recordFailure(context: string) {
  consecutiveFailures++;
  if (consecutiveFailures === MAX_CONSECUTIVE_FAILURES) {
    console.warn(
      `[dreamcatcher] Circuit breaker tripped after ${MAX_CONSECUTIVE_FAILURES} ` +
        `consecutive failures (last: ${context}). Pausing until next session.`
    );
  }
}

function circuitOpen(): boolean {
  return consecutiveFailures >= MAX_CONSECUTIVE_FAILURES;
}

// ── Core memory operations ──────────────────────────────────────────

/**
 * Fetch personal context from the parametric memory model.
 * Called before each turn to inject relevant knowledge into the prompt.
 */
async function fetchContext(query: string): Promise<string> {
  if (!available || circuitOpen()) return lastContext;

  try {
    const resp = await dcFetch("/context", {
      method: "POST",
      body: JSON.stringify({
        query: query || "general context",
        agent_name: config.agentName,
      }),
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const data = await resp.json();
    const context = data.response ?? "";
    lastContext = context;
    recordSuccess();
    return context;
  } catch (e) {
    recordFailure(`prefetch: ${e}`);
    return lastContext; // Return stale context rather than nothing
  }
}

/**
 * Flush the accumulated transcript to Dreamcatcher for tonight's training.
 * Called at session boundaries (command:new or explicit session end).
 */
async function ingestTranscript(): Promise<void> {
  if (!available || transcriptLines.length === 0) return;

  const transcript = transcriptLines.join("\n\n");
  transcriptLines = [];

  try {
    const resp = await dcFetch("/ingest", {
      method: "POST",
      timeoutMs: INGEST_TIMEOUT_MS,
      body: JSON.stringify({
        transcript,
        agent_name: config.agentName,
      }),
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const data = await resp.json();
    recordSuccess();
    console.log(
      `[dreamcatcher] Ingested session ${data.session_id ?? "?"} (${transcript.length} chars)`
    );
  } catch (e) {
    recordFailure(`ingest: ${e}`);
    // Re-add lines so they aren't lost on transient failure
    transcriptLines = transcript.split("\n\n").concat(transcriptLines);
  }
}

// ── Tool handlers ───────────────────────────────────────────────────

async function toolRecall(query: string): Promise<ToolResult> {
  if (!available) {
    return { content: "Dreamcatcher server is not available." };
  }

  try {
    const resp = await dcFetch("/recall", {
      method: "POST",
      body: JSON.stringify({ query }),
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const data = await resp.json();
    const memories: Array<{ category?: string; content?: string }> =
      data.memories ?? [];

    if (memories.length === 0) {
      return { content: "No memories found for this query." };
    }

    const lines = memories.map(
      (m) => `[${m.category ?? "?"}] ${m.content ?? ""}`
    );
    return { content: lines.join("\n") };
  } catch (e) {
    return { content: `Memory recall failed: ${e}` };
  }
}

async function toolStatus(): Promise<ToolResult> {
  if (!available) {
    return { content: "Dreamcatcher server is not available." };
  }

  try {
    const resp = await dcFetch("/health");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const health = await resp.json();
    const stats = health.stats ?? {};

    return {
      content: JSON.stringify(
        {
          status: "ok",
          model_loaded: health.model_loaded ?? false,
          active_memories: stats.active_memories ?? 0,
          total_training_examples: stats.total_training_examples ?? 0,
          training_runs: stats.training_runs ?? 0,
          memories_by_category: stats.memories_by_category ?? {},
        },
        null,
        2
      ),
    };
  } catch (e) {
    return { content: `Status check failed: ${e}` };
  }
}

// ── Plugin entry point ──────────────────────────────────────────────

export default function register(api: PluginAPI) {
  // ── Load config ─────────────────────────────────────────────────

  config = {
    serverUrl:
      process.env.DREAMCATCHER_SERVER_URL ??
      api.config?.serverUrl ??
      DEFAULT_CONFIG.serverUrl,
    agentName:
      process.env.DREAMCATCHER_AGENT_NAME ??
      api.config?.agentName ??
      DEFAULT_CONFIG.agentName,
  };

  // ── Health check on startup ─────────────────────────────────────

  dcFetch("/health", { timeoutMs: 3000 })
    .then(async (resp) => {
      if (resp.ok) {
        const health = await resp.json();
        available = true;
        console.log(
          `[dreamcatcher] Connected at ${config.serverUrl} ` +
            `(model_loaded=${health.model_loaded ?? false})`
        );
      } else {
        console.warn(
          `[dreamcatcher] Server returned ${resp.status} — ` +
            `running without parametric memory`
        );
      }
    })
    .catch((e) => {
      console.warn(
        `[dreamcatcher] Server not reachable at ${config.serverUrl}: ${e}. ` +
          `Start it with: dreamcatcher serve`
      );
    });

  // ── Register memory search (owns the memory slot) ───────────────

  api.registerMemorySearch(async (query: string) => {
    const context = await fetchContext(query);
    if (!context) return [];

    return [
      {
        source: "dreamcatcher",
        content: context,
        score: 1.0,
      },
    ];
  });

  // ── Register tools ──────────────────────────────────────────────

  api.registerTool({
    name: "dreamcatcher_recall",
    description:
      "Query the user's parametric memory model for specific information. " +
      "Use this when you need to recall specific facts about the user that " +
      "weren't in the automatic context injection — for example, details " +
      "about a project they mentioned weeks ago, or a preference they " +
      "expressed in a different context.",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "What to recall about the user",
        },
      },
      required: ["query"],
    },
    handler: async (params: { query: string }) => toolRecall(params.query),
  });

  api.registerTool({
    name: "dreamcatcher_status",
    description:
      "Check the health and status of the user's memory model. " +
      "Shows when the model was last trained, how many memories " +
      "it contains, and whether the nightly pipeline is healthy.",
    parameters: {
      type: "object",
      properties: {},
    },
    handler: async () => toolStatus(),
  });

  // ── Hook: message_sending → accumulate transcript ───────────────

  api.on("message_sending", (event) => {
    const { role, content } = event;
    if (typeof content !== "string") return;

    if (role === "user") {
      transcriptLines.push(`User: ${content}`);
    } else if (role === "assistant") {
      const truncated =
        content.length > 4000 ? content.slice(0, 4000) + "..." : content;
      transcriptLines.push(`Assistant: ${truncated}`);
    }
  });

  // ── Hook: command:new → flush transcript (session boundary) ─────

  api.on("command:new", () => {
    ingestTranscript().catch((e) =>
      console.warn(`[dreamcatcher] Failed to flush transcript on /new: ${e}`)
    );
  });

  // ── System prompt injection ─────────────────────────────────────

  api.registerSystemPrompt(() => {
    if (!available) return "";

    return (
      "## Parametric Memory (Dreamcatcher)\n" +
      "You have access to a personal memory model that has been trained on " +
      "the user's complete interaction history. Personal context about the " +
      "user's projects, preferences, relationships, and patterns is " +
      "automatically injected at the start of each turn — you don't need " +
      "to request it. For specific queries about things not in the automatic " +
      "context, use the dreamcatcher_recall tool.\n"
    );
  });
}
