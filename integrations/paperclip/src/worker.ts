/**
 * Living Memory Worker for Paperclip
 * ====================================
 * Main plugin logic: registers tools, subscribes to events,
 * and schedules nightly training triggers.
 *
 * Architecture:
 *   Paperclip companyId → Living Memory team_id (1:1 mapping)
 *   Each company gets its own isolated SQLite DB, model, and training cycle.
 *   All agent sessions within a company are shared across the team.
 *
 * Integration points:
 *   tools  → recall_team_memory, save_to_team_memory, team_memory_status
 *   events → agent.run.completed (auto-ingest transcripts)
 *   jobs   → nightly-memory-training (trigger at 3 AM)
 */

import { LivingMemoryClient } from "./client.js";
import {
  TOOL_RECALL,
  TOOL_SAVE,
  TOOL_STATUS,
  JOB_NIGHTLY,
  SECRET_URL,
  DEFAULT_URL,
} from "./constants.js";

// ── Type stubs for Paperclip plugin SDK ───────────────────────
// These mirror the Paperclip SDK interfaces. When building against
// the actual SDK, replace with proper imports.

interface ToolContext {
  scope: [string, string]; // e.g. ['company', 'abc123']
}

interface PluginEvent {
  id: string;
  type: string;
  scope: [string, string];
  payload: Record<string, unknown>;
  timestamp: number;
}

interface PluginContext {
  tools: {
    register(def: {
      name: string;
      description: string;
      schema: Record<string, unknown>;
      handle: (input: Record<string, unknown>, ctx: ToolContext) => Promise<unknown>;
    }): void;
  };
  events: {
    subscribe(
      event: string,
      handler: (event: PluginEvent) => Promise<void>
    ): void;
  };
  jobs: {
    register(def: {
      key: string;
      schedule: string;
      handle: (ctx: unknown) => Promise<void>;
    }): void;
  };
  secrets: {
    resolve(key: string): Promise<string | undefined>;
  };
  activity: {
    log(type: string, data?: Record<string, unknown>): void;
  };
  metrics: {
    write(key: string, value: number): void;
  };
  http: unknown;
  state: unknown;
}

// ── Plugin setup ──────────────────────────────────────────────

export async function setup(ctx: PluginContext): Promise<void> {
  // Resolve the Living Memory server URL from secrets or use default
  const url = (await ctx.secrets.resolve(SECRET_URL)) || DEFAULT_URL;
  const client = new LivingMemoryClient(url);

  // Verify connectivity at startup
  const available = await client.healthCheck();
  if (available) {
    ctx.activity.log("memory.connected", { url });
  } else {
    ctx.activity.log("memory.unavailable", { url });
    console.warn(
      `[living-memory] Server not reachable at ${url}. ` +
        `Start it with: dreamcatcher serve`
    );
  }

  // ── Tool: recall_team_memory ──────────────────────────────

  ctx.tools.register({
    name: TOOL_RECALL,
    description:
      "Search the team's shared memory for relevant context about users, " +
      "projects, decisions, preferences, and patterns. Use this when you " +
      "need to recall information that other agents on the team have learned.",
    schema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "What to search for in team memory",
        },
      },
      required: ["query"],
    },
    handle: async (
      input: Record<string, unknown>,
      toolCtx: ToolContext
    ) => {
      const companyId = toolCtx.scope[1];
      try {
        const result = await client.recall(companyId, input.query as string);
        ctx.metrics.write("memory.recall.latency_ms", result.latency_ms);
        ctx.activity.log("memory.recall", {
          team_id: companyId,
          query: input.query,
          memories_found: result.memories.length,
          source: result.source,
        });
        return result;
      } catch (err) {
        ctx.activity.log("memory.recall.error", {
          team_id: companyId,
          error: String(err),
        });
        return { response: "Memory recall failed — server may be unavailable.", memories: [] };
      }
    },
  });

  // ── Tool: save_to_team_memory ─────────────────────────────

  ctx.tools.register({
    name: TOOL_SAVE,
    description:
      "Save a conversation transcript to the team's shared memory. " +
      "The transcript will be processed during the nightly training cycle " +
      "and become available to all agents on the team.",
    schema: {
      type: "object",
      properties: {
        transcript: {
          type: "string",
          description: "The full conversation transcript to save",
        },
        agent_name: {
          type: "string",
          description: "Name of the agent saving the session",
        },
      },
      required: ["transcript"],
    },
    handle: async (
      input: Record<string, unknown>,
      toolCtx: ToolContext
    ) => {
      const companyId = toolCtx.scope[1];
      const agentName = (input.agent_name as string) || "paperclip-agent";
      try {
        const result = await client.ingest(
          companyId,
          input.transcript as string,
          agentName
        );
        ctx.activity.log("memory.ingest", {
          team_id: companyId,
          session_id: result.session_id,
          agent_name: agentName,
          chars: (input.transcript as string).length,
        });
        return result;
      } catch (err) {
        ctx.activity.log("memory.ingest.error", {
          team_id: companyId,
          error: String(err),
        });
        return { status: "error", error: String(err) };
      }
    },
  });

  // ── Tool: team_memory_status ──────────────────────────────

  ctx.tools.register({
    name: TOOL_STATUS,
    description:
      "Check the health and statistics of the team's shared memory. " +
      "Shows active memories, training runs, and model status.",
    schema: {
      type: "object",
      properties: {},
    },
    handle: async (
      _input: Record<string, unknown>,
      toolCtx: ToolContext
    ) => {
      const companyId = toolCtx.scope[1];
      try {
        const stats = await client.getStats(companyId);
        ctx.activity.log("memory.status_check", { team_id: companyId });
        return stats;
      } catch (err) {
        return { status: "error", error: String(err) };
      }
    },
  });

  // ── Event: auto-ingest on agent run completion ────────────

  ctx.events.subscribe(
    "agent.run.completed",
    async (event: PluginEvent) => {
      const companyId = event.scope[1];
      const payload = event.payload;
      // Only ingest full transcripts — summaries strip the specific details
      // (names, decisions, reasoning, exact phrasing) that make memories useful.
      const transcript = payload.transcript as string | undefined;
      if (!transcript) {
        if (payload.summary) {
          ctx.activity.log("memory.auto_ingest.skipped", {
            team_id: companyId,
            reason: "only summary available, need full transcript",
          });
        }
        return;
      }

      const agentName = (payload.agentName as string) || "unknown";
      try {
        const result = await client.ingest(companyId, transcript, agentName);
        ctx.activity.log("memory.auto_ingest", {
          team_id: companyId,
          session_id: result.session_id,
          agent_name: agentName,
          chars: transcript.length,
        });
      } catch (err) {
        ctx.activity.log("memory.auto_ingest.error", {
          team_id: companyId,
          error: String(err),
        });
      }
    }
  );

  // ── Job: nightly training trigger ─────────────────────────

  ctx.jobs.register({
    key: JOB_NIGHTLY,
    schedule: "0 3 * * *", // 3 AM daily
    handle: async () => {
      try {
        await client.triggerNightly();
        ctx.activity.log("memory.nightly_triggered", {
          timestamp: new Date().toISOString(),
          status: "started",
        });
      } catch (err) {
        ctx.activity.log("memory.nightly_trigger_failed", {
          timestamp: new Date().toISOString(),
          error: String(err),
        });
    },
  });
}

// ── Plugin entry point ──────────────────────────────────────

// When Paperclip SDK is available, use:
// import { definePlugin } from "@paperclip/plugin-sdk";
// export default definePlugin({ setup });

// For now, export setup directly for manual integration
export default { setup };
