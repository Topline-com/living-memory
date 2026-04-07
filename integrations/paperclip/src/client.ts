/**
 * Living Memory HTTP Client for Paperclip Plugin
 * ================================================
 * Thin wrapper over the Living Memory team API endpoints.
 * Maps Paperclip companyId → Living Memory team_id.
 */

import { REQUEST_TIMEOUT_MS, MAX_CONSECUTIVE_FAILURES } from "./constants.js";

// ── Response types ──────────────────────────────────────────────

export interface Memory {
  category: string;
  content: string;
  confidence: number;
}

export interface MemoryResponse {
  response: string;
  memories: Memory[];
  source: string;
  latency_ms: number;
}

export interface IngestResult {
  team_id: string;
  session_id: string;
  status: string;
  memories_extracted?: number;
  extraction_error?: string;
}

export interface TeamStats {
  team_id: string;
  total_sessions: number;
  unprocessed_sessions: number;
  active_memories: number;
  total_training_examples: number;
  training_runs: number;
  memories_by_category: Record<string, number>;
}

export interface HealthStatus {
  team_id: string;
  status: string;
  model_path: string;
  model_age_hours: number | null;
  stats: TeamStats;
}

// ── Client ──────────────────────────────────────────────────────

export class LivingMemoryClient {
  private baseUrl: string;
  private consecutiveFailures = 0;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  get circuitOpen(): boolean {
    return this.consecutiveFailures >= MAX_CONSECUTIVE_FAILURES;
  }

  private recordSuccess(): void {
    this.consecutiveFailures = 0;
  }

  private recordFailure(): void {
    this.consecutiveFailures++;
  }

  private async request<T>(
    path: string,
    options: RequestInit = {}
  ): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const resp = await fetch(`${this.baseUrl}${path}`, {
        ...options,
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
      }

      const data = (await resp.json()) as T;
      this.recordSuccess();
      return data;
    } catch (err) {
      this.recordFailure();
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  // ── Team-scoped operations ──────────────────────────────────

  async ingest(
    teamId: string,
    transcript: string,
    agentName: string = "paperclip-agent"
  ): Promise<IngestResult> {
    return this.request<IngestResult>(`/teams/${teamId}/ingest`, {
      method: "POST",
      body: JSON.stringify({ transcript, agent_name: agentName }),
    });
  }

  async recall(
    teamId: string,
    query: string,
    maxTokens: number = 256
  ): Promise<MemoryResponse> {
    return this.request<MemoryResponse>(`/teams/${teamId}/recall`, {
      method: "POST",
      body: JSON.stringify({ query, max_tokens: maxTokens }),
    });
  }

  async getContext(
    teamId: string,
    query: string,
    agentName: string = "paperclip-agent",
    maxTokens: number = 512
  ): Promise<MemoryResponse> {
    return this.request<MemoryResponse>(`/teams/${teamId}/context`, {
      method: "POST",
      body: JSON.stringify({
        query,
        agent_name: agentName,
        max_tokens: maxTokens,
      }),
    });
  }

  async getMemories(
    teamId: string,
    category?: string,
    limit: number = 50
  ): Promise<{ team_id: string; memories: Memory[]; count: number }> {
    const params = new URLSearchParams({ limit: String(limit) });
    if (category) params.set("category", category);
    return this.request(`/teams/${teamId}/memories?${params}`);
  }

  async getStats(teamId: string): Promise<TeamStats> {
    return this.request<TeamStats>(`/teams/${teamId}/stats`);
  }

  async health(teamId: string): Promise<HealthStatus> {
    return this.request<HealthStatus>(`/teams/${teamId}/health`);
  }

  async healthCheck(): Promise<boolean> {
    try {
      const resp = await fetch(`${this.baseUrl}/health`, {
        signal: AbortSignal.timeout(3000),
      });
      return resp.ok;
    } catch {
      return false;
    }
  }
}
