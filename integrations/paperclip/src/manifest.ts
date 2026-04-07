/**
 * Living Memory Plugin Manifest for Paperclip
 * =============================================
 * Declares capabilities needed by the plugin.
 */

// NOTE: This file documents the manifest that would be provided to
// Paperclip's defineManifest(). The exact import path depends on the
// Paperclip SDK version installed. Uncomment when building against
// the actual Paperclip plugin SDK.

// import { defineManifest } from "@paperclip/plugin-sdk";

export const manifest = {
  apiVersion: 1,
  id: "living-memory",
  version: "0.1.0",
  name: "Living Memory",
  description:
    "Shared team memory for AI agents — knowledge lives in weights, not retrieval. " +
    "All agents in a company share a memory pool with nightly model re-fine-tuning.",

  capabilities: {
    // Register memory tools for agents to invoke
    "tools.register": {},
    // Subscribe to agent run events for auto-ingest
    "events.subscribe": {},
    // Schedule nightly training trigger
    "jobs.register": {},
    // Call Living Memory HTTP API
    "http.outbound": {},
    // Resolve LIVING_MEMORY_URL secret
    "secrets.resolve": {},
    // Cache team connection state
    "state.read": {},
    "state.write": {},
    // Log memory operations for audit
    "activity.log": {},
    // Track recall latency and hit rates
    "metrics.write": {},
  },

  secrets: ["LIVING_MEMORY_URL"],

  worker: {
    path: "./dist/worker.js",
  },
};
