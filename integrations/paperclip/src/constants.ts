/**
 * Living Memory Plugin Constants for Paperclip
 */

export const PLUGIN_ID = "living-memory";
export const PLUGIN_VERSION = "0.1.0";

// Tool names registered with Paperclip
export const TOOL_RECALL = "recall_team_memory";
export const TOOL_SAVE = "save_to_team_memory";
export const TOOL_STATUS = "team_memory_status";

// Job keys
export const JOB_NIGHTLY = "nightly-memory-training";

// Secrets
export const SECRET_URL = "LIVING_MEMORY_URL";

// Defaults
export const DEFAULT_URL = "http://localhost:8420";
export const REQUEST_TIMEOUT_MS = 10000;
export const MAX_CONSECUTIVE_FAILURES = 5;
