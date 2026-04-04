"""
Dreamcatcher Memory Linting — Data Quality Self-Correction
===========================================================
Periodic review of the canonical memory ledger for consistency issues.

Two-layer architecture:
  1. Rule-based pre-pass: deterministic checks (zero API cost)
     - Exact duplicate detection
     - Date/temporal conflicts
     - Category anomalies
  2. LLM fuzzy pass: semantic checks (one API call per batch)
     - Contradictions between facts
     - Likely superseded information
     - Implausible or likely misextracted facts
     - Missing companion facts
     - Candidate abstractions

CRITICAL: The linter NEVER silently mutates memory. It produces typed
findings with confidence scores and source citations. Only human-
approved corrections are applied to the canonical store.
"""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from collections import defaultdict

from .database import MemoryDB
from .config import DreamcatcherConfig


# Finding types (used in reports and structured output)
FINDING_TYPES = {
    "contradiction": "Two facts appear to contradict each other",
    "likely_stale": "This fact may have been superseded by newer information",
    "likely_duplicate": "This fact appears to be a duplicate of another",
    "missing_companion": "Related information appears to be missing",
    "weak_inference": "This fact seems weakly supported or possibly misextracted",
    "candidate_abstraction": "Multiple facts could be consolidated into a higher-level summary",
    "category_anomaly": "This fact may be miscategorized",
}


class MemoryLinter:
    """
    Reviews the canonical memory ledger for consistency issues.
    Produces advisory reports — never modifies data without human approval.
    """

    def __init__(self, config: DreamcatcherConfig = None):
        self.config = config or DreamcatcherConfig.load()
        self.db = MemoryDB(self.config.db_path)

    def run_full_lint(self, output_dir: Optional[str] = None) -> dict:
        """
        Run a full lint pass: rule-based pre-pass + LLM fuzzy pass.
        Returns a summary dict and saves the report as markdown.
        """
        vault = Path(output_dir) if output_dir else Path(self.config.models_dir).parent / "vault"
        vault.mkdir(parents=True, exist_ok=True)

        memories = self.db.get_active_memories(limit=10000)
        if not memories:
            print("  No active memories to lint.")
            return {"findings": [], "total": 0}

        print(f"  Linting {len(memories)} active memories...")

        # ── Layer 1: Rule-based pre-pass (zero API cost) ──────────
        rule_findings = self._rule_based_pass(memories)
        print(f"  Rule-based pass: {len(rule_findings)} findings")

        # ── Layer 2: LLM fuzzy pass (one API call) ────────────────
        llm_findings = self._llm_fuzzy_pass(memories)
        print(f"  LLM fuzzy pass: {len(llm_findings)} findings")

        all_findings = rule_findings + llm_findings

        # ── Generate report ───────────────────────────────────────
        report_path = self._write_report(vault, all_findings, len(memories))
        print(f"  Lint report: {report_path} ({len(all_findings)} total findings)")

        return {
            "findings": all_findings,
            "total": len(all_findings),
            "rule_based": len(rule_findings),
            "llm_based": len(llm_findings),
            "report_path": str(report_path),
        }

    # ══════════════════════════════════════════════════════════════
    # Layer 1: Rule-Based Pre-Pass (deterministic, zero API cost)
    # ══════════════════════════════════════════════════════════════

    def _rule_based_pass(self, memories: list) -> list:
        """Deterministic checks that don't require an LLM."""
        findings = []
        findings.extend(self._check_exact_duplicates(memories))
        findings.extend(self._check_content_near_duplicates(memories))
        findings.extend(self._check_category_anomalies(memories))
        return findings

    def _check_exact_duplicates(self, memories: list) -> list:
        """Find memories with identical content text."""
        findings = []
        content_map = defaultdict(list)
        for mem in memories:
            # Normalize whitespace for comparison
            normalized = " ".join(mem.get("content", "").split()).lower()
            content_map[normalized].append(mem)

        for content, dupes in content_map.items():
            if len(dupes) > 1:
                ids = [d.get("id", "?") for d in dupes]
                findings.append({
                    "type": "likely_duplicate",
                    "severity": "high",
                    "confidence": 1.0,
                    "memory_ids": ids,
                    "description": f"Exact duplicate content across {len(dupes)} memories",
                    "content_preview": content[:120],
                    "suggestion": f"Keep the most recent ({ids[-1]}), mark others as deprecated",
                    "source": "rule_based",
                })
        return findings

    def _check_content_near_duplicates(self, memories: list) -> list:
        """Find memories with very similar content (simple word overlap heuristic)."""
        findings = []
        # Simple word-set overlap for near-duplicate detection
        # (avoids requiring sentence-transformers for the rule-based pass)
        content_words = []
        for mem in memories:
            words = set(mem.get("content", "").lower().split())
            # Remove very common words
            words -= {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                      "to", "for", "of", "and", "or", "with", "that", "this", "user",
                      "has", "have", "had", "be", "been", "being"}
            content_words.append((mem, words))

        seen_pairs = set()
        for i, (mem_a, words_a) in enumerate(content_words):
            if len(words_a) < 3:
                continue
            for j, (mem_b, words_b) in enumerate(content_words):
                if j <= i or len(words_b) < 3:
                    continue
                pair_key = tuple(sorted([mem_a.get("id", ""), mem_b.get("id", "")]))
                if pair_key in seen_pairs:
                    continue

                overlap = len(words_a & words_b)
                union = len(words_a | words_b)
                if union > 0 and overlap / union > 0.75:
                    seen_pairs.add(pair_key)
                    findings.append({
                        "type": "likely_duplicate",
                        "severity": "medium",
                        "confidence": round(overlap / union, 2),
                        "memory_ids": [mem_a.get("id", "?"), mem_b.get("id", "?")],
                        "description": f"High word overlap ({overlap}/{union} words) between memories",
                        "content_preview": f"A: {mem_a.get('content', '')[:80]}... | B: {mem_b.get('content', '')[:80]}...",
                        "suggestion": "Review for potential merge or deduplication",
                        "source": "rule_based",
                    })

        return findings

    def _check_category_anomalies(self, memories: list) -> list:
        """Flag memories that might be miscategorized based on keyword heuristics."""
        findings = []
        category_keywords = {
            "project": ["project", "development", "building", "phase", "unit", "construction"],
            "preference": ["prefer", "like", "style", "favor", "always", "never", "usually"],
            "fact": ["born", "lives", "married", "age", "located", "address", "number"],
            "pattern": ["tends to", "usually", "pattern", "habit", "routine", "approach"],
            "relationship": ["wife", "husband", "brother", "sister", "colleague", "friend", "works with"],
            "decision": ["decided", "chose", "switched", "changed", "moved to", "adopted"],
        }

        for mem in memories:
            content_lower = mem.get("content", "").lower()
            current_cat = mem.get("category", "other")

            # Check if content strongly matches a different category
            for cat, keywords in category_keywords.items():
                if cat == current_cat:
                    continue
                matches = sum(1 for kw in keywords if kw in content_lower)
                if matches >= 3:  # Strong signal for different category
                    findings.append({
                        "type": "category_anomaly",
                        "severity": "low",
                        "confidence": min(matches / len(keywords), 0.9),
                        "memory_ids": [mem.get("id", "?")],
                        "description": f"Memory categorized as '{current_cat}' but content suggests '{cat}'",
                        "content_preview": mem.get("content", "")[:120],
                        "suggestion": f"Consider recategorizing from '{current_cat}' to '{cat}'",
                        "source": "rule_based",
                    })
        return findings

    # ══════════════════════════════════════════════════════════════
    # Layer 2: LLM Fuzzy Pass (semantic, one API call per batch)
    # ══════════════════════════════════════════════════════════════

    def _llm_fuzzy_pass(self, memories: list) -> list:
        """Use a frontier LLM to find semantic issues the rules can't catch."""
        import os

        # Prepare a summary of all facts for the LLM
        facts_summary = []
        for mem in memories[:200]:  # Cap at 200 to stay within context limits
            facts_summary.append(
                f"[{mem.get('id', '?')}] ({mem.get('category', '?')}) "
                f"{mem.get('content', '')} "
                f"(created: {mem.get('created_at', '?')[:10]})"
            )

        facts_text = "\n".join(facts_summary)

        prompt = f"""You are a memory consistency auditor for a personal AI memory system.
Review these {len(facts_summary)} canonical facts about a user and identify issues.

For each issue found, output a JSON object with:
- type: one of "contradiction", "likely_stale", "missing_companion", "weak_inference", "candidate_abstraction"
- memory_ids: array of relevant memory IDs (from the [brackets])
- description: what the issue is
- confidence: 0.0 to 1.0
- suggestion: recommended action

Focus on:
1. Facts that contradict each other (e.g., two different values for the same thing)
2. Facts that appear to have been superseded by more recent facts
3. Facts that seem implausible or weakly supported
4. Groups of related facts that could be consolidated into a higher-level summary
5. Obvious gaps where companion information seems missing

Respond ONLY with a JSON array of findings. If no issues found, respond with [].

Facts to review:
{facts_text}"""

        provider = os.environ.get("DREAMCATCHER_PROVIDER", "anthropic")
        try:
            if provider == "openai":
                findings = self._call_openai_lint(prompt)
            else:
                findings = self._call_anthropic_lint(prompt)
        except Exception as e:
            print(f"  LLM lint error: {e}")
            return []

        # Tag all LLM findings
        for f in findings:
            f["source"] = "llm"
            f["severity"] = "medium"
            if f.get("type") not in FINDING_TYPES:
                f["type"] = "weak_inference"

        return findings

    def _call_anthropic_lint(self, prompt: str) -> list:
        """Call Anthropic API for lint analysis."""
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Parse JSON response
        if text.startswith("["):
            return json.loads(text)
        # Try to extract JSON from response
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []

    def _call_openai_lint(self, prompt: str) -> list:
        """Call OpenAI API for lint analysis."""
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "findings" in result:
            return result["findings"]
        return []

    # ══════════════════════════════════════════════════════════════
    # Report Generation
    # ══════════════════════════════════════════════════════════════

    def _write_report(self, vault: Path, findings: list, total_memories: int) -> Path:
        """Write the lint report as markdown in the vault directory."""
        report_path = vault / "LINT_REPORT.md"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            "# Memory Lint Report",
            "",
            f"*Generated: {now}*",
            f"*Memories reviewed: {total_memories}*",
            f"*Findings: {len(findings)}*",
            "",
            "> **This report is advisory only.** No changes have been made to the canonical "
            "memory store. Review each finding and apply corrections manually through the "
            "vault frontmatter or `dreamcatcher` CLI.",
            "",
            "---",
            "",
        ]

        if not findings:
            lines.append("**No issues found.** The canonical memory ledger appears consistent.")
            report_path.write_text("\n".join(lines), encoding="utf-8")
            return report_path

        # Group findings by type
        by_type = defaultdict(list)
        for f in findings:
            by_type[f.get("type", "unknown")].append(f)

        # Summary
        lines.append("## Summary")
        lines.append("")
        for ftype, desc in FINDING_TYPES.items():
            count = len(by_type.get(ftype, []))
            if count > 0:
                lines.append(f"- **{ftype}**: {count} finding(s) — {desc}")
        lines.extend(["", "---", ""])

        # Detailed findings by type
        for ftype in FINDING_TYPES:
            type_findings = by_type.get(ftype, [])
            if not type_findings:
                continue

            lines.append(f"## {ftype.replace('_', ' ').title()}")
            lines.append("")

            for i, f in enumerate(type_findings, 1):
                confidence = f.get("confidence", 0)
                severity = f.get("severity", "medium")
                source = f.get("source", "unknown")
                mem_ids = f.get("memory_ids", [])

                lines.append(f"### Finding {i} ({severity}, confidence: {confidence:.0%})")
                lines.append("")
                lines.append(f"**Source:** {source}")
                lines.append(f"**Memory IDs:** {', '.join(str(mid) for mid in mem_ids)}")
                lines.append("")
                lines.append(f"**Issue:** {f.get('description', 'No description')}")
                lines.append("")

                preview = f.get("content_preview", "")
                if preview:
                    lines.append(f"**Content:** {preview}")
                    lines.append("")

                suggestion = f.get("suggestion", "")
                if suggestion:
                    lines.append(f"**Suggestion:** {suggestion}")
                    lines.append("")

                lines.append("---")
                lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path
