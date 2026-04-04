"""
Dreamcatcher Wiki Export — Browsable Knowledge Surface
======================================================
Exports the canonical memory ledger (SQLite) as a structured directory
of markdown files compatible with Obsidian, Logseq, or any markdown tool.

Design principles:
  - SQLite is the SOLE canonical store. The vault is a curated VIEW.
  - Each memory entry carries a stable memory_id in YAML frontmatter.
  - User corrections operate through structured frontmatter actions
    (status: deprecated, delete: true) that map to canonical-store operations.
  - The vault is regenerated on each export; human edits to frontmatter
    are preserved through a sync-back pass before regeneration.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .database import MemoryDB
from .config import DreamcatcherConfig


class WikiExporter:
    """
    Exports the canonical memory ledger as browsable markdown files.
    Produces one file per category plus an index, cross-reference map,
    and stats dashboard.
    """

    def __init__(self, config: DreamcatcherConfig = None):
        self.config = config or DreamcatcherConfig.load()
        self.db = MemoryDB(self.config.db_path)
        self.vault_dir = Path(self.config.models_dir).parent / "vault"

    def export(self, output_dir: Optional[str] = None) -> str:
        """
        Export the full canonical ledger to markdown.
        Returns the path to the generated vault directory.
        """
        vault = Path(output_dir) if output_dir else self.vault_dir
        vault.mkdir(parents=True, exist_ok=True)

        # First, sync back any human edits from existing vault files
        self._sync_edits_from_vault(vault)

        # Gather all active memories
        memories = self.db.get_active_memories(limit=10000)
        if not memories:
            print("  No active memories to export.")
            return str(vault)

        # Group by category
        by_category = {}
        for mem in memories:
            cat = mem.get("category", "other")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(mem)

        # Get training pair counts per memory for organic reinforcement visibility
        pair_counts = self._get_pair_counts()

        # Generate category files
        for category, mems in sorted(by_category.items()):
            self._write_category_file(vault, category, mems, pair_counts)

        # Generate index file
        self._write_index(vault, by_category, pair_counts)

        # Generate stats dashboard
        self._write_stats(vault, memories, pair_counts)

        total_memories = len(memories)
        total_categories = len(by_category)
        print(f"  Exported {total_memories} memories across {total_categories} categories → {vault}")
        return str(vault)

    def _write_category_file(self, vault: Path, category: str, memories: list,
                              pair_counts: dict):
        """Write a single category markdown file with YAML frontmatter per entry."""
        filepath = vault / f"{category}.md"
        lines = [
            f"# {category.title()}",
            "",
            f"*{len(memories)} memories in this category. "
            f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
            "",
            "---",
            "",
        ]

        # Sort by creation date, most recent first
        sorted_mems = sorted(memories, key=lambda m: m.get("created_at", ""), reverse=True)

        for mem in sorted_mems:
            mem_id = mem.get("id", "unknown")
            content = mem.get("content", "")
            confidence = mem.get("confidence", 1.0)
            created = mem.get("created_at", "unknown")
            session_id = mem.get("session_id", "unknown")
            n_pairs = pair_counts.get(mem_id, 0)

            # YAML frontmatter block for this entry
            lines.append(f"### `{mem_id}`")
            lines.append("")
            lines.append("```yaml")
            lines.append(f"memory_id: {mem_id}")
            lines.append(f"category: {category}")
            lines.append(f"confidence: {confidence}")
            lines.append(f"created_at: {created}")
            lines.append(f"session_id: {session_id}")
            lines.append(f"training_pairs: {n_pairs}")
            lines.append(f"status: current")
            lines.append(f"# To mark for deletion, change to: status: delete")
            lines.append(f"# To mark as outdated, change to: status: deprecated")
            lines.append("```")
            lines.append("")
            lines.append(f"> {content}")
            lines.append("")

            # Show organic reinforcement density indicator
            if n_pairs > 0:
                density = "█" * min(n_pairs, 30)
                lines.append(f"*Organic density: {density} ({n_pairs} training pairs)*")
                lines.append("")

            lines.append("---")
            lines.append("")

        filepath.write_text("\n".join(lines), encoding="utf-8")

    def _write_index(self, vault: Path, by_category: dict, pair_counts: dict):
        """Write the vault index file with cross-references."""
        filepath = vault / "INDEX.md"
        lines = [
            "# Dreamcatcher Memory Vault",
            "",
            f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
            "",
            "This vault is a **read-mostly projection** of the canonical SQLite memory ledger. "
            "SQLite is the source of truth. Edits to `status` fields in YAML frontmatter "
            "will be synced back to SQLite on the next export.",
            "",
            "---",
            "",
            "## Categories",
            "",
        ]

        for category, mems in sorted(by_category.items()):
            total_pairs = sum(pair_counts.get(m.get("id", ""), 0) for m in mems)
            lines.append(f"- **[{category.title()}]({category}.md)** — "
                        f"{len(mems)} memories, {total_pairs} training pairs")

        lines.extend([
            "",
            "---",
            "",
            "## How to Edit",
            "",
            "To correct a memory, find its YAML frontmatter block and change the `status` field:",
            "",
            "- `status: current` — active memory, included in training (default)",
            "- `status: deprecated` — kept in database but excluded from training",
            "- `status: delete` — will be removed from the canonical store on next sync",
            "",
            "Do **not** edit the `memory_id`, `session_id`, or `created_at` fields — "
            "these are structural identifiers.",
            "",
            "Run `dreamcatcher wiki --sync` to apply changes back to SQLite.",
            "",
            "## Related Files",
            "",
            "- **[Stats](STATS.md)** — Memory statistics and organic reinforcement overview",
        ])

        # Add lint report link if it exists
        lint_path = vault / "LINT_REPORT.md"
        if lint_path.exists():
            lines.append("- **[Lint Report](LINT_REPORT.md)** — Weekly consistency check findings")

        lines.append("")
        filepath.write_text("\n".join(lines), encoding="utf-8")

    def _write_stats(self, vault: Path, memories: list, pair_counts: dict):
        """Write a stats dashboard showing memory health and organic reinforcement."""
        filepath = vault / "STATS.md"
        stats = self.db.stats()

        # Calculate organic reinforcement distribution
        pair_values = [pair_counts.get(m.get("id", ""), 0) for m in memories]
        high_density = sum(1 for p in pair_values if p >= 15)
        medium_density = sum(1 for p in pair_values if 5 <= p < 15)
        low_density = sum(1 for p in pair_values if 0 < p < 5)
        zero_density = sum(1 for p in pair_values if p == 0)

        lines = [
            "# Memory Statistics",
            "",
            f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
            "",
            "## Overview",
            "",
            f"- **Total sessions:** {stats.get('total_sessions', 0)}",
            f"- **Unprocessed sessions:** {stats.get('unprocessed_sessions', 0)}",
            f"- **Active memories:** {stats.get('active_memories', 0)}",
            f"- **Total training pairs:** {stats.get('total_training_examples', 0)}",
            f"- **Training runs:** {stats.get('training_runs', 0)}",
            "",
            "## Organic Reinforcement Distribution",
            "",
            "Memories that matter get discussed more often, accumulating more training pairs.",
            "",
            f"- **High density (15+ pairs):** {high_density} memories — deeply encoded",
            f"- **Medium density (5-14 pairs):** {medium_density} memories — moderately encoded",
            f"- **Low density (1-4 pairs):** {low_density} memories — lightly encoded",
            f"- **No training pairs:** {zero_density} memories — metadata only",
            "",
            "## By Category",
            "",
        ]

        for cat, count in sorted(stats.get("memories_by_category", {}).items()):
            lines.append(f"- **{cat}:** {count} memories")

        lines.append("")
        filepath.write_text("\n".join(lines), encoding="utf-8")

    def _sync_edits_from_vault(self, vault: Path):
        """
        Read existing vault files and apply any structured edits back to SQLite.
        Only processes status changes in YAML frontmatter blocks — does not
        parse freeform markdown edits to prevent bidirectional drift.
        """
        if not vault.exists():
            return

        synced = 0
        for md_file in vault.glob("*.md"):
            if md_file.name in ("INDEX.md", "STATS.md", "LINT_REPORT.md"):
                continue

            content = md_file.read_text(encoding="utf-8")

            # Find all YAML frontmatter blocks within the file
            # Pattern: ```yaml ... memory_id: xxx ... status: yyy ... ```
            blocks = re.findall(
                r'```yaml\n(.*?)```',
                content, re.DOTALL
            )

            for block in blocks:
                # Extract memory_id and status
                id_match = re.search(r'memory_id:\s*(\S+)', block)
                status_match = re.search(r'status:\s*(\S+)', block)

                if not id_match or not status_match:
                    continue

                memory_id = id_match.group(1)
                status = status_match.group(1).lower()

                if status == "delete":
                    # Mark as inactive in the canonical store
                    with self.db._conn() as conn:
                        conn.execute(
                            "UPDATE memories SET active = 0 WHERE id = ?",
                            (memory_id,)
                        )
                    synced += 1
                    print(f"  Synced: deleted memory {memory_id}")

                elif status == "deprecated":
                    # Mark as inactive but keep in database
                    with self.db._conn() as conn:
                        conn.execute(
                            "UPDATE memories SET active = 0 WHERE id = ?",
                            (memory_id,)
                        )
                    synced += 1
                    print(f"  Synced: deprecated memory {memory_id}")

        if synced > 0:
            print(f"  Applied {synced} vault edits to canonical store.")

    def _get_pair_counts(self) -> dict:
        """Get the number of training pairs associated with each memory_id."""
        counts = {}
        with self.db._conn() as conn:
            rows = conn.execute(
                "SELECT memory_ids, COUNT(*) as cnt FROM training_examples GROUP BY memory_ids"
            ).fetchall()
            for row in rows:
                try:
                    ids = json.loads(row["memory_ids"])
                    for mid in ids:
                        counts[mid] = counts.get(mid, 0) + row["cnt"]
                except (json.JSONDecodeError, TypeError):
                    pass
        return counts
