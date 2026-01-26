"""Cumulative token usage tracking for TLDR.

Tracks every tldr query and records:
- Timestamp - when the query was made
- Project - which project/directory
- Command - which tldr command (tree, structure, context, calls, etc.)
- Raw tokens - estimated tokens if reading raw files
- TLDR tokens - actual tokens returned by tldr
- Savings - percentage saved

Storage: JSONL file at ~/.tldr/stats/usage.jsonl
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class UsageTracker:
    """Tracks cumulative token usage across all tldr queries.

    Stores usage records in append-only JSONL format for durability
    and efficient aggregation.
    """

    def __init__(self, stats_dir: Path | str | None = None):
        """Initialize usage tracker.

        Args:
            stats_dir: Directory for stats files. Defaults to ~/.tldr/stats/
        """
        if stats_dir is None:
            self.stats_dir = Path.home() / ".tldr" / "stats"
        else:
            self.stats_dir = Path(stats_dir)

        self.usage_file = self.stats_dir / "usage.jsonl"

    def record_usage(
        self,
        command: str,
        project: str,
        raw_tokens: int,
        tldr_tokens: int,
    ) -> None:
        """Record a single tldr command usage.

        Args:
            command: The tldr command (tree, structure, context, etc.)
            project: The project path/directory
            raw_tokens: Estimated tokens if reading raw files
            tldr_tokens: Actual tokens returned by tldr
        """
        # Calculate savings
        if raw_tokens > 0:
            savings_percent = ((raw_tokens - tldr_tokens) / raw_tokens) * 100
        else:
            savings_percent = 0.0

        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "command": command,
            "project": project,
            "raw_tokens": raw_tokens,
            "tldr_tokens": tldr_tokens,
            "savings_percent": round(savings_percent, 1),
        }

        # Ensure directory exists
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        # Append to JSONL file
        with open(self.usage_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _read_all_records(self) -> list[dict[str, Any]]:
        """Read all usage records from the JSONL file.

        Returns:
            List of all usage records
        """
        if not self.usage_file.exists():
            return []

        records = []
        with open(self.usage_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return records

    def get_cumulative_stats(self) -> dict[str, Any]:
        """Get cumulative statistics across all recorded usage.

        Returns:
            Dict with total_queries, total_raw_tokens, total_tldr_tokens,
            total_savings_percent, period_start, period_end
        """
        records = self._read_all_records()

        if not records:
            return {
                "total_queries": 0,
                "total_raw_tokens": 0,
                "total_tldr_tokens": 0,
                "total_savings_percent": 0.0,
                "period_start": None,
                "period_end": None,
            }

        total_raw = sum(r.get("raw_tokens", 0) for r in records)
        total_tldr = sum(r.get("tldr_tokens", 0) for r in records)

        if total_raw > 0:
            savings_percent = ((total_raw - total_tldr) / total_raw) * 100
        else:
            savings_percent = 0.0

        # Get date range
        timestamps = [r.get("timestamp") for r in records if r.get("timestamp")]
        period_start = min(timestamps) if timestamps else None
        period_end = max(timestamps) if timestamps else None

        return {
            "total_queries": len(records),
            "total_raw_tokens": total_raw,
            "total_tldr_tokens": total_tldr,
            "total_savings_percent": round(savings_percent, 1),
            "period_start": period_start,
            "period_end": period_end,
        }

    def get_stats_by_project(self) -> dict[str, dict[str, Any]]:
        """Get statistics grouped by project.

        Returns:
            Dict mapping project path to stats (queries, raw_tokens, tldr_tokens, savings_percent)
        """
        records = self._read_all_records()

        by_project: dict[str, dict[str, Any]] = {}

        for record in records:
            project = record.get("project", "unknown")

            if project not in by_project:
                by_project[project] = {
                    "queries": 0,
                    "raw_tokens": 0,
                    "tldr_tokens": 0,
                }

            by_project[project]["queries"] += 1
            by_project[project]["raw_tokens"] += record.get("raw_tokens", 0)
            by_project[project]["tldr_tokens"] += record.get("tldr_tokens", 0)

        # Calculate savings for each project
        for project, stats in by_project.items():
            if stats["raw_tokens"] > 0:
                stats["savings_percent"] = round(
                    ((stats["raw_tokens"] - stats["tldr_tokens"]) / stats["raw_tokens"]) * 100,
                    1,
                )
            else:
                stats["savings_percent"] = 0.0

        return by_project

    def get_stats_by_command(self) -> dict[str, dict[str, Any]]:
        """Get statistics grouped by command type.

        Returns:
            Dict mapping command name to stats (queries, raw_tokens, tldr_tokens, savings_percent)
        """
        records = self._read_all_records()

        by_command: dict[str, dict[str, Any]] = {}

        for record in records:
            command = record.get("command", "unknown")

            if command not in by_command:
                by_command[command] = {
                    "queries": 0,
                    "raw_tokens": 0,
                    "tldr_tokens": 0,
                }

            by_command[command]["queries"] += 1
            by_command[command]["raw_tokens"] += record.get("raw_tokens", 0)
            by_command[command]["tldr_tokens"] += record.get("tldr_tokens", 0)

        # Calculate savings for each command
        for command, stats in by_command.items():
            if stats["raw_tokens"] > 0:
                stats["savings_percent"] = round(
                    ((stats["raw_tokens"] - stats["tldr_tokens"]) / stats["raw_tokens"]) * 100,
                    1,
                )
            else:
                stats["savings_percent"] = 0.0

        return by_command

    def format_cumulative_report(self) -> str:
        """Format a cumulative stats report for display.

        Returns:
            Formatted string report with totals, by-command, by-project, and cost estimates
        """
        cumulative = self.get_cumulative_stats()

        if cumulative["total_queries"] == 0:
            return "No usage data recorded yet. Run some tldr commands first!"

        by_command = self.get_stats_by_command()
        by_project = self.get_stats_by_project()

        # Format period
        period_str = ""
        if cumulative["period_start"] and cumulative["period_end"]:
            try:
                start = datetime.fromisoformat(cumulative["period_start"])
                end = datetime.fromisoformat(cumulative["period_end"])
                days = (end - start).days + 1
                period_str = f"  Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({days} days)"
            except (ValueError, TypeError):
                period_str = ""

        lines = [
            "",
            "=" * 70,
            "  TLDR Cumulative Token Savings",
            "=" * 70,
            f"  Total Queries: {cumulative['total_queries']:,}",
        ]

        if period_str:
            lines.append(period_str)

        lines.extend([
            "=" * 70,
            "",
            f"  {'Totals':<25} {'Raw':>12} {'TLDR':>12} {'Savings':>10}",
            f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}",
            f"  {'All time':<25} {cumulative['total_raw_tokens']:>12,} {cumulative['total_tldr_tokens']:>12,} {cumulative['total_savings_percent']:>9.1f}%",
            "",
            "  By Command:",
        ])

        # Sort commands by raw_tokens descending
        sorted_commands = sorted(
            by_command.items(),
            key=lambda x: x[1]["raw_tokens"],
            reverse=True,
        )
        for cmd, stats in sorted_commands[:5]:
            lines.append(
                f"    {cmd:<23} {stats['raw_tokens']:>12,} {stats['tldr_tokens']:>12,} {stats['savings_percent']:>9.1f}%"
            )

        lines.append("")
        lines.append("  Top Projects:")

        # Sort projects by raw_tokens descending
        sorted_projects = sorted(
            by_project.items(),
            key=lambda x: x[1]["raw_tokens"],
            reverse=True,
        )
        for proj, stats in sorted_projects[:5]:
            # Use basename for display
            proj_name = Path(proj).name if proj else "unknown"
            lines.append(
                f"    {proj_name:<23} {stats['raw_tokens']:>12,} {stats['tldr_tokens']:>12,} {stats['savings_percent']:>9.1f}%"
            )

        # Cost estimates
        saved_tokens = cumulative["total_raw_tokens"] - cumulative["total_tldr_tokens"]
        sonnet_saved = saved_tokens * 3 / 1_000_000  # $3/M input tokens
        opus_saved = saved_tokens * 15 / 1_000_000  # $15/M input tokens

        lines.extend([
            "",
            "  Estimated Savings:",
            "  " + "-" * 50,
            f"  Claude Sonnet ($3/M):   ${sonnet_saved:.2f} saved",
            f"  Claude Opus ($15/M):    ${opus_saved:.2f} saved",
            "=" * 70,
            "",
        ])

        return "\n".join(lines)


# Singleton instance for convenient access
_default_tracker: UsageTracker | None = None


def get_default_tracker() -> UsageTracker:
    """Get the default usage tracker (singleton).

    Returns:
        UsageTracker instance using ~/.tldr/stats/
    """
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = UsageTracker()
    return _default_tracker


def record_usage(
    command: str,
    project: str,
    raw_tokens: int,
    tldr_tokens: int,
) -> None:
    """Convenience function to record usage with default tracker.

    Args:
        command: The tldr command (tree, structure, context, etc.)
        project: The project path/directory
        raw_tokens: Estimated tokens if reading raw files
        tldr_tokens: Actual tokens returned by tldr
    """
    tracker = get_default_tracker()
    tracker.record_usage(command, project, raw_tokens, tldr_tokens)
