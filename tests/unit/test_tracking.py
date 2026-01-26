"""Tests for cumulative token usage tracking (TDD - write tests first).

These tests define the expected behavior for:
1. Recording individual command usage
2. JSONL persistence to ~/.tldr/stats/usage.jsonl
3. Cumulative stats aggregation
4. Stats by project and command
"""

import json
import os
import tempfile
from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import patch

import pytest


class TestRecordUsage:
    """Tests for recording individual tldr command usage."""

    def test_record_usage_creates_file(self, tmp_path):
        """Should create usage.jsonl if it doesn't exist."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker(stats_dir=tmp_path)
        tracker.record_usage(
            command="structure",
            project="/home/user/myproject",
            raw_tokens=10000,
            tldr_tokens=1000,
        )

        usage_file = tmp_path / "usage.jsonl"
        assert usage_file.exists()

    def test_record_usage_appends_jsonl(self, tmp_path):
        """Should append each usage record as a JSON line."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker(stats_dir=tmp_path)

        # Record two usages
        tracker.record_usage("structure", "/proj1", 10000, 1000)
        tracker.record_usage("context", "/proj2", 20000, 2000)

        usage_file = tmp_path / "usage.jsonl"
        lines = usage_file.read_text().strip().split("\n")
        assert len(lines) == 2

        record1 = json.loads(lines[0])
        record2 = json.loads(lines[1])

        assert record1["command"] == "structure"
        assert record1["project"] == "/proj1"
        assert record1["raw_tokens"] == 10000
        assert record1["tldr_tokens"] == 1000

        assert record2["command"] == "context"
        assert record2["project"] == "/proj2"

    def test_record_usage_includes_timestamp(self, tmp_path):
        """Should include ISO timestamp in each record."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker(stats_dir=tmp_path)
        tracker.record_usage("tree", "/proj", 5000, 500)

        usage_file = tmp_path / "usage.jsonl"
        record = json.loads(usage_file.read_text().strip())

        assert "timestamp" in record
        # Should be ISO format with timezone
        ts = datetime.fromisoformat(record["timestamp"])
        assert ts.tzinfo is not None

    def test_record_usage_calculates_savings(self, tmp_path):
        """Should calculate and store savings percentage."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker(stats_dir=tmp_path)
        tracker.record_usage("structure", "/proj", 10000, 1000)  # 90% savings

        usage_file = tmp_path / "usage.jsonl"
        record = json.loads(usage_file.read_text().strip())

        assert record["savings_percent"] == 90.0

    def test_record_usage_handles_zero_raw_tokens(self, tmp_path):
        """Should handle edge case of zero raw tokens."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker(stats_dir=tmp_path)
        tracker.record_usage("tree", "/proj", 0, 0)

        usage_file = tmp_path / "usage.jsonl"
        record = json.loads(usage_file.read_text().strip())

        assert record["savings_percent"] == 0.0


class TestGetCumulativeStats:
    """Tests for aggregating cumulative statistics."""

    def test_get_cumulative_stats_empty(self, tmp_path):
        """Should return zeros when no usage recorded."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker(stats_dir=tmp_path)
        stats = tracker.get_cumulative_stats()

        assert stats["total_queries"] == 0
        assert stats["total_raw_tokens"] == 0
        assert stats["total_tldr_tokens"] == 0
        assert stats["total_savings_percent"] == 0.0

    def test_get_cumulative_stats_sums_totals(self, tmp_path):
        """Should sum all usage records."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker(stats_dir=tmp_path)
        tracker.record_usage("structure", "/proj1", 10000, 1000)
        tracker.record_usage("context", "/proj2", 20000, 2000)
        tracker.record_usage("calls", "/proj1", 30000, 3000)

        stats = tracker.get_cumulative_stats()

        assert stats["total_queries"] == 3
        assert stats["total_raw_tokens"] == 60000
        assert stats["total_tldr_tokens"] == 6000
        assert stats["total_savings_percent"] == 90.0  # (60000-6000)/60000 * 100

    def test_get_cumulative_stats_includes_period(self, tmp_path):
        """Should include date range of tracked usage."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker(stats_dir=tmp_path)
        tracker.record_usage("structure", "/proj", 10000, 1000)

        stats = tracker.get_cumulative_stats()

        assert "period_start" in stats
        assert "period_end" in stats


class TestGetStatsByProject:
    """Tests for grouping stats by project."""

    def test_get_stats_by_project_groups_correctly(self, tmp_path):
        """Should group usage by project path."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker(stats_dir=tmp_path)
        tracker.record_usage("structure", "/proj1", 10000, 1000)
        tracker.record_usage("context", "/proj1", 20000, 2000)
        tracker.record_usage("structure", "/proj2", 30000, 3000)

        by_project = tracker.get_stats_by_project()

        assert "/proj1" in by_project
        assert "/proj2" in by_project

        assert by_project["/proj1"]["queries"] == 2
        assert by_project["/proj1"]["raw_tokens"] == 30000
        assert by_project["/proj1"]["tldr_tokens"] == 3000

        assert by_project["/proj2"]["queries"] == 1
        assert by_project["/proj2"]["raw_tokens"] == 30000

    def test_get_stats_by_project_uses_basename(self, tmp_path):
        """Should show project basename in display name."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker(stats_dir=tmp_path)
        tracker.record_usage("structure", "/home/user/llm-tldr", 10000, 1000)

        by_project = tracker.get_stats_by_project()

        # Full path should be key
        assert "/home/user/llm-tldr" in by_project


class TestGetStatsByCommand:
    """Tests for grouping stats by command type."""

    def test_get_stats_by_command_groups_correctly(self, tmp_path):
        """Should group usage by command name."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker(stats_dir=tmp_path)
        tracker.record_usage("structure", "/proj1", 10000, 1000)
        tracker.record_usage("structure", "/proj2", 20000, 2000)
        tracker.record_usage("context", "/proj1", 30000, 3000)

        by_command = tracker.get_stats_by_command()

        assert "structure" in by_command
        assert "context" in by_command

        assert by_command["structure"]["queries"] == 2
        assert by_command["structure"]["raw_tokens"] == 30000
        assert by_command["structure"]["tldr_tokens"] == 3000

        assert by_command["context"]["queries"] == 1
        assert by_command["context"]["raw_tokens"] == 30000


class TestDefaultLocation:
    """Tests for default stats file location."""

    def test_default_stats_dir(self):
        """Should use ~/.tldr/stats/ by default."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker()
        expected = Path.home() / ".tldr" / "stats"
        assert tracker.stats_dir == expected

    def test_default_usage_file_path(self):
        """Should use usage.jsonl filename."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker()
        expected = Path.home() / ".tldr" / "stats" / "usage.jsonl"
        assert tracker.usage_file == expected


class TestFormatCumulativeReport:
    """Tests for the cumulative stats report formatting."""

    def test_format_report_includes_all_sections(self, tmp_path):
        """Should include totals, by-command, by-project, and cost estimates."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker(stats_dir=tmp_path)
        tracker.record_usage("structure", "/home/user/llm-tldr", 100000, 10000)
        tracker.record_usage("context", "/home/user/opc", 200000, 10000)

        report = tracker.format_cumulative_report()

        # Check key sections exist
        assert "TLDR Cumulative Token Savings" in report
        assert "Total Queries:" in report
        assert "By Command:" in report
        assert "Top Projects:" in report
        assert "Estimated Savings:" in report
        assert "Claude Sonnet" in report
        assert "Claude Opus" in report

    def test_format_report_shows_savings(self, tmp_path):
        """Should show savings percentage correctly."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker(stats_dir=tmp_path)
        tracker.record_usage("structure", "/proj", 10000, 1000)  # 90% savings

        report = tracker.format_cumulative_report()

        assert "90.0%" in report

    def test_format_report_empty_stats(self, tmp_path):
        """Should handle empty stats gracefully."""
        from tldr.tracking import UsageTracker

        tracker = UsageTracker(stats_dir=tmp_path)
        report = tracker.format_cumulative_report()

        assert "No usage data recorded yet" in report or "Total Queries: 0" in report
