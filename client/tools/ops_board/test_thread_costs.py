#!/usr/bin/env python3
"""Tests for tools/thread_costs.py.

Run: python3 tools/ops_board/test_thread_costs.py   (prints failures, then "THREAD_COSTS_TESTS passed=N failed=M")
"""
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thread_costs  # noqa: E402

WORKER = {"ccr": {
    "id": "session_A", "title": "Arena decor", "created_at": "2026-10-22T19:22:06.849464Z",
    "updated_at": "2026-10-22T20:10:38.683931Z", "status_bucket": "SESSION_STATUS_BUCKET_COMPLETED",
    "parent_session_id": "session_LEAD", "session_context": {"model": "claude-opus-5-5[1m]", "effort_level": "medium"},
    "external_metadata": {"last_served_model": "claude-opus-5-5", "context_usage": {"used_tokens": 277397},
                          "usage": {"cache_read_tokens": 20230771, "cache_write_tokens": 241408,
                                    "cost_usd": 8.88039844, "input_tokens": 199, "output_tokens": 83476}}}}
LEAD = {"id": "session_LEAD", "title": "Coordinator", "created_at": "2026-10-22T16:55:40Z",
        "updated_at": "2026-10-22T21:41:37Z", "status_bucket": "SESSION_STATUS_BUCKET_WORKING",
        "session_context": {"model": "claude-opus-5-5[1m]"}, "external_metadata": {}}


class ThreadCostsTest(unittest.TestCase):
    def test_rows(self) -> None:
        text = json.dumps(WORKER) + "\n\n" + json.dumps(LEAD) + "\n"
        a, b = [thread_costs.row(r) for r in thread_costs.parse_records(text)]
        self.assertEqual((a["role"], a["status"], a["model"], a["minutes"]), ("worker", "done", "claude-opus-5-5", 48.5))
        self.assertEqual((a["cost_usd"], a["cost_per_hour"]), (8.88, 10.99))
        self.assertEqual(a["cache_read_tokens"], 20230771)
        # a record with no usage yet is a zero-cost row, not an error
        self.assertEqual((b["role"], b["status"], b["model"], b["cost_usd"]), ("lead", "running", "claude-opus-5-5", 0.0))

    def test_main_writes_sorted_rows_and_total(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "s.jsonl")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(LEAD) + "\n" + json.dumps(WORKER) + "\n")
            sys.argv = ["thread_costs.py", src, "--out", tmp]
            out = io.StringIO()
            old, sys.stdout = sys.stdout, out
            try:
                self.assertEqual(thread_costs.main(), 0)
            finally:
                sys.stdout = old
            self.assertTrue(out.getvalue().startswith("THREAD_COSTS threads=2 cost_usd=8.88"))
            rows = json.load(open(os.path.join(tmp, "threads.json"), encoding="utf-8"))
            total = json.load(open(os.path.join(tmp, "threads_total.json"), encoding="utf-8"))[0]
            self.assertEqual([r["id"] for r in rows], ["session_A", "session_LEAD"])
            self.assertEqual((total["threads"], total["running"]), (2, 1))
            self.assertGreater(total["cache_read_share"], 0.98)


if __name__ == "__main__":
    out = io.StringIO()
    result = unittest.TextTestRunner(stream=out, verbosity=0).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ThreadCostsTest))
    failed = len(result.failures) + len(result.errors)
    if failed:
        print(out.getvalue())
    print("THREAD_COSTS_TESTS passed=%d failed=%d" % (result.testsRun - failed, failed))
    sys.exit(1 if failed else 0)
