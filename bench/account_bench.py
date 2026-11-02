"""Time AccountStore.verify: sequential and through a thread pool.

Run: venv\\Scripts\\python.exe bench/account_bench.py --n 200 --workers 4
Prints: ACCOUNT_BENCH verify_mean_ms=<x> p95_ms=<y> pool_wall_ms=<z> n=200 workers=4
"""
import argparse
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.config import TICK_MS  # noqa: E402
from net.accounts import AccountStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    pw = "correct horse battery"
    with tempfile.TemporaryDirectory() as d:
        store = AccountStore(Path(d))
        err = store.create("benchuser", pw)
        if err is not None:
            print(f"create failed: {err}")
            return 1
        store.verify("benchuser", pw)  # warm up
        times = []
        for _ in range(args.n):
            t0 = time.perf_counter()
            ok = store.verify("benchuser", pw)
            times.append((time.perf_counter() - t0) * 1000.0)
            if not ok:
                print("verify failed")
                return 1
        t0 = time.perf_counter()
        with ThreadPoolExecutor(args.workers) as ex:
            oks = list(ex.map(lambda _: store.verify("benchuser", pw), range(args.n)))
        pool_wall = (time.perf_counter() - t0) * 1000.0
        if not all(oks):
            print("pool verify failed")
            return 1
    times.sort()
    mean = statistics.fmean(times)
    p50 = times[len(times) // 2]
    p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
    print(f"ACCOUNT_BENCH verify_mean_ms={mean:.1f} p95_ms={p95:.1f} pool_wall_ms={pool_wall:.0f} "
          f"n={args.n} workers={args.workers}")
    print(f"p50_ms={p50:.1f} pool_per_verify_ms={pool_wall / args.n:.1f}")
    limit = 0.1 * TICK_MS
    if mean > limit:
        print(f"run verify in the thread pool: {mean:.1f} ms > 10% of TICK_MS ({limit:.1f} ms)")
    else:
        print(f"verify may run inline: {mean:.1f} ms <= 10% of TICK_MS ({limit:.1f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
