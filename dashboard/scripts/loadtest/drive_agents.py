"""Simulate M concurrent protocol agents driving full backtests.

Each agent: POST /api/v1/runs (3 trading days ~= 21 hourly steps), then loop
GET steps/next -> POST decision until completed. Records per-endpoint latency,
end-to-end run wall time, errors, steps lost to the decision deadline, and the
server-reported timeout_holds counter (present once T3 ships; 0 before).

Usage:
    python dashboard/scripts/loadtest/drive_agents.py 100 --artifacts /tmp/atl_loadtest_xxx
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("agents", type=int, nargs="?", default=10)
parser.add_argument("--artifacts", required=True,
                    help="artifacts dir printed by stress_serve.py")
parser.add_argument("--base", default="http://127.0.0.1:8402")
parser.add_argument("--allow-remote", action="store_true",
                    help="required to target anything but localhost")
args = parser.parse_args()

host = urllib.parse.urlparse(args.base).hostname or ""
if host not in ("127.0.0.1", "localhost", "::1") and not args.allow_remote:
    sys.exit(f"refusing non-localhost target {args.base!r} (pass --allow-remote)")

BASE = args.base
M = args.agents
AGENTS = json.load(open(os.path.join(args.artifacts, "agents.json")))
PID_FILE = os.path.join(args.artifacts, "server.pid")

samples = []          # (kind, ms, http_status)
lock = threading.Lock()
deadline_losses = []  # steps auto-held because our decision arrived too late
server_holds = []     # server-reported timeout_holds per completed run (T3+)
run_walls = []        # end-to-end seconds per completed run
failures = []


def req(method, path, key, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"X-API-Key": key, "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=180) as resp:
            payload = json.loads(resp.read())
            return resp.status, payload, (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read())
        except Exception:
            payload = {}
        return e.code, payload, (time.perf_counter() - t0) * 1000
    except Exception as e:
        return -1, {"error": str(e)}, (time.perf_counter() - t0) * 1000


def record(kind, ms, status):
    with lock:
        samples.append((kind, ms, status))


def drive(agent):
    key = agent["api_key"]
    t_run = time.perf_counter()
    status, body, ms = req("POST", "/api/v1/runs", key, {
        "config": {"start_date": "2026-06-01", "end_date": "2026-06-03"},
    })
    record("create_run", ms, status)
    if status != 200:
        with lock:
            failures.append(("create", status, str(body)[:150]))
        return
    run_id = body["run_id"]
    lost = 0
    first = True
    for _ in range(400):  # hard cap: a run is ~21 steps
        status, step, ms = req("GET", f"/api/v1/runs/{run_id}/steps/next", key)
        record("steps_next", ms, status)
        if status != 200:
            with lock:
                failures.append(("steps_next", status, str(step)[:150]))
            return
        st = step.get("status")
        if st == "loading":
            time.sleep(0.25)
            continue
        if st == "completed":
            break
        if st != "awaiting_decision":
            with lock:
                failures.append(("unexpected_step", 200, st or "?"))
            return
        orders = []
        if first:
            orders = [{"symbol": "AAPL", "side": "buy", "quantity": 5}]
            first = False
        status, res, ms = req(
            "POST", f"/api/v1/runs/{run_id}/steps/{step['step_id']}/decision", key,
            {"idempotency_key": uuid.uuid4().hex, "orders": orders,
             "rationale": "load test decision"},
        )
        record("decision", ms, status)
        if status == 409:
            if "deadline" in str(res) or "finalized" in str(res):
                lost += 1
                continue  # step was auto-held under us; keep going
            with lock:
                failures.append(("decision409", 409, str(res)[:150]))
            return
        if status != 200:
            with lock:
                failures.append(("decision", status, str(res)[:150]))
            return
        if res.get("run_status") == "completed":
            break
    status, view, _ = req("GET", f"/api/v1/runs/{run_id}", key)
    holds = 0
    if status == 200:
        holds = (view.get("engine_status") or {}).get("timeout_holds") or 0
    with lock:
        run_walls.append(time.perf_counter() - t_run)
        deadline_losses.append(lost)
        server_holds.append(holds)


def server_stats():
    try:
        pid = int(open(PID_FILE).read())
        st = open(f"/proc/{pid}/status").read()
        rss = next(l for l in st.splitlines() if l.startswith("VmRSS")).split()[1]
        thr = next(l for l in st.splitlines() if l.startswith("Threads")).split()[1]
        return int(rss) // 1024, int(thr)
    except Exception:
        return -1, -1


def dist(label, vals):
    if not vals:
        print(f"  {label:14s} (none)")
        return
    s = sorted(vals)
    print(f"  {label:14s} n={len(s):5d}  med={statistics.median(s):8.1f}  "
          f"p95={s[max(0, int(len(s)*0.95)-1)]:8.1f}  max={s[-1]:8.1f}")


rss0, thr0 = server_stats()
print(f"\n===== {M} concurrent agents =====  (server before: {rss0} MB RSS, {thr0} threads)")
t0 = time.perf_counter()
threads = [threading.Thread(target=drive, args=(a,)) for a in AGENTS[:M]]
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.perf_counter() - t0
rss1, thr1 = server_stats()

by_kind = {}
for kind, ms, status in samples:
    by_kind.setdefault(kind, []).append(ms)
print(f"total wall: {wall:.1f}s  |  requests: {len(samples)}  |  "
      f"throughput: {len(samples)/wall:.1f} req/s")
print(f"server after: {rss1} MB RSS ({rss1-rss0:+d}), {thr1} threads ({thr1-thr0:+d})")
print("per-request latency (ms):")
for kind in ("create_run", "steps_next", "decision"):
    dist(kind, by_kind.get(kind, []))
print("end-to-end run wall time (s):")
dist("full_run", run_walls)
total_lost = sum(deadline_losses)
runs_hit = sum(1 for x in deadline_losses if x)
print(f"completed runs: {len(run_walls)}/{M}  |  client-observed deadline losses: "
      f"{total_lost} (across {runs_hit} runs)  |  server timeout_holds: {sum(server_holds)}")
if failures:
    print(f"FAILURES: {len(failures)}")
    for f in failures[:8]:
        print("  ", f)
