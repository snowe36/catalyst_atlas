#!/usr/bin/env python3
"""Watch ColabFold AF funnel on RunPod: pull+score on success, always request pod stop.

Terminal outcomes write ``out/design_watchdog_action.json`` with action ``stop``.
Does not run MD. Scope ends at AF import + mechanistic ranking.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
LOG = OUT / "colabfold_monitor.log"
ACTION = OUT / "design_watchdog_action.json"
STATUS = OUT / "design_run_status.json"

DEFAULT_POD = "ggn7kl3zubwjfj"
DEFAULT_HOST = "194.14.47.19"
DEFAULT_PORT = "23358"
DEFAULT_TARGET_PDB = 88
# No progress for this long → failure (MSA can be slow; WT×8 + designs).
STALL_SEC = 45 * 60
SSH_FAIL_LIMIT = 8
POLL_SEC = 90


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{_utc()} {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def _ssh(host: str, port: str, key: str, remote: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh",
        "-i",
        key,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "BatchMode=yes",
        "-p",
        port,
        f"root@{host}",
        remote,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def _probe(host: str, port: str, key: str) -> dict:
    remote = r"""python3 - <<'PY'
import json
from pathlib import Path
out = Path("/workspace/catalyst_atlas/data/processed/design/colabfold_out")
pred = Path("/workspace/catalyst_atlas/data/processed/design/predictions")
log = Path("/workspace/colabfold_queue.log")
text = log.read_text(errors="replace") if log.exists() else ""
n_pdb = len(list(out.rglob("*.pdb"))) if out.exists() else 0
n_metrics = len(list(pred.rglob("metrics.json"))) if pred.exists() else 0
qs = [l for l in text.splitlines() if "Query " in l]
takes = [l for l in text.splitlines() if "took " in l]
traceback = "Traceback (most recent call last)" in text
wrote = "wrote metrics" in text
# live workers (exact cmdline; ignore this probe)
import os
alive = False
for p in Path("/proc").iterdir():
    if not p.name.isdigit():
        continue
    try:
        cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except Exception:
        continue
    if "colabfold_batch" in cmd or (
        "run_colabfold_queue.py" in cmd and "python" in cmd and "bash" not in cmd.split()[0]
    ):
        alive = True
        break
print(json.dumps({
    "n_pdb": n_pdb,
    "n_metrics": n_metrics,
    "n_query_lines": len(qs),
    "n_took": len(takes),
    "last_query": qs[-1] if qs else "",
    "wrote_metrics": wrote,
    "traceback": traceback,
    "alive": alive,
    "log_bytes": len(text),
}))
PY"""
    proc = _ssh(host, port, key, remote)
    if proc.returncode != 0:
        return {"ok": False, "stderr": (proc.stderr or "")[-400:]}
    try:
        data = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"ok": False, "stderr": (proc.stdout or "")[-400:]}
    data["ok"] = True
    return data


def _request_stop(pod_id: str, reason: str, *, success: bool, detail: dict) -> None:
    payload = {
        "action": "stop",
        "pod_id": pod_id,
        "reason": reason,
        "success": success,
        "at": _utc(),
        "detail": detail,
    }
    ACTION.write_text(json.dumps(payload, indent=2))
    STATUS.write_text(
        json.dumps(
            {
                "status": "watchdog_stop_requested" if not success else "af_complete_stop_requested",
                "pod_id": pod_id,
                "reason": reason,
                "success": success,
                "at": _utc(),
                "detail": detail,
            },
            indent=2,
        )
    )
    # Machine-readable line for agent notify_on_output → MCP stop-pod
    print(f"STOP_POD {pod_id} reason={reason}", flush=True)
    _log(f"STOP_POD requested reason={reason}")


def _rsync(host: str, port: str, key: str) -> None:
    dest = ROOT / "data" / "processed" / "design"
    dest.mkdir(parents=True, exist_ok=True)
    ssh_e = f"ssh -i {key} -o StrictHostKeyChecking=no -p {port}"
    for sub in ("colabfold_out", "predictions", "colabfold_a3m"):
        src = f"root@{host}:/workspace/catalyst_atlas/data/processed/design/{sub}/"
        local = dest / sub
        local.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-az", "--timeout=120", "-e", ssh_e, src, str(local) + "/"],
            check=False,
        )


def _score_and_report() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from catalyst_atlas.design.report import write_design_case_study
    from catalyst_atlas.design.score import run_score

    df = run_score(af_queue_only=True, mock_predictions=False)
    panel_path = ROOT / "data" / "processed" / "design" / "panel.json"
    panel = json.loads(panel_path.read_text())
    panel = [p for p in panel if p["enzyme_id"] in set(df.enzyme_id)]
    report = write_design_case_study(df, panel=panel)
    meta = json.loads((ROOT / "data" / "processed" / "design" / "score_meta.json").read_text())
    return {"n_scores": int(len(df)), "report": str(report), "score_meta": meta}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pod-id", default=os.environ.get("RUNPOD_POD_ID", DEFAULT_POD))
    ap.add_argument("--host", default=os.environ.get("RUNPOD_HOST", DEFAULT_HOST))
    ap.add_argument("--port", default=os.environ.get("RUNPOD_PORT", DEFAULT_PORT))
    ap.add_argument("--key", default=os.path.expanduser("~/.ssh/runpod_ed25519"))
    ap.add_argument("--target-pdb", type=int, default=DEFAULT_TARGET_PDB)
    ap.add_argument("--poll-sec", type=int, default=POLL_SEC)
    ap.add_argument("--stall-sec", type=int, default=STALL_SEC)
    ap.add_argument("--max-hours", type=float, default=10.0)
    args = ap.parse_args()

    if not Path(args.key).exists():
        _log(f"missing ssh key {args.key}")
        return 2

    _log(
        f"watchdog start pod={args.pod_id} {args.host}:{args.port} "
        f"target_pdb={args.target_pdb} (no MD)"
    )

    best_pdb = -1
    best_took = -1
    last_progress = time.time()
    ssh_fails = 0
    deadline = time.time() + args.max_hours * 3600

    while time.time() < deadline:
        probe = _probe(args.host, args.port, args.key)
        if not probe.get("ok"):
            ssh_fails += 1
            _log(f"ssh_fail {ssh_fails}/{SSH_FAIL_LIMIT} {probe.get('stderr', '')[:120]}")
            if ssh_fails >= SSH_FAIL_LIMIT:
                _request_stop(
                    args.pod_id,
                    "ssh_unreachable",
                    success=False,
                    detail={"ssh_fails": ssh_fails},
                )
                return 1
            time.sleep(args.poll_sec)
            continue
        ssh_fails = 0

        n_pdb = int(probe["n_pdb"])
        n_took = int(probe["n_took"])
        alive = bool(probe["alive"])
        _log(
            f"probe n_pdb={n_pdb} n_took={n_took} alive={alive} "
            f"wrote={probe['wrote_metrics']} q={probe.get('last_query', '')[-80:]}"
        )

        if n_pdb > best_pdb or n_took > best_took:
            best_pdb = max(best_pdb, n_pdb)
            best_took = max(best_took, n_took)
            last_progress = time.time()

        success = bool(probe["wrote_metrics"]) or n_pdb >= args.target_pdb
        if success:
            _log(f"success n_pdb={n_pdb} — rsync + score (no MD)")
            _rsync(args.host, args.port, args.key)
            try:
                summary = _score_and_report()
            except Exception as exc:  # noqa: BLE001 — surface then still stop pod
                _log(f"score_failed {exc}")
                _request_stop(
                    args.pod_id,
                    "score_failed_after_af",
                    success=False,
                    detail={"error": str(exc), "n_pdb": n_pdb},
                )
                return 1
            _request_stop(
                args.pod_id,
                "af_complete",
                success=True,
                detail={"n_pdb": n_pdb, **summary},
            )
            return 0

        stalled = (time.time() - last_progress) > args.stall_sec
        if probe.get("traceback") and (not alive or stalled):
            _request_stop(
                args.pod_id,
                "traceback_dead",
                success=False,
                detail=probe,
            )
            return 1

        if stalled and not alive:
            _request_stop(
                args.pod_id,
                "stalled_dead",
                success=False,
                detail={"best_pdb": best_pdb, "best_took": best_took, **probe},
            )
            return 1
        if stalled and alive:
            # Alive but no new PDBs/folds for stall_sec (MSA hang or zombie "alive")
            _request_stop(
                args.pod_id,
                "stalled_no_progress",
                success=False,
                detail={"stall_sec": args.stall_sec, "best_pdb": best_pdb, **probe},
            )
            return 1

        time.sleep(args.poll_sec)

    _request_stop(
        args.pod_id,
        "timeout",
        success=False,
        detail={"best_pdb": best_pdb, "max_hours": args.max_hours},
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
