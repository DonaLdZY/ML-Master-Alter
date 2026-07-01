from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKDIR = str(ROOT_DIR)


def now_ts() -> float:
    return time.time()


def _is_interrupted_exit_code(exit_code: int | None) -> bool:
    # Windows CTRL_C_EVENT/CTRL_BREAK_EVENT is reported as 0xC000013A.
    return exit_code in {3221225786, -1073741510, 130, -2, -15}


class StartAutoMLRequest(BaseModel):
    task_id: str
    python_executable: str = "python"
    working_dir: str = DEFAULT_WORKDIR
    env_overrides: dict[str, str] = Field(default_factory=dict)
    args: list[str] = Field(default_factory=list)
    log_dir: str
    workspace_dir: str
    graceful_shutdown_buffer_secs: int = Field(default=600, ge=0, le=3600)


class StopRequest(BaseModel):
    job_id: str


class SnapshotRequest(BaseModel):
    log_dir: str = ""
    run_dir: str = ""
    task_name: str = ""


class JobStatus(BaseModel):
    job_id: str
    task_id: str
    status: str
    started_at: float
    updated_at: float
    log_dir: str
    workspace_dir: str
    exit_code: int | None = None
    last_error: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class JobRuntime:
    job_id: str
    task_id: str
    log_dir: str
    workspace_dir: str
    process: subprocess.Popen[str] | None
    status: str
    started_at: float
    updated_at: float
    exit_code: int | None = None
    last_error: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    stop_requested: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRuntime] = {}

    def create(self, task_id: str, log_dir: str, workspace_dir: str) -> JobRuntime:
        with self._lock:
            for j in self._jobs.values():
                if j.task_id == task_id and j.status == "running":
                    proc = j.process
                    if proc is not None and proc.poll() is not None:
                        j.status = "failed" if (proc.returncode or 0) != 0 else "completed"
                        j.exit_code = proc.returncode
                        j.updated_at = now_ts()
                        continue
                    raise HTTPException(status_code=400, detail="task already running in AutoML service")
            job_id = uuid.uuid4().hex
            ts = now_ts()
            job = JobRuntime(
                job_id=job_id,
                task_id=task_id,
                log_dir=log_dir,
                workspace_dir=workspace_dir,
                process=None,
                status="pending",
                started_at=ts,
                updated_at=ts,
            )
            self._jobs[job_id] = job
            return job

    def _get_unlocked(self, job_id: str) -> JobRuntime:
        job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    def get(self, job_id: str) -> JobRuntime:
        with self._lock:
            return self._get_unlocked(job_id)

    def set_process(self, job_id: str, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            job = self._get_unlocked(job_id)
            job.process = proc
            job.status = "running"
            job.updated_at = now_ts()

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._get_unlocked(job_id)
            for k, v in kwargs.items():
                setattr(job, k, v)
            job.updated_at = now_ts()

    def status(self, job_id: str) -> JobStatus:
        job = self.get(job_id)
        return JobStatus(
            job_id=job.job_id,
            task_id=job.task_id,
            status=job.status,
            started_at=job.started_at,
            updated_at=job.updated_at,
            log_dir=job.log_dir,
            workspace_dir=job.workspace_dir,
            exit_code=job.exit_code,
            last_error=job.last_error,
            stdout_tail=job.stdout_tail[-60000:],
            stderr_tail=job.stderr_tail[-60000:],
        )


store = JobStore()
app = FastAPI(title="ML-Master Service API", version="0.1.0")


def _tail_text(text: str, limit: int = 200000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _extract_time_limit_secs(args: list[str]) -> int | None:
    for item in args:
        if not isinstance(item, str):
            continue
        key = "agent.time_limit="
        if item.startswith(key):
            raw = item[len(key) :].strip().strip('"').strip("'")
            try:
                val = int(float(raw))
                return max(1, val)
            except Exception:
                return None
    return None


def _safe_read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _parse_jsonl(path: Path, limit: int = 500) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return []
    return rows


def _pick_log_dir(req: SnapshotRequest) -> Path | None:
    raw = req.log_dir.strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        if p.exists():
            return p
    if req.run_dir.strip() and req.task_name.strip():
        candidate = Path(req.run_dir).expanduser().resolve() / "automl" / "logs" / req.task_name.strip()
        if candidate.exists():
            return candidate
    return None


def _build_snapshot(req: SnapshotRequest) -> dict[str, Any]:
    log_dir = _pick_log_dir(req)
    if log_dir is None:
        return {}

    out: dict[str, Any] = {"log_dir": str(log_dir)}
    out["events"] = _parse_jsonl(log_dir / "event_stream.jsonl", limit=400)
    journal = _safe_read_json(log_dir / "journal.json", {})
    nodes: list[dict[str, Any]] = []
    best_id = None
    if isinstance(journal, dict) and journal:
        j_nodes = journal.get("nodes", [])
        node2parent = journal.get("node2parent", {})
        best_metric = None
        best_maximize = True
        global_maximize: bool | None = None
        for n in j_nodes:
            m = (n.get("metric") or {}).get("maximize")
            if isinstance(m, bool):
                global_maximize = m
                break
        for n in j_nodes:
            metric_obj = n.get("metric") or {}
            metric_val = metric_obj.get("value")
            maximize_raw = metric_obj.get("maximize")
            maximize = maximize_raw if isinstance(maximize_raw, bool) else global_maximize
            if metric_val is not None:
                if best_metric is None:
                    best_metric = metric_val
                    best_id = n.get("id")
                    best_maximize = True if maximize is None else maximize
                else:
                    if best_maximize:
                        if metric_val > best_metric:
                            best_metric = metric_val
                            best_id = n.get("id")
                    else:
                        if metric_val < best_metric:
                            best_metric = metric_val
                            best_id = n.get("id")
            node_id = n.get("id")
            nodes.append(
                {
                    "id": node_id,
                    "parent_id": node2parent.get(node_id),
                    "stage": n.get("stage"),
                    "plan": n.get("plan"),
                    "code": n.get("code"),
                    "result": "".join(n.get("_term_out", [])) if isinstance(n.get("_term_out"), list) else str(n.get("_term_out")),
                    "insight": n.get("analysis"),
                    "metric": metric_val,
                    "maximize": maximize,
                    "is_buggy": n.get("is_buggy"),
                    "is_valid": n.get("is_valid"),
                    "visits": n.get("visits"),
                    "total_reward": n.get("total_reward"),
                    "uct": n.get("_uct"),
                    "finish_time": n.get("finish_time"),
                }
            )
    out["nodes"] = nodes
    out["best_node_id"] = best_id
    out["ml_log"] = (log_dir / "ml-master.log").read_text(encoding="utf-8", errors="ignore")[-60000:] if (log_dir / "ml-master.log").exists() else ""
    out["frontend_stdout"] = (log_dir / "_frontend_stdout.log").read_text(encoding="utf-8", errors="ignore")[-60000:] if (log_dir / "_frontend_stdout.log").exists() else ""
    out["frontend_stderr"] = (log_dir / "_frontend_stderr.log").read_text(encoding="utf-8", errors="ignore")[-60000:] if (log_dir / "_frontend_stderr.log").exists() else ""
    return out


def _run_job(job_id: str, req: StartAutoMLRequest) -> None:
    cmd = [req.python_executable or "python", "main_mcts.py", *req.args]
    env = os.environ.copy()
    env.update(req.env_overrides or {})
    workdir = req.working_dir.strip() or DEFAULT_WORKDIR

    try:
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
    except Exception as e:
        store.update(job_id, status="failed", last_error=f"start failed: {e}")
        return

    store.set_process(job_id, proc)
    time_limit_secs = _extract_time_limit_secs(req.args)
    timed_out = False
    if time_limit_secs is not None:
        total_timeout = time_limit_secs + int(req.graceful_shutdown_buffer_secs)
        try:
            out, err = proc.communicate(timeout=total_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                proc.terminate()
                out, err = proc.communicate(timeout=15)
            except Exception:
                proc.kill()
                out, err = proc.communicate()
    else:
        out, err = proc.communicate()
    exit_code = proc.returncode
    current_job = store.get(job_id)
    stop_requested = bool(current_job.stop_requested or current_job.status in {"stopping", "stopped"})
    if timed_out:
        status = "failed"
    elif stop_requested or _is_interrupted_exit_code(exit_code):
        status = "stopped"
    elif exit_code == 0:
        status = "completed"
    else:
        status = "failed"
    last_error = None
    if timed_out:
        last_error = (
            "AutoML exceeded service timeout and was terminated by service. "
            f"search_limit={time_limit_secs}s, "
            f"grace={int(req.graceful_shutdown_buffer_secs)}s."
        )
    elif status == "stopped":
        if stop_requested:
            last_error = "AutoML stopped by user."
        else:
            last_error = f"AutoML interrupted by console/control signal (exit code {exit_code})."
    elif exit_code != 0:
        tail = (err or out or "").strip()
        if tail:
            last_error = tail.splitlines()[-1][:300]
        else:
            last_error = f"AutoML exited with code {exit_code}"

    # Service-side capture for gateway debug surface.
    try:
        log_dir = Path(req.log_dir).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        if out:
            (log_dir / "_service_stdout.log").write_text(_tail_text(out), encoding="utf-8", errors="ignore")
        if err:
            (log_dir / "_service_stderr.log").write_text(_tail_text(err), encoding="utf-8", errors="ignore")
    except Exception:
        pass

    store.update(
        job_id,
        status=status,
        exit_code=exit_code,
        last_error=last_error,
        stdout_tail=_tail_text(out or ""),
        stderr_tail=_tail_text(err or ""),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs/start")
def start_job(req: StartAutoMLRequest) -> dict[str, Any]:
    job = store.create(task_id=req.task_id, log_dir=req.log_dir, workspace_dir=req.workspace_dir)
    thread = threading.Thread(target=_run_job, args=(job.job_id, req), daemon=True)
    thread.start()
    return {
        "job_id": job.job_id,
        "status": "started",
        "log_dir": req.log_dir,
        "workspace_dir": req.workspace_dir,
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return store.status(job_id).model_dump()


@app.post("/jobs/stop")
def stop_job(req: StopRequest) -> dict[str, Any]:
    job = store.get(req.job_id)
    proc = job.process
    if proc is None or proc.poll() is not None:
        return {"status": "not_running", "job_id": req.job_id}
    store.update(req.job_id, status="stopping", stop_requested=True, last_error="stop requested by user")
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[arg-type]
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            proc.terminate()
    except Exception:
        try:
            proc.kill()
        except Exception:
            proc.terminate()
    store.update(req.job_id, status="stopped", last_error="stopped by user")
    return {"status": "stopping", "job_id": req.job_id}


@app.post("/snapshot")
def snapshot(req: SnapshotRequest) -> dict[str, Any]:
    try:
        return _build_snapshot(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"snapshot failed: {e}")
