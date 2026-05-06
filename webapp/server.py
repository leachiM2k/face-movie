"""FastAPI server for face-movie.

Single-tenant by design: in-memory job state, no auth, jobs survive only
as long as the process. Wraps run_pipeline() from main.py so the CLI and
the web UI share one code path. Intended to be run locally inside the
Docker container; do not expose publicly without adding auth + a queue.

Routes:
    GET  /                      static SPA (index.html)
    POST /api/render            multipart upload + params, starts a job
    GET  /api/events/{job_id}   server-sent events stream of progress
    GET  /api/download/{job_id} the rendered MP4
"""

from __future__ import annotations

import asyncio
import json
import secrets
import shutil
import sys
import tempfile
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile

# Allow running `uvicorn webapp.server:app` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import run_pipeline  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
# Sanity cap. Picked above any realistic "selfie a day for 25 years" upload.
# python-multipart's default of 1000 is too low; we override it explicitly
# when parsing the form below.
MAX_FILES = 25_000

app = FastAPI(title="Face-Movie")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@dataclass
class Job:
    id: str
    work_dir: Path
    input_dir: Path
    output_path: Path
    stage: str = "queued"
    current: int = 0
    total: int = 0
    error: str | None = None
    encoder: str | None = None
    skipped: int = 0
    used: int = 0
    duration_s: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


JOBS: dict[str, Job] = {}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/api/render")
async def start_render(request: Request):
    # We parse the form ourselves so we can raise the multipart parser's
    # max_files / max_fields limits beyond python-multipart's 1000 default.
    # FastAPI's File()/Form() dependencies don't expose those knobs.
    try:
        form = await request.form(
            max_files=MAX_FILES + 1,
            max_fields=MAX_FILES + 32,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not parse upload: {e}") from e

    files = [f for f in form.getlist("files") if isinstance(f, UploadFile)]
    if len(files) < 2:
        raise HTTPException(400, "need at least 2 images")
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"too many files (>{MAX_FILES})")

    try:
        scale = float(form.get("scale", "1.0"))
        frames_per_pair = int(form.get("frames_per_pair", "6"))
        fps = int(form.get("fps", "30"))
        overlay = str(form.get("overlay", "true")).lower() in ("true", "1", "yes", "on")
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"invalid parameter: {e}") from e

    if not (0.1 <= scale <= 1.0):
        raise HTTPException(400, "scale must be between 0.1 and 1.0")
    if not (1 <= frames_per_pair <= 60):
        raise HTTPException(400, "frames_per_pair must be between 1 and 60")
    if not (10 <= fps <= 60):
        raise HTTPException(400, "fps must be between 10 and 60")

    job_id = secrets.token_hex(8)
    work_dir = Path(tempfile.mkdtemp(prefix=f"fm-{job_id}-"))
    input_dir = work_dir / "input"
    input_dir.mkdir()
    output_path = work_dir / "out.mp4"

    saved = 0
    for upload in files:
        # webkitdirectory uploads carry the relative path in `filename`.
        # Flatten to basename and dedupe to avoid collisions.
        raw = upload.filename or "img.jpg"
        ext = Path(raw).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = Path(raw).stem or "img"
        target = input_dir / f"{stem}{ext}"
        i = 0
        while target.exists():
            i += 1
            target = input_dir / f"{stem}-{i}{ext}"
        target.write_bytes(await upload.read())
        saved += 1

    if saved < 2:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(400, f"need at least 2 image files; got {saved} valid")

    job = Job(id=job_id, work_dir=work_dir, input_dir=input_dir, output_path=output_path)
    JOBS[job_id] = job

    asyncio.get_running_loop().run_in_executor(
        None,
        _run_job_blocking,
        job, scale, frames_per_pair, fps, overlay,
    )
    return {"job_id": job_id, "files": saved}


def _run_job_blocking(
    job: Job, scale: float, frames_per_pair: int, fps: int, overlay: bool,
) -> None:
    """Runs run_pipeline() on a worker thread. Mutates job under its lock."""
    def progress(stage: str, current: int, total: int) -> None:
        with job.lock:
            job.stage = stage
            job.current = current
            job.total = total

    try:
        with job.lock:
            job.stage = "starting"
        result = run_pipeline(
            input_dir=job.input_dir,
            output_path=job.output_path,
            scale=scale,
            frames_per_pair=frames_per_pair,
            fps=fps,
            overlay=overlay,
            on_progress=progress,
        )
        with job.lock:
            job.stage = "done"
            job.encoder = result.encoder
            job.used = len(result.used_files)
            job.skipped = len(result.skipped_files)
            job.duration_s = result.duration_seconds
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        with job.lock:
            job.stage = "error"
            job.error = str(e) or e.__class__.__name__


@app.get("/api/events/{job_id}")
async def events(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "no such job")

    job = JOBS[job_id]

    async def stream():
        last: tuple | None = None
        while True:
            with job.lock:
                snap = (job.stage, job.current, job.total, job.error, job.encoder,
                        job.used, job.skipped, job.duration_s)
            if snap != last:
                payload = {
                    "stage": snap[0],
                    "current": snap[1],
                    "total": snap[2],
                    "error": snap[3],
                    "encoder": snap[4],
                    "used": snap[5],
                    "skipped": snap[6],
                    "duration_s": snap[7],
                }
                yield f"data: {json.dumps(payload)}\n\n"
                last = snap
            if snap[0] in ("done", "error"):
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)
    job = JOBS[job_id]
    if job.stage != "done":
        raise HTTPException(409, "not ready")
    return FileResponse(
        job.output_path, media_type="video/mp4", filename="face-movie.mp4",
    )


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Caller cleans up after themselves once they have the file."""
    job = JOBS.pop(job_id, None)
    if job is None:
        raise HTTPException(404)
    shutil.rmtree(job.work_dir, ignore_errors=True)
    return {"ok": True}
