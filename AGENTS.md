# AGENTS.md — agent onboarding for face-movie

This file is for **agents picking the project up cold**. Read it once,
then act. It is not user-facing documentation — README.md is for users.

---

## What this project is

A Python tool that turns a series of portrait photos (e.g. "one selfie a
day since 2016") into a smoothly morphed time-lapse MP4. Faces stay
locked in place, real per-triangle morphing between consecutive shots.
Picasa 3 used to do this; Picasa was killed in 2016, this fills that gap.

Two surfaces:

- **CLI** — `python main.py` or the Docker default
- **Web UI** — `face-movie web`, FastAPI on port 8080, drop a folder of
  selfies, get a video back. Single-tenant, local-only by intent.

Distributed as `leachim2k/face-movie:latest` on Docker Hub. Multi-arch
(linux/amd64 + linux/arm64) via GitHub Actions on push to `main`.

---

## Repo layout

```
main.py                          single-file pipeline. The PUBLIC API is
                                 run_pipeline(...) -> RenderResult.
                                 The CLI in main() is a thin wrapper.

webapp/
  server.py                      FastAPI, in-memory job state. Wraps
                                 run_pipeline() with multipart upload,
                                 SSE progress, MP4 download.
  static/index.html              vanilla HTML+CSS+JS, no build step.
                                 Drag-drop files OR a folder.

.github/workflows/
  docker-publish.yml             multi-arch build + push to Docker Hub
                                 on `main` push, tag pushes (v*), PRs
                                 (build-only). Tags only `latest` after PR #19.
  dockerhub-description.yml      mirrors README.md to Hub overview on
                                 README change. Rewrites relative img
                                 paths to absolute GitHub raw URLs.

Dockerfile                       python:3.12-slim + ffmpeg + libgl1.
                                 ENTRYPOINT ./start.sh; CMD [].
start.sh                         dispatches: arg1=="web" → uvicorn,
                                 else → python main.py "$@".

payload/
  input/                         gitignored EXCEPT for sample_*.jpeg.
                                 The user's real selfies (~1800) also
                                 live here on their machine — do NOT
                                 touch them.
  out_morphed.mp4                tracked demo video (~1.6 MB) generated
                                 from the 3 Bezos samples.
  output/                        gitignored, used only with --keep-aligned

docs/
  demo.gif                       README hero image, 1.1 MB
  logo.png                       512x512 repo logo (PR #18); Docker Hub
                                 repo image must be uploaded MANUALLY
                                 (no public API for that).
```

---

## How the pipeline works (one paragraph)

`run_pipeline()` in `main.py`:

1. Detect 468 face landmarks per image (MediaPipe FaceLandmarker, GPU
   delegate where available, RGBA input). 478 are returned but the iris
   ones jitter, kept first 468 only.
2. Procrustes-align each image onto a canonical canvas using a stable
   5-point subset (eye corners, nose tip, mouth corners). Aligned
   landmarks are clipped to canvas bounds — chin/ear points sometimes
   land outside.
3. Build a Delaunay triangulation ONCE on the *mean* of the aligned
   landmark sets, with a 4×4 boundary-point grid added so warps cover
   the full canvas.
4. For each consecutive pair, generate N intermediate frames via
   per-triangle affine warps + cross-dissolve. Filename overlay
   switches at t≥0.5.
5. Pipe BGR frames to ffmpeg; encoder picked at runtime by probing
   `videotoolbox`/`nvenc`/`qsv`/`amf`/`v4l2m2m`/`libx264` (see Gotchas).

Memory peak ≈ N_used × W × H × 3 bytes (all aligned images held in RAM).

---

## Run it

### Local dev

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CLI smoke (3 bundled samples):
python main.py --input /tmp/face-movie-samples --video /tmp/out.mp4 --frames-per-pair 4 --scale 0.4

# Web UI dev:
uvicorn webapp.server:app --reload --host 127.0.0.1 --port 8080
```

The user's `.venv` is Python 3.11 (mediapipe 0.10.x doesn't have wheels
for 3.13/3.14 yet on macOS). Inside the Docker image it's 3.12.

### Docker

```bash
docker build -t face-movie:test .
docker run --rm -p 18080:8080 --name face-movie-test face-movie:test web
# CLI mode in the same image:
docker run --rm -v "$PWD/payload:/app/payload" face-movie:test --frames-per-pair 4 --scale 0.4
```

Use port 18080 in tests — 8080 is often busy.

### End-to-end smoke check

```bash
# (assumes container running on 18080)
curl -fsS -X POST http://127.0.0.1:18080/api/render \
  -F "files=@/tmp/face-movie-samples/sample_1.jpeg" \
  -F "files=@/tmp/face-movie-samples/sample_2.jpeg" \
  -F "files=@/tmp/face-movie-samples/sample_3.jpeg" \
  -F "scale=0.4" -F "frames_per_pair=2" -F "fps=30"
# → {"job_id":"<id>","files":3}

curl -fsSN --max-time 60 http://127.0.0.1:18080/api/events/<id>
# Last event should be: data: {"stage": "done", ...}

curl -fsS -o /tmp/out.mp4 http://127.0.0.1:18080/api/download/<id>
ffprobe -v quiet -print_format json -show_streams /tmp/out.mp4
```

The 3 samples render in ~1 second on M3 Max.

---

## Conventions

### Commit messages

Lowercase, imperative, descriptive subject + a "why" body. Match the
existing log:

```
modernize stack: mediapipe + delaunay morphing, drop tensorflow/dlib
clamp aligned landmarks to canvas bounds
ci: only push the `latest` tag
```

**Do NOT add `Co-Authored-By: ... <noreply@anthropic.com>` trailers.**
A repo content-integrity hook blocks them. Use a temp file passed via
`-F`; heredocs occasionally trip the bash parser:

```bash
git commit -F /tmp/commit-msg.txt
```

### Branching + PRs

- `main` is the only long-lived branch. Pushing directly to `main` is
  blocked by a hook — use a feature branch + PR.
- Branch names: `feat/...`, `ci/...`, `fix/...`.
- PR body via `--body-file` — but **the file must have been written or
  read in this transcript first**, or the hook denies "unverifiable
  agent-inferred parameters". Read the file before invoking `gh pr create`.
- One PR per coherent change. Multiple parallel PRs are fine if they
  don't conflict.

### Destructive git ops

`git reset --hard`, `git push --force`, `git branch -D` etc. on the
user's branches are blocked unless authorized for the specific scope.
If you need to rewind, ask first or use a soft path (e.g. branch off
`origin/main` to leave the local copy alone).

### Code style

- `main.py` is intentionally one file. Don't split prematurely.
- Comments only when the *why* is non-obvious (workaround, hidden
  invariant, surprising behavior). The user dislikes narrating-WHAT
  comments.
- Type hints on public APIs (`run_pipeline`, dataclasses). No need
  on internal helpers.

---

## Gotchas (things that already bit me)

These are bugs / surprises that already cost a debug round. Read them.

### MediaPipe

- **GPU delegate aborts on RGB input on macOS.** The Metal path expects
  RGBA. Pass `mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)`,
  not SRGB. The error is a `F0000` libabsl `LOG(FATAL)` that
  **terminates the Python process** — uncatchable from `try/except`.
- Escape hatch for the GPU path: `FACE_MOVIE_DELEGATE=cpu` env var
  forces CPU XNNPACK. Use it if a future driver/SDK combo crashes.
- `mp.solutions.face_mesh` was removed somewhere around mediapipe
  0.10.30. Use `mp.tasks.vision.FaceLandmarker` exclusively. The model
  bundle is downloaded on first run to `~/.cache/face-movie/`.

### ffmpeg encoder selection

- `ffmpeg -encoders` listing only proves compile-time inclusion, **not**
  runtime usability. The slim apt ffmpeg in the Docker image lists
  `h264_nvenc`, but it dies with `Cannot load libcuda.so.1` at runtime.
  `pick_encoder()` therefore probes each candidate with a 1-frame
  null-output encode before committing. Never trust the listing alone.
- Probe results are cached at module scope. Restart the process to
  reprobe.

### Numerics + OpenCV

- **`cv2.boundingRect` of a triangle vertex right at the canvas edge
  can yield a rect that extends past the canvas array.** numpy slices
  the dst silently to size, but mask/warped stay full size → shape
  mismatch ValueError. Fix: clip aligned landmarks to `[0, W-1] × [0, H-1]`
  in `align_to_canvas()`. Already done — don't undo it.
- The Delaunay-triangle index lookup uses a `{(rounded x, rounded y) → i}`
  dict. Point coincidences after averaging across 1000+ images are
  rare but possible. If you ever see triangles drop out of the mesh
  unexpectedly, that's the first place to look.
- `halo_points()` adds 36 anchors around the face oval, extrapolated
  outward by `halo_factor` (default 1.25). Without them the area
  between face-edge and canvas-edge is one big triangle that pulls on
  the face-oval anchors as the face shape changes — visible as
  rubbery edges. Halo points are clipped to canvas (extrapolation can
  push them off-image when the face fills the frame).

### Pose filtering

- `is_front_facing()` uses MediaPipe's `facial_transformation_matrixes`
  (a 4×4 mapping from canonical face → camera). Column 2 of the
  rotation 3×3 is the face's forward axis in camera coords; its z
  component is `cos(angle to camera)`. Threshold against
  `cos(max_head_tilt_deg)`.
- An earlier version used 2D-symmetry of the 5 anchors (`max_asymmetry`).
  It was structurally blind to profiles: MediaPipe Face Mesh fills
  occluded landmarks from a 3D template, so the 2D points stay roughly
  symmetric even at 80° yaw. Don't reintroduce that path.
- MediaPipe sometimes returns `face_landmarks` but no
  `facial_transformation_matrixes` — almost always for full-profile
  shots where the pose solver fails. We treat `matrix is None` as
  "not front-facing" so those still get filtered.
- `make_landmarker()` must enable `output_facial_transformation_matrixes=True`
  for the filter to work.

### FastAPI / Starlette / multipart

- **`from fastapi import UploadFile` returns FastAPI's *subclass* of
  `starlette.datastructures.UploadFile`.** When you parse the form
  yourself via `request.form()`, you get plain Starlette UploadFile
  instances and `isinstance(f, fastapi.UploadFile)` returns False.
  Import directly: `from starlette.datastructures import UploadFile`.
- python-multipart 0.0.18+ caps multipart uploads at 1000 files by
  default. FastAPI's `File()`/`Form()` dependencies don't expose the
  knob. We bypass by calling `request.form(max_files=…)` directly.
  Current cap is 25 000.

### Docker

- `CMD ["./start.sh"]` is **overridden** when the user runs
  `docker run face-movie web` — Docker treats `web` as the new command
  and tries to exec it as a binary. Use `ENTRYPOINT ["./start.sh"]` +
  `CMD []` so runtime args are passed *to* the script.
- Test the container, not just `uvicorn` locally. The first webapp PR
  shipped with this bug because I only tested locally — the user
  caught it on first `docker run`.

### Git on this repo

- The default branch was renamed `master → main`. The first push of
  `main` was triggered before `gh repo edit --default-branch main`
  could propagate, so the metadata-action's `{{is_default_branch}}`
  resolved to `false` for that one run. Subsequent runs are fine.
- The old `master` branch had a divergent abandoned dlib-based
  prototype (`facemesh.py`, no `payload/` layout). It was deleted with
  no loss.
- Only one tag is pushed: `latest`. PR #19 simplified from `main +
  sha-<short>` + several others. Don't reintroduce SHA tags without
  asking — the user explicitly asked for one moving target.

---

## Active state (as of last edit)

| Branch on origin | Status |
|---|---|
| `main` | trunk |
| `feat/repo-logo` | open PR #18 — adds docs/logo.png + README header img |
| `ci/only-latest-tag` | open PR #19 — strip tags down to just `latest` |
| `feat/webapp` | open PR #20 — web UI + GPU/HW-encoder + multipart fix |

PRs are independent (different files) — merge in any order. After merge,
`docker-publish.yml` will fire on `main` push and produce the new image.
`dockerhub-description.yml` only fires when README.md is touched.

The user's local `main` may have one unpushed commit ahead of `origin/main`
(an early CI tweak that's superseded by PR #19's simpler version). Either
let it linger or `git reset --hard origin/main` once the PRs land — needs
explicit user OK because of the destructive-ops hook.

### Required GitHub secrets (already set)

- `DOCKERHUB_USERNAME` = `leachim2k`
- `DOCKERHUB_TOKEN` — Docker Hub PAT, scope **Read/Write/Delete**
  (the description-update endpoint needs the bigger scope; the
  smaller "Read & Write" returns Forbidden on PATCH)

### Manual ops the user owns

- Docker Hub repo logo upload (no public API). User uploads
  `docs/logo.png` once via the Hub web UI under "General" →
  "Repository image".
- Cleanup of legacy tags on Docker Hub (`main`, `sha-d670b8a`) once
  PR #19 merges.

---

## What's deliberately out of scope

Don't add these without explicit ask:

- **CUDA OpenCV** for `cv2.cuda.warpAffine` morph — would 5× the morph
  step but needs a separate CUDA-only image variant (~3 GB), NVIDIA
  Container Toolkit, plus a parallel Dockerfile/workflow. Deferred
  until requested.
- **Live preview** during morphing in the web UI — nice but adds
  complexity and the user picked v1 = no preview.
- **Multi-tenant queue / auth** — project intent is the local-Docker
  single-tenant model. The README explicitly tells users not to expose
  this publicly.
- **License file** — user hasn't decided. Don't claim MIT or anything else.
- **README "Background"** as a separate section from "Why" — they were
  merged. The "Why?" section is the canonical motivation block, with
  the Picasa 3 lineage call-out.

---

## Performance ballpark (M3 Max, native)

| Run | Frames | Time | Encoder |
|---|---|---|---|
| 3 samples, scale 0.4, frames-per-pair 4 | 9 | ~0.3 s | videotoolbox |
| 3 samples, scale 0.5, frames-per-pair 24 | 49 | ~2 s | videotoolbox |
| 1224 selfies, scale 0.4, frames-per-pair 4 | ~4341 | ~3 min | videotoolbox |
| same in linux/amd64 Docker (no GPU) | same | ~10–15 min | libx264 |

Useful when the user asks "how long will my N photos take".

---

## Communication

The user (`leachiM2k`) prefers **German** for explanations and rationale.
Commit messages and code comments are English. Be terse. Say what,
why, and what's next — no padding, no marketing voice.
