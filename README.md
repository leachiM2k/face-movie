# Face-Movie

> **Take a selfie every morning. Years later, watch yourself grow up.**

Face-Movie turns a folder of portraits into a smoothly morphed time-lapse
video. Faces stay locked in place — eyes at the same spot in every frame —
while expressions, lighting, and the years flow past.

![demo](docs/demo.gif)

## What it's for

- The "one selfie a day" project, finally watchable
- Beard, hair-loss, weight-change, gym, or makeup journeys
- Pregnancy progression, post-surgery recovery
- Watching kids grow up
- Any series of portraits where the face moves around between shots

## Why?

Google Picasa 3 used to ship a feature called **Face Movie** that did
exactly this — align faces across a batch of photos and stitch them into
a smooth video. Picasa was discontinued in 2016, and nothing in the major
photo apps has replaced it.

This project began in August 2016. The author started taking a selfie
every morning at 9am, planning to one day watch a smooth time-lapse of
how he changed. Five years and 750 photos later, a naive slideshow looked
like a strobe — head in a different spot, different tilt, every single
frame. Face-Movie is the fix: what Picasa 3 used to do, kept alive with
a modern toolchain.

## How it works

This is real face morphing — not a crossfade. The pipeline:

1. **Detects 468 face landmarks** per image with MediaPipe Face Mesh
2. **Aligns** each face onto a canonical pose with a Procrustes transform
   (eyes, nose, and mouth corners are the anchors — robust across head tilt,
   pose, and expression)
3. **Triangulates** the mean face shape with Delaunay
4. **Morphs** every consecutive pair via piecewise-affine warps over the
   triangle mesh, plus a cross-dissolve in pixel space
5. **Encodes** H.264 with the best hardware encoder available — `h264_videotoolbox`
   on macOS, `h264_nvenc` / `h264_qsv` / `h264_amf` / `h264_v4l2m2m` on Linux,
   or `libx264` if no HW path is present. MediaPipe also uses the GPU delegate
   (Metal / OpenGL ES) for landmark detection when present. CPU-only systems
   still work unchanged.

A photo from 2016 visibly *becomes* a photo from 2024, instead of fading
through a ghost.

## Quick start

### Docker — Web UI

```bash
docker run --rm -p 8080:8080 leachim2k/face-movie:latest web
```

Open <http://localhost:8080>, drop in a folder of selfies, watch the result.
Photos never leave your machine — the container processes everything locally.

### Docker — CLI

```bash
docker run --rm -v "$PWD/payload:/app/payload" leachim2k/face-movie:latest
```

Drop your portraits into `payload/input/` first. The result lands at
`payload/out_morphed.mp4`.

### Python

```bash
git clone https://github.com/leachiM2k/face-movie.git
cd face-movie
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py                                                      # CLI
uvicorn webapp.server:app --host 127.0.0.1 --port 8080              # Web UI
```

Three sample portraits ship in `payload/input/` so you can see output
on the first run without supplying any photos of your own.

## Options

| Flag | Default | What it does |
|------|---------|--------------|
| `--input DIR` | `payload/input` | source folder of JPEG/PNG portraits |
| `--video PATH` | `payload/out_morphed.mp4` | where to write the result |
| `--scale N` | `1.0` | proportional scale on the auto-detected canvas (`0.5` = half-size) |
| `--width N` / `--height N` | auto | override canvas dimensions explicitly |
| `--frames-per-pair N` | `6` | morph frames between two photos — higher = slower transitions |
| `--fps N` | `30` | output frame rate |
| `--no-overlay` | off | disable the burned-in filename caption |
| `--keep-aligned` | off | also dump aligned still frames (debug) |
| `--encoder NAME` | auto | force a specific ffmpeg encoder (`libx264`, `h264_nvenc`, …) |
| `--front-facing-only` | off | skip photos where the head is turned away from the camera |
| `--max-asymmetry N` | `0.20` | front-facing tolerance — lower = stricter (only with `--front-facing-only`) |
| `--halo-factor N` | `1.25` | extra anchor ring around the face for smoother face-edge morphing — `1.0` disables |

The web UI exposes a **"Only include front-facing photos"** checkbox for
the same filter, useful when cleaning up a large archive that mixes
selfies with off-camera shots.

Environment variable `FACE_MOVIE_DELEGATE=cpu` forces MediaPipe onto the CPU
path — useful if the GPU delegate aborts on your driver/SDK combination.

For 750 photos at default settings, expect 10–15 min on an M-series Mac.

## What's under the hood

Python 3.12 · MediaPipe Face Mesh · OpenCV · ffmpeg. No GPU. Native on
macOS (Apple Silicon + Intel) and Linux (x86 + ARM).

The 2.x rewrite replaced TensorFlow + dlib + face-recognition + MTCNN
with MediaPipe alone, dropped the Docker image from ~1.5 GB to ~600 MB,
and cut the build time from 14 min to ~2 min.
