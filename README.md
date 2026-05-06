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

## Why it works

This is real face morphing — not a crossfade. The pipeline:

1. **Detects 468 face landmarks** per image with MediaPipe Face Mesh
2. **Aligns** each face onto a canonical pose with a Procrustes transform
   (eyes, nose, and mouth corners are the anchors — robust across head tilt,
   pose, and expression)
3. **Triangulates** the mean face shape with Delaunay
4. **Morphs** every consecutive pair via piecewise-affine warps over the
   triangle mesh, plus a cross-dissolve in pixel space
5. **Encodes** H.264 with `h264_videotoolbox` on macOS or `libx264`
   elsewhere — no GPU required

A photo from 2016 visibly *becomes* a photo from 2024, instead of fading
through a ghost.

## Quick start

### Docker

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
python main.py
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

For 750 photos at default settings, expect 10–15 min on an M-series Mac.

## What's under the hood

Python 3.12 · MediaPipe Face Mesh · OpenCV · ffmpeg. No GPU. Native on
macOS (Apple Silicon + Intel) and Linux (x86 + ARM).

The 2.x rewrite replaced TensorFlow + dlib + face-recognition + MTCNN
with MediaPipe alone, dropped the Docker image from ~1.5 GB to ~600 MB,
and cut the build time from 14 min to ~2 min.

## Background

Started in August 2016 as a personal project. The author began taking a
selfie every morning at 9am. Five years and 750 photos later, a naive
slideshow looked like a strobe — head in a different place, different
tilt, every frame. Face-Movie was the fix.
