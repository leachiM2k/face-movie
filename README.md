# Face-Movie

Aligns faces across a series of photos and renders a smoothly morphed video —
for example, "one selfie a day for ten years" turning into a 30-second clip.

**Why?**

Since August 2016 I've been taking a selfie every morning at 9am. Now, several
hundred photos in, I wanted to watch how my face changed over the years. But
the head sits in a different spot, at a different tilt, in every photo — so a
naive slideshow looks like a strobe. This tool aligns each face to a canonical
pose, then morphs between consecutive photos with real piecewise-affine warps
(not just a crossfade).

## Pipeline

1. **Detect** 468 face landmarks per image with MediaPipe Face Mesh.
2. **Align** every image onto a canonical canvas via a Procrustes (similarity)
   transform on a stable 5-point subset (eye corners, nose tip, mouth corners).
3. **Triangulate** once: Delaunay over the mean of all aligned landmark sets,
   plus a grid of pinned boundary points so the warp covers the full canvas.
4. **Morph** N intermediate frames between every consecutive pair via
   per-triangle affine warps + cross-dissolve (Beier/Wolberg-style).
5. **Encode** to MP4 with ffmpeg — `h264_videotoolbox` on macOS, `libx264`
   elsewhere.

No GPU needed. Runs on macOS (Apple Silicon + Intel) and Linux.

## Quick start

Drop your JPEGs into `./payload/input` and run:

### Docker

```bash
docker run --rm -v "${PWD}/payload:/app/payload" --name face-movie \
    leachim2k/face-movie:latest
```

To build locally: `docker build --tag leachim2k/face-movie:latest .`
(image is ~600 MB; build takes ~2 minutes — down from 14 min in the old
TensorFlow-based version).

### Local Python

Requires Python 3.10–3.12 (mediapipe doesn't yet support 3.13+). Use a venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./start.sh
```

The output lands at `payload/out_morphed.mp4`.

## CLI options

```
python main.py [--input DIR]            # default: payload/input
               [--video PATH]           # default: payload/out_morphed.mp4
               [--size N]               # canvas px (default 720; try 1080 for sharper)
               [--frames-per-pair N]    # morph frames between photos (default 6)
               [--fps N]                # default 30
               [--keep-aligned]         # also dump aligned stills to --output
```

A 750-photo run with defaults produces a ~150-second video and takes roughly
10–15 minutes on an M-series Mac.

## Stack

- Python 3.12
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) Face Mesh
- OpenCV (Subdiv2D for Delaunay, warpAffine for piecewise warps)
- ffmpeg
