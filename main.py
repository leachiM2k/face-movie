"""Face-Movie 2.0 — face alignment + real morphing pipeline.

Pipeline:
    1. Detect 468 face landmarks per input image (MediaPipe Face Mesh).
    2. Procrustes-align every image onto a canonical canvas using a stable
       5-point landmark subset (eyes outer corners, nose tip, mouth corners).
    3. Compute Delaunay triangulation once on the mean of the aligned
       landmark sets (plus image-boundary points so warps cover the canvas).
    4. For each consecutive pair (A, B), render N intermediate frames via
       per-triangle affine warps + cross-dissolve — classic Beier/Wolberg-style
       face morphing.
    5. Pipe raw frames to ffmpeg for H.264 encode (h264_videotoolbox on
       macOS, libx264 on Linux).

Runs natively on macOS (Apple Silicon + Intel) and Linux. No GPU required.
"""

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

# MediaPipe FaceLandmarker model bundle. Cached under ~/.cache/face-movie.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path.home() / ".cache" / "face-movie" / "face_landmarker.task"


# Stable 5-point subset from the MediaPipe Face Mesh topology.
# Used for Procrustes alignment — robust across head pose and expression.
ANCHOR_INDICES = (
    33,   # right eye outer corner
    263,  # left eye outer corner
    1,    # nose tip
    61,   # right mouth corner
    291,  # left mouth corner
)


def ensure_model() -> Path:
    """Download the FaceLandmarker model on first run; cache it under ~/.cache."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading face landmarker model -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def make_landmarker() -> mp_vision.FaceLandmarker:
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def detect_landmarks(image_bgr: np.ndarray, landmarker) -> np.ndarray | None:
    """Return Nx2 landmark pixel coordinates, or None if no face was found.

    The FaceLandmarker returns 478 points (468 face mesh + 10 iris). We keep
    the first 468 — the iris points jitter when eyes blink and aren't useful
    for morph triangulation.
    """
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0][:468]
    return np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)


def canonical_template(canvas_w: int, canvas_h: int) -> np.ndarray:
    """Target positions (canvas coords) for the 5 anchor points.

    Face is centered horizontally; eyes sit ~40% down, mouth ~58% down. The
    inter-eye distance is sized to ~28% of the short edge so a typical portrait
    fills the frame without cropping the chin in landscape, nor cropping the
    sides in portrait.
    """
    cx = canvas_w / 2
    short = min(canvas_w, canvas_h)
    eye_dx = 0.28 * short / 2
    mouth_dx = 0.10 * short / 2
    return np.array([
        (cx - eye_dx,  0.40 * canvas_h),  # right eye
        (cx + eye_dx,  0.40 * canvas_h),  # left eye
        (cx,           0.50 * canvas_h),  # nose tip
        (cx - mouth_dx, 0.58 * canvas_h), # right mouth corner
        (cx + mouth_dx, 0.58 * canvas_h), # left mouth corner
    ], dtype=np.float32)


def align_to_canvas(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    template: np.ndarray,
    canvas_w: int,
    canvas_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply similarity transform so anchors match the template.

    Returns (aligned_image, aligned_landmarks).
    """
    src = landmarks[list(ANCHOR_INDICES)]
    M, _ = cv2.estimateAffinePartial2D(src, template, method=cv2.RANSAC)
    if M is None:
        raise RuntimeError("could not estimate similarity transform")
    aligned = cv2.warpAffine(
        image_bgr, M, (canvas_w, canvas_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    pts = np.hstack([landmarks, np.ones((len(landmarks), 1), dtype=np.float32)])
    aligned_lm = (pts @ M.T).astype(np.float32)
    # Pixels outside the canvas don't exist in `aligned`; clamping is correct
    # and prevents triangle bounding rects from slicing past the dst array.
    aligned_lm[:, 0] = np.clip(aligned_lm[:, 0], 0, canvas_w - 1)
    aligned_lm[:, 1] = np.clip(aligned_lm[:, 1], 0, canvas_h - 1)
    return aligned, aligned_lm


def boundary_points(canvas_w: int, canvas_h: int) -> np.ndarray:
    """Pinned points on the canvas border so warps cover the whole frame."""
    xs = np.linspace(0, canvas_w - 1, 4)
    ys = np.linspace(0, canvas_h - 1, 4)
    pts = []
    for x in xs:
        pts.append((x, 0))
        pts.append((x, canvas_h - 1))
    for y in ys[1:-1]:
        pts.append((0, y))
        pts.append((canvas_w - 1, y))
    return np.array(pts, dtype=np.float32)


def delaunay_triangles(
    points: np.ndarray, canvas_w: int, canvas_h: int,
) -> list[tuple[int, int, int]]:
    """Compute Delaunay triangulation on `points`. Returns triangle index triples."""
    rect = (0, 0, canvas_w, canvas_h)
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))
    raw = subdiv.getTriangleList()

    # Map triangle vertices back to indices in `points`.
    lookup = {(round(float(p[0]), 1), round(float(p[1]), 1)): i for i, p in enumerate(points)}
    triangles: list[tuple[int, int, int]] = []
    for t in raw:
        verts = [(round(float(t[i]), 1), round(float(t[i + 1]), 1)) for i in (0, 2, 4)]
        idx = [lookup.get(v) for v in verts]
        if all(i is not None for i in idx):
            triangles.append((idx[0], idx[1], idx[2]))  # type: ignore[arg-type]
    return triangles


def warp_triangle(
    src: np.ndarray, dst: np.ndarray,
    t_src: np.ndarray, t_dst: np.ndarray,
) -> None:
    """Affine-warp the triangle from `src` into `dst` (modifies dst in place)."""
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    if r1[2] == 0 or r1[3] == 0 or r2[2] == 0 or r2[3] == 0:
        return

    t1_local = np.float32([(p[0] - r1[0], p[1] - r1[1]) for p in t_src])
    t2_local = np.float32([(p[0] - r2[0], p[1] - r2[1]) for p in t_dst])

    src_patch = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    M = cv2.getAffineTransform(t1_local, t2_local)
    warped = cv2.warpAffine(
        src_patch, M, (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_local), (1.0, 1.0, 1.0), 16, 0)

    region = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    region[:] = region * (1 - mask) + warped * mask


def morph_pair(
    img1: np.ndarray, img2: np.ndarray,
    pts1: np.ndarray, pts2: np.ndarray,
    triangles: list[tuple[int, int, int]],
    n_frames: int,
    include_endpoint: bool,
) -> Iterable[np.ndarray]:
    """Yield N morphed frames from (img1, pts1) to (img2, pts2)."""
    img1f = img1.astype(np.float32)
    img2f = img2.astype(np.float32)

    if include_endpoint:
        ts = np.linspace(0, 1, n_frames + 1, endpoint=True)
    else:
        ts = np.linspace(0, 1, n_frames, endpoint=False)

    for t in ts:
        pts_t = (1 - t) * pts1 + t * pts2
        warped1 = np.zeros_like(img1f)
        warped2 = np.zeros_like(img1f)
        for a, b, c in triangles:
            tri_t = pts_t[[a, b, c]]
            warp_triangle(img1f, warped1, pts1[[a, b, c]], tri_t)
            warp_triangle(img2f, warped2, pts2[[a, b, c]], tri_t)
        frame = (1 - t) * warped1 + t * warped2
        yield np.clip(frame, 0, 255).astype(np.uint8)


def pick_encoder() -> list[str]:
    """Prefer hardware encoder on macOS; fall back to libx264."""
    if sys.platform == "darwin":
        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, check=False,
            )
            if "h264_videotoolbox" in r.stdout:
                return ["-c:v", "h264_videotoolbox", "-b:v", "8M"]
        except FileNotFoundError:
            pass
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]


def encode_video(
    frames: Iterable[np.ndarray],
    output_path: Path,
    canvas_w: int,
    canvas_h: int,
    fps: int,
) -> None:
    """Pipe BGR frames into ffmpeg as raw video and encode to H.264."""
    # H.264 requires even dimensions for yuv420p — round up via crop (cheap).
    pad_w = canvas_w + (canvas_w & 1)
    pad_h = canvas_h + (canvas_h & 1)
    vf_args = []
    if pad_w != canvas_w or pad_h != canvas_h:
        vf_args = ["-vf", f"pad={pad_w}:{pad_h}"]

    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{canvas_w}x{canvas_h}",
        "-r", str(fps),
        "-i", "-",
        *vf_args,
        *pick_encoder(),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with status {rc}")


def draw_overlay(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw a filename caption in the bottom-left corner (matches the legacy tool)."""
    out = frame.copy()
    h = out.shape[0]
    pos = (10, h - 12)
    # Black outline + white fill for legibility on any background.
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Align faces and morph between them.")
    ap.add_argument("--input", default="payload/input", type=Path)
    ap.add_argument("--output", default="payload/output", type=Path,
                    help="Directory for aligned still frames (only used with --keep-aligned).")
    ap.add_argument("--video", default="payload/out_morphed.mp4", type=Path)
    ap.add_argument("--width", type=int, default=None,
                    help="Output width. Default: auto-detect from first input image.")
    ap.add_argument("--height", type=int, default=None,
                    help="Output height. Default: auto-detect from first input image.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Proportional scale on the auto-detected canvas (0.5 = half).")
    ap.add_argument("--frames-per-pair", default=6, type=int,
                    help="Morph frames generated between each pair of photos.")
    ap.add_argument("--fps", default=30, type=int)
    ap.add_argument("--no-overlay", action="store_true",
                    help="Disable the filename caption burned into each frame.")
    ap.add_argument("--keep-aligned", action="store_true",
                    help="Also dump aligned still frames into --output.")
    args = ap.parse_args()

    files = sorted(
        p for p in args.input.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(files) < 2:
        print(f"need at least 2 images in {args.input}", file=sys.stderr)
        return 1

    # Determine canvas dimensions — explicit flags win, else first valid image.
    if args.width and args.height:
        canvas_w, canvas_h = args.width, args.height
    else:
        probe = None
        for f in files:
            probe = cv2.imread(str(f))
            if probe is not None:
                break
        if probe is None:
            print(f"could not read any image in {args.input}", file=sys.stderr)
            return 1
        h0, w0 = probe.shape[:2]
        canvas_w = args.width or int(round(w0 * args.scale))
        canvas_h = args.height or int(round(h0 * args.scale))

    print(f"[1/4] detecting landmarks in {len(files)} images "
          f"(canvas {canvas_w}x{canvas_h})")
    template = canonical_template(canvas_w, canvas_h)
    aligned_images: list[np.ndarray] = []
    aligned_landmarks: list[np.ndarray] = []
    used_files: list[Path] = []
    skipped: list[Path] = []

    landmarker = make_landmarker()
    try:
        for f in tqdm(files):
            img = cv2.imread(str(f))
            if img is None:
                skipped.append(f)
                continue
            lm = detect_landmarks(img, landmarker)
            if lm is None:
                skipped.append(f)
                continue
            try:
                aligned_img, aligned_lm = align_to_canvas(
                    img, lm, template, canvas_w, canvas_h,
                )
            except RuntimeError:
                skipped.append(f)
                continue
            aligned_images.append(aligned_img)
            aligned_landmarks.append(aligned_lm)
            used_files.append(f)
    finally:
        landmarker.close()

    if skipped:
        print(f"  skipped {len(skipped)} (no face / unreadable):")
        for p in skipped[:10]:
            print(f"    - {p.name}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    if len(aligned_images) < 2:
        print("not enough usable images after alignment", file=sys.stderr)
        return 1

    if args.keep_aligned:
        args.output.mkdir(parents=True, exist_ok=True)
        for f, img in zip(used_files, aligned_images, strict=True):
            cv2.imwrite(str(args.output / f"c_{f.name}"), img)

    print("[2/4] computing template + Delaunay triangulation")
    bnd = boundary_points(canvas_w, canvas_h)
    full_landmarks = [np.vstack([lm, bnd]) for lm in aligned_landmarks]
    mean_pts = np.mean(np.stack(full_landmarks), axis=0)
    triangles = delaunay_triangles(mean_pts, canvas_w, canvas_h)
    print(f"  {len(triangles)} triangles over {len(mean_pts)} vertices")

    n_pairs = len(aligned_images) - 1
    print(f"[3/4] morphing {n_pairs} pairs × {args.frames_per_pair} frames each")

    def all_frames() -> Iterable[np.ndarray]:
        pbar = tqdm(total=n_pairs)
        for i in range(n_pairs):
            include_endpoint = (i == n_pairs - 1)
            label_a = used_files[i].stem
            label_b = used_files[i + 1].stem
            for j, frame in enumerate(morph_pair(
                aligned_images[i], aligned_images[i + 1],
                full_landmarks[i], full_landmarks[i + 1],
                triangles, args.frames_per_pair, include_endpoint,
            )):
                if not args.no_overlay:
                    # Show A's label until t crosses 0.5, then B's. With
                    # frames_per_pair=N, t = j/N (or j/(N-1) on last pair).
                    t = j / args.frames_per_pair
                    frame = draw_overlay(frame, label_b if t >= 0.5 else label_a)
                yield frame
            pbar.update(1)
        pbar.close()

    print(f"[4/4] encoding to {args.video}")
    args.video.parent.mkdir(parents=True, exist_ok=True)
    encode_video(all_frames(), args.video, canvas_w, canvas_h, args.fps)

    total_frames = n_pairs * args.frames_per_pair + 1
    print(f"done: {args.video} ({total_frames} frames @ {args.fps} fps "
          f"= {total_frames / args.fps:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
