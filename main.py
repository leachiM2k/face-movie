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
    5. Pipe raw frames to ffmpeg for H.264 encode (hardware encoder when
       available — h264_videotoolbox on macOS, h264_nvenc / h264_qsv /
       h264_amf / h264_v4l2m2m on Linux — falls back to libx264).

Runs natively on macOS (Apple Silicon + Intel) and Linux. GPU is used
opportunistically when present (MediaPipe landmark detection + ffmpeg HW
encoder); CPU-only systems work unchanged.

Public API:
    run_pipeline(input_dir, output_path, ..., on_progress=cb) -> RenderResult

The CLI in main() is a thin wrapper that turns the on_progress callback
into tqdm progress bars. The web UI uses the same callback to push SSE
events to the browser.
"""

import argparse
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

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

# MediaPipe FACEMESH_FACE_OVAL — the 36 indices that trace the face contour
# (forehead → temple → cheek → jaw → chin → up the other side). Used to
# build the "halo" ring of extra anchors just outside the face.
FACE_OVAL_INDICES = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)

ProgressCallback = Callable[[str, int, int], None]


@dataclass
class RenderResult:
    output_path: Path
    canvas_w: int
    canvas_h: int
    fps: int
    total_frames: int
    used_files: list[Path]
    skipped_files: list[Path] = field(default_factory=list)
    pose_filtered_files: list[Path] = field(default_factory=list)
    encoder: str = ""

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.fps


def ensure_model() -> Path:
    """Download the FaceLandmarker model on first run; cache it under ~/.cache."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading face landmarker model -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def make_landmarker() -> mp_vision.FaceLandmarker:
    """Try the GPU delegate first, fall back to CPU.

    On macOS the GPU delegate uses Metal via TensorFlow Lite; on Linux it
    needs an OpenGL ES context. Init failures raise and we move on to CPU.

    Set FACE_MOVIE_DELEGATE=cpu to skip the GPU attempt entirely. This is
    the escape hatch for systems where the GPU path passes init but aborts
    deep inside libabsl on first detection (uncatchable from Python).
    """
    model_path = str(ensure_model())
    Delegate = mp_tasks.BaseOptions.Delegate

    forced = os.environ.get("FACE_MOVIE_DELEGATE", "auto").lower()
    if forced == "cpu":
        order = (Delegate.CPU,)
    elif forced == "gpu":
        order = (Delegate.GPU,)
    else:
        order = (Delegate.GPU, Delegate.CPU)

    for delegate in order:
        try:
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=model_path,
                    delegate=delegate,
                ),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                # Returns the 4×4 transform from the canonical face model to
                # camera space; we use it for an honest head-pose filter.
                output_facial_transformation_matrixes=True,
            )
            return mp_vision.FaceLandmarker.create_from_options(options)
        except Exception:
            continue
    raise RuntimeError("could not initialize MediaPipe FaceLandmarker")


def detect_landmarks(
    image_bgr: np.ndarray, landmarker,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Return ((Nx2 landmarks), 4x4 pose matrix or None), or None if no face.

    The FaceLandmarker returns 478 points (468 face mesh + 10 iris). We keep
    the first 468 — the iris points jitter when eyes blink and aren't useful
    for morph triangulation. RGBA is used (not RGB) so the Metal/OpenGL GPU
    delegate path doesn't abort on the ImageFrame conversion.

    The pose matrix is `None` when MediaPipe found a face but couldn't fit a
    transform — typically extreme profile shots where the 3D solver fails.
    Treated as "not front-facing" by the filter.
    """
    h, w = image_bgr.shape[:2]
    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0][:468]
    coords = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
    matrix = None
    if result.facial_transformation_matrixes:
        matrix = np.array(result.facial_transformation_matrixes[0], dtype=np.float32)
    return coords, matrix


def is_front_facing(matrix: np.ndarray | None, max_head_tilt_deg: float) -> bool:
    """True if the head is roughly facing the camera, by 3D pose.

    MediaPipe's facial transformation matrix maps the canonical face model
    into camera space. Its rotation 3×3 carries the head pose; column 2 is
    the face's forward axis (out of the nose). Its z-component equals
    cos(angle between forward and the camera optical axis) — exactly what
    we need.

    Threshold tuning:
        max_head_tilt_deg = 15  → cos ≈ 0.966   strict frontal
        max_head_tilt_deg = 20  → cos ≈ 0.940   default, ~all clean selfies pass
        max_head_tilt_deg = 30  → cos ≈ 0.866   loose, lets through ~30° turns

    A `None` matrix (face detected but pose solver failed) counts as not
    front-facing — that case shows up almost exclusively for full profiles.

    The 2D-symmetry filter this replaces missed real profiles entirely:
    MediaPipe Face Mesh interpolates occluded landmarks from a 3D template,
    so 2D symmetry stays high even when the head is fully turned.
    """
    if matrix is None:
        return False
    forward_z = float(matrix[2, 2])
    return forward_z >= float(np.cos(np.radians(max_head_tilt_deg)))


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


def halo_points(
    landmarks: np.ndarray,
    factor: float,
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    """Extrapolate FACEMESH_FACE_OVAL points outward from the face center.

    The face mesh densely covers the face surface but stops at the oval.
    Between the oval and the canvas border, the triangulation has only the
    sparse boundary points, so the few large triangles spanning that gap
    pull on the face-edge anchors as the face shape changes — visible as
    a "rubbery edge" in the morph.

    For each oval landmark P we add an extra anchor at:
        center + (P - center) * factor

    A factor of 1 disables the ring (anchors coincide with the oval, no
    benefit); 1.25 is a gentle outward push that adds dedicated anchors
    in the immediate ring around the face. Points are clipped to the
    canvas because the extrapolation can push them off-image when the
    face fills the frame.
    """
    if factor <= 1.0:
        return np.empty((0, 2), dtype=np.float32)
    oval = landmarks[list(FACE_OVAL_INDICES)]
    center = oval.mean(axis=0)
    halo = (center + (oval - center) * factor).astype(np.float32)
    halo[:, 0] = np.clip(halo[:, 0], 0, canvas_w - 1)
    halo[:, 1] = np.clip(halo[:, 1], 0, canvas_h - 1)
    return halo


def delaunay_triangles(
    points: np.ndarray, canvas_w: int, canvas_h: int,
) -> list[tuple[int, int, int]]:
    """Compute Delaunay triangulation on `points`. Returns triangle index triples."""
    rect = (0, 0, canvas_w, canvas_h)
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))
    raw = subdiv.getTriangleList()

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


def _ffmpeg_encoders() -> set[str]:
    """Names of all encoders the ffmpeg binary advertises."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return set()
    names: set[str] = set()
    for line in r.stdout.splitlines():
        # encoder lines look like " V..... h264_nvenc           NVIDIA NVENC ..."
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return names


def _encoder_runtime_works(codec: str, extra: list[str]) -> bool:
    """Probe a codec with a one-frame null-output encode.

    `ffmpeg -encoders` listing only proves the codec was compiled in, not
    that its runtime dependencies are present (e.g., h264_nvenc is in every
    apt ffmpeg build but needs libcuda + an NVIDIA device at run time).
    """
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1:r=1",
                "-c:v", codec, *extra,
                "-pix_fmt", "yuv420p",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


_picked_encoder_cache: tuple[str, list[str]] | None = None


def pick_encoder(override: str | None = None) -> tuple[str, list[str]]:
    """Choose an H.264 encoder. Returns (name, ffmpeg-args).

    Honours --encoder override; otherwise prefers any hardware encoder that
    survives a runtime probe, and falls back to libx264. The auto choice is
    cached at process scope so repeated run_pipeline() calls don't reprobe.
    """
    if override:
        candidate = override, ["-c:v", override]
        if not _encoder_runtime_works(override, []):
            raise RuntimeError(
                f"encoder {override!r} cannot encode (missing driver / device?)"
            )
        return candidate

    global _picked_encoder_cache
    if _picked_encoder_cache is not None:
        return _picked_encoder_cache

    available = _ffmpeg_encoders()
    candidates: list[tuple[str, list[str]]] = []
    if sys.platform == "darwin":
        candidates.append(("h264_videotoolbox", ["-b:v", "8M"]))
    candidates.extend([
        ("h264_nvenc",     ["-preset", "p4", "-b:v", "8M"]),  # NVIDIA
        ("h264_qsv",       ["-b:v", "8M"]),                   # Intel QuickSync
        ("h264_amf",       ["-b:v", "8M"]),                   # AMD VCN
        ("h264_v4l2m2m",   ["-b:v", "8M"]),                   # Raspberry Pi 4+
    ])

    for codec, extra in candidates:
        if codec in available and _encoder_runtime_works(codec, extra):
            _picked_encoder_cache = codec, ["-c:v", codec, *extra]
            return _picked_encoder_cache

    _picked_encoder_cache = "libx264", ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    return _picked_encoder_cache


def encode_video(
    frames: Iterable[np.ndarray],
    output_path: Path,
    canvas_w: int,
    canvas_h: int,
    fps: int,
    encoder_args: list[str],
) -> None:
    """Pipe BGR frames into ffmpeg as raw video and encode to H.264."""
    pad_w = canvas_w + (canvas_w & 1)
    pad_h = canvas_h + (canvas_h & 1)
    vf_args: list[str] = []
    if pad_w != canvas_w or pad_h != canvas_h:
        vf_args = ["-vf", f"pad={pad_w}:{pad_h}"]

    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{canvas_w}x{canvas_h}",
        "-r", str(fps),
        "-i", "-",
        *vf_args,
        *encoder_args,
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
    """Draw a filename caption in the bottom-left corner."""
    out = frame.copy()
    h = out.shape[0]
    pos = (10, h - 12)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_dir: Path,
    output_path: Path,
    *,
    width: int | None = None,
    height: int | None = None,
    scale: float = 1.0,
    frames_per_pair: int = 6,
    fps: int = 30,
    overlay: bool = True,
    keep_aligned_dir: Path | None = None,
    encoder: str | None = None,
    front_facing_only: bool = False,
    max_head_tilt_deg: float = 20.0,
    halo_factor: float = 1.25,
    on_progress: ProgressCallback | None = None,
) -> RenderResult:
    """Run the full face-movie pipeline.

    `on_progress(stage, current, total)` is called as work proceeds. Stages:
        "detect"      — landmark detection (current = files done, total = files)
        "triangulate" — single tick at completion (1, 1)
        "morph"       — pair morphing (current = pairs done, total = pairs)
        "done"        — completed (1, 1)
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(files) < 2:
        raise ValueError(f"need at least 2 images in {input_dir}")

    # Determine canvas dimensions — explicit args win, else first valid image.
    if width and height:
        canvas_w, canvas_h = width, height
    else:
        probe = None
        for f in files:
            probe = cv2.imread(str(f))
            if probe is not None:
                break
        if probe is None:
            raise ValueError(f"could not read any image in {input_dir}")
        h0, w0 = probe.shape[:2]
        canvas_w = width or int(round(w0 * scale))
        canvas_h = height or int(round(h0 * scale))

    template = canonical_template(canvas_w, canvas_h)
    aligned_images: list[np.ndarray] = []
    aligned_landmarks: list[np.ndarray] = []
    used_files: list[Path] = []
    skipped: list[Path] = []
    pose_filtered: list[Path] = []

    if on_progress:
        on_progress("detect", 0, len(files))

    landmarker = make_landmarker()
    try:
        for i, f in enumerate(files, 1):
            img = cv2.imread(str(f))
            if img is None:
                skipped.append(f)
            else:
                detected = detect_landmarks(img, landmarker)
                if detected is None:
                    skipped.append(f)
                else:
                    lm, matrix = detected
                    if front_facing_only and not is_front_facing(matrix, max_head_tilt_deg):
                        pose_filtered.append(f)
                    else:
                        try:
                            aligned_img, aligned_lm = align_to_canvas(
                                img, lm, template, canvas_w, canvas_h,
                            )
                            aligned_images.append(aligned_img)
                            aligned_landmarks.append(aligned_lm)
                            used_files.append(f)
                        except RuntimeError:
                            skipped.append(f)
            if on_progress:
                on_progress("detect", i, len(files))
    finally:
        landmarker.close()

    if len(aligned_images) < 2:
        msg = (
            f"need at least 2 images with detectable faces; got {len(aligned_images)} "
            f"after skipping {len(skipped)}"
        )
        if pose_filtered:
            msg += f" and pose-filtering {len(pose_filtered)}"
        raise RuntimeError(msg)

    if keep_aligned_dir:
        keep_aligned_dir = Path(keep_aligned_dir)
        keep_aligned_dir.mkdir(parents=True, exist_ok=True)
        for f, img in zip(used_files, aligned_images, strict=True):
            cv2.imwrite(str(keep_aligned_dir / f"c_{f.name}"), img)

    bnd = boundary_points(canvas_w, canvas_h)
    full_landmarks = [
        np.vstack([
            lm,
            halo_points(lm, halo_factor, canvas_w, canvas_h),
            bnd,
        ])
        for lm in aligned_landmarks
    ]
    mean_pts = np.mean(np.stack(full_landmarks), axis=0)
    triangles = delaunay_triangles(mean_pts, canvas_w, canvas_h)
    if on_progress:
        on_progress("triangulate", 1, 1)

    n_pairs = len(aligned_images) - 1
    if on_progress:
        on_progress("morph", 0, n_pairs)

    def all_frames() -> Iterable[np.ndarray]:
        for i in range(n_pairs):
            include_endpoint = (i == n_pairs - 1)
            label_a = used_files[i].stem
            label_b = used_files[i + 1].stem
            for j, frame in enumerate(morph_pair(
                aligned_images[i], aligned_images[i + 1],
                full_landmarks[i], full_landmarks[i + 1],
                triangles, frames_per_pair, include_endpoint,
            )):
                if overlay:
                    t = j / frames_per_pair
                    frame = draw_overlay(frame, label_b if t >= 0.5 else label_a)
                yield frame
            if on_progress:
                on_progress("morph", i + 1, n_pairs)

    encoder_name, encoder_args = pick_encoder(encoder)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encode_video(all_frames(), output_path, canvas_w, canvas_h, fps, encoder_args)

    total_frames = n_pairs * frames_per_pair + 1
    if on_progress:
        on_progress("done", 1, 1)

    return RenderResult(
        output_path=output_path,
        canvas_w=canvas_w, canvas_h=canvas_h,
        fps=fps, total_frames=total_frames,
        used_files=used_files, skipped_files=skipped,
        pose_filtered_files=pose_filtered,
        encoder=encoder_name,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_progress() -> ProgressCallback:
    """Translate on_progress events into tqdm bars + headers."""
    bars: dict[str, tqdm] = {}
    seen_stages: set[str] = set()

    headers = {
        "detect":      "[1/3] detecting landmarks",
        "triangulate": "[2/3] computing template + Delaunay triangulation",
        "morph":       "[3/3] morphing + encoding",
    }

    def cb(stage: str, current: int, total: int) -> None:
        if stage in headers and stage not in seen_stages:
            print(headers[stage])
            seen_stages.add(stage)
        if stage == "triangulate" or stage == "done":
            return
        if stage not in bars:
            bars[stage] = tqdm(total=total, desc=stage, leave=True)
        bars[stage].n = current
        bars[stage].refresh()
        if current == total:
            bars[stage].close()

    return cb


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
    ap.add_argument("--encoder", default=None,
                    help="Force a specific ffmpeg encoder (e.g. libx264, h264_nvenc). "
                         "Default: auto-detect, prefer hardware.")
    ap.add_argument("--front-facing-only", action="store_true",
                    help="Skip photos where the head is turned away from the camera. "
                         "Useful for cleaning up large archives.")
    ap.add_argument("--max-head-tilt-deg", default=20.0, type=float,
                    help="Front-facing tolerance in degrees (any axis). "
                         "20°≈default, 15°=strict, 30°=loose. "
                         "Only used with --front-facing-only.")
    ap.add_argument("--halo-factor", default=1.25, type=float,
                    help="Outward extrapolation of FACEMESH_FACE_OVAL anchors "
                         "for smoother face-edge morphing. 1.0 disables; "
                         "1.25 is the default; 1.5+ gets aggressive.")
    args = ap.parse_args()

    try:
        result = run_pipeline(
            args.input, args.video,
            width=args.width, height=args.height, scale=args.scale,
            frames_per_pair=args.frames_per_pair, fps=args.fps,
            overlay=not args.no_overlay,
            keep_aligned_dir=args.output if args.keep_aligned else None,
            encoder=args.encoder,
            front_facing_only=args.front_facing_only,
            max_head_tilt_deg=args.max_head_tilt_deg,
            halo_factor=args.halo_factor,
            on_progress=_cli_progress(),
        )
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if result.skipped_files:
        print(f"  skipped {len(result.skipped_files)} (no face / unreadable):")
        for p in result.skipped_files[:10]:
            print(f"    - {p.name}")
        if len(result.skipped_files) > 10:
            print(f"    ... and {len(result.skipped_files) - 10} more")

    if result.pose_filtered_files:
        print(f"  filtered {len(result.pose_filtered_files)} (not front-facing):")
        for p in result.pose_filtered_files[:10]:
            print(f"    - {p.name}")
        if len(result.pose_filtered_files) > 10:
            print(f"    ... and {len(result.pose_filtered_files) - 10} more")

    print(
        f"done: {result.output_path} "
        f"({result.total_frames} frames @ {result.fps} fps "
        f"= {result.duration_seconds:.1f}s) [{result.encoder}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
