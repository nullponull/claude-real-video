"""Core pipeline: fetch a video (URL or file), extract scene-aware + deduplicated
frames, optionally transcribe audio, and write a manifest an LLM can read."""
from __future__ import annotations
import glob
import os
import shutil
import subprocess
from dataclasses import dataclass, field


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _have(tool: str) -> bool:
    return shutil.which(tool) is not None


@dataclass
class Result:
    out_dir: str
    video: str
    duration: int
    frames_dir: str
    frame_count: int
    scene_frames: int
    transcript_path: str | None
    manifest_path: str


def fetch_video(src: str, out_dir: str, cookies: str | None = None) -> str:
    """Download via yt-dlp (URL) or copy a local file. cookies is an optional
    Netscape-format cookie file for sites that require login (your own,
    authorised use only)."""
    dest = os.path.join(out_dir, "source.mp4")
    if src.startswith(("http://", "https://")):
        if not _have("yt-dlp"):
            raise RuntimeError("yt-dlp not found. Install it: pip install yt-dlp")
        base = ["yt-dlp", src, "-o", dest, "--merge-output-format", "mp4", "--no-warnings", "-q"]
        _run(base)
        if not os.path.exists(dest) and cookies:
            _run(base + ["--cookies", cookies])
        if not os.path.exists(dest):
            # yt-dlp may have written a different extension
            hits = sorted(glob.glob(os.path.join(out_dir, "source.*")))
            if hits:
                dest = hits[0]
        if not os.path.exists(dest):
            raise RuntimeError("Download failed (private video? try --cookies your_cookies.txt)")
    else:
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        shutil.copy(src, dest)
    return dest


def _duration(video: str) -> int:
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "default=nw=1:nk=1", video])
    try:
        return int(float(r.stdout.strip()))
    except (ValueError, AttributeError):
        return 0


def extract_frames(video: str, frames_dir: str, scene: float, fps_floor: float,
                   max_frames: int) -> tuple[int, int]:
    """Scene-change frames (every visual change) + a density floor (so dynamic
    videos are never under-sampled). Returns (scene_count, total_before_dedup)."""
    os.makedirs(frames_dir, exist_ok=True)
    _run(["ffmpeg", "-i", video, "-vf", f"select='gt(scene,{scene})',scale=640:-1",
          "-vsync", "vfr", os.path.join(frames_dir, "scene_%03d.jpg"),
          "-hide_banner", "-loglevel", "error"])
    scene_n = len(glob.glob(os.path.join(frames_dir, "scene_*.jpg")))
    _run(["ffmpeg", "-i", video, "-vf", f"fps=1/{fps_floor},scale=640:-1",
          os.path.join(frames_dir, "floor_%03d.jpg"),
          "-hide_banner", "-loglevel", "error"])
    total = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if total > max_frames:
        floors = sorted(glob.glob(os.path.join(frames_dir, "floor_*.jpg")))
        for i, f in enumerate(floors):
            if i % 3 != 0:
                os.remove(f)
    return scene_n, len(glob.glob(os.path.join(frames_dir, "*.jpg")))


def dedup_frames(frames_dir: str, threshold: int = 8) -> int:
    """Drop near-identical consecutive frames via average-hash. This is the key
    win over fixed-budget extractors: a static slide collapses to one frame."""
    try:
        from PIL import Image
    except ImportError:
        return len(glob.glob(os.path.join(frames_dir, "*.jpg")))

    def ahash(path: str, size: int = 12) -> list[int]:
        im = Image.open(path).convert("L").resize((size, size))
        px = list(im.getdata())
        avg = sum(px) / len(px)
        return [1 if v > avg else 0 for v in px]

    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    kept: list[str] = []
    last: list[int] | None = None
    for f in frames:
        h = ahash(f)
        if last is None or sum(a != b for a, b in zip(h, last)) > threshold:
            kept.append(f)
            last = h
        else:
            os.remove(f)
    for i, f in enumerate(sorted(kept), 1):
        os.rename(f, os.path.join(frames_dir, f"tmp_{i:03d}.jpg"))
    for f in sorted(os.listdir(frames_dir)):
        if f.startswith("tmp_"):
            os.rename(os.path.join(frames_dir, f), os.path.join(frames_dir, "frame_" + f[4:]))
    return len(kept)


def transcribe(video: str, out_dir: str, lang: str | None) -> str | None:
    """Optional: extract audio + run Whisper if the `whisper` CLI is installed."""
    if not _have("whisper"):
        return None
    wav = os.path.join(out_dir, "audio.wav")
    _run(["ffmpeg", "-i", video, "-vn", "-ar", "16000", "-ac", "1", wav,
          "-hide_banner", "-loglevel", "error"])
    if not os.path.exists(wav):
        return None
    cmd = ["whisper", wav, "--model", "base", "--output_format", "txt", "--output_dir", out_dir]
    if lang and lang != "auto":
        cmd += ["--language", lang]
    _run(cmd)
    src = os.path.join(out_dir, "audio.txt")
    dst = os.path.join(out_dir, "transcript.txt")
    if os.path.exists(src):
        os.replace(src, dst)
        return dst
    return None


def process(src: str, out_dir: str, *, scene: float = 0.30, fps_floor: float = 1.0,
            max_frames: int = 150, lang: str | None = "auto", cookies: str | None = None,
            do_transcribe: bool = True, dedup_threshold: int = 8) -> Result:
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    video = fetch_video(src, out_dir, cookies=cookies)
    dur = _duration(video)
    scene_n, _ = extract_frames(video, frames_dir, scene, fps_floor, max_frames)
    kept = dedup_frames(frames_dir, dedup_threshold)
    transcript = transcribe(video, out_dir, lang) if do_transcribe else None

    manifest = os.path.join(out_dir, "MANIFEST.txt")
    lines = [
        f"source: {src}",
        f"duration: {dur}s | frames: {kept} (scene {scene_n} + density floor, deduped)",
        f"frames dir: {frames_dir}",
        f"transcript: {transcript or '(none — install openai-whisper to enable)'}",
        "--- transcript ---",
    ]
    if transcript and os.path.exists(transcript):
        lines.append(open(transcript, encoding="utf-8").read().strip())
    open(manifest, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    return Result(out_dir=out_dir, video=video, duration=dur, frames_dir=frames_dir,
                  frame_count=kept, scene_frames=scene_n,
                  transcript_path=transcript, manifest_path=manifest)
