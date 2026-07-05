"""Core pipeline: fetch a video (URL or file), extract scene-aware + deduplicated
frames, optionally transcribe audio, and write a manifest an LLM can read."""
from __future__ import annotations
import glob
import os
import re
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
    extracted_frames: int
    transcript_path: str | None
    manifest_path: str
    transcript_note: str = ""
    audio_path: str | None = None
    report_path: str | None = None
    frame_times: dict = field(default_factory=dict)


def fetch_video(src: str, out_dir: str, cookies: str | None = None, cookies_from_browser: str | None = None) -> str:
    """Download via yt-dlp (URL) or copy a local file. cookies is an optional
    Netscape-format cookie file for sites that require login (your own,
    authorised use only)."""
    dest = os.path.join(out_dir, "source.mp4")
    if src.startswith(("http://", "https://")):
        if not _have("yt-dlp"):
            raise RuntimeError("yt-dlp not found. Install it: pip install yt-dlp")
        base = ["yt-dlp", src, "-o", dest, "--merge-output-format", "mp4", "--no-warnings", "-q"]
        _run(base)
        if not os.path.exists(dest) and cookies_from_browser:
            _run(base + ["--cookies-from-browser", cookies_from_browser])
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


def _has_audio(video: str) -> bool:
    """True if the file carries at least one audio stream."""
    r = _run(["ffprobe", "-v", "error", "-select_streams", "a",
              "-show_entries", "stream=codec_type", "-of", "csv=p=0", video])
    return bool(r.stdout.strip())


def _fps(video: str) -> float:
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=avg_frame_rate", "-of", "default=nw=1:nk=1", video])
    try:
        num, den = r.stdout.strip().split("/")
        return float(num) / float(den) if float(den) else 25.0
    except (ValueError, ZeroDivisionError, AttributeError):
        return 25.0


def _fmt_ts(t: float | None) -> str:
    """Seconds -> MM:SS (or H:MM:SS past an hour). None -> --:-- so a frame whose
    timestamp we couldn't recover still lines up in the manifest columns."""
    if t is None:
        return "--:--"
    t = int(round(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _parse_ts(ts: str) -> float:
    """An srt/vtt timecode ('00:01:02,500' or '01:02.5') -> seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = "0", parts[0], parts[1]
        else:
            h, m, s = "0", "0", parts[0]
        return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        return 0.0


def extract_frames(video: str, frames_dir: str, scene: float, fps_floor: float) -> dict[str, float | None]:
    """One chronological pass: every scene change OR one frame per `fps_floor`
    seconds, whichever comes first. A single select filter keeps the frames in
    time order, so dedup compares true neighbours (two passes used to interleave
    scene_/floor_ files out of order).

    The `showinfo` filter prints one `pts_time:` line per frame that survives the
    select, in output order — so its Nth value is raw_0000N.jpg's real position
    in the video. That is what lets the manifest say *when* each kept frame is,
    and align a visual change with the words spoken at that moment. Returns a
    {raw_filename: timestamp_seconds} map (timestamp None if a build without
    showinfo left the counts out of step — we degrade rather than mis-align)."""
    os.makedirs(frames_dir, exist_ok=True)
    every_n = max(1, round(_fps(video) * fps_floor))
    r = _run(["ffmpeg", "-i", video,
              "-vf", f"select='gt(scene,{scene})+not(mod(n,{every_n}))',showinfo,scale=640:-1",
              "-vsync", "vfr", os.path.join(frames_dir, "raw_%05d.jpg"),
              "-hide_banner", "-loglevel", "info"])
    times = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", r.stderr)]
    raws = sorted(glob.glob(os.path.join(frames_dir, "raw_*.jpg")))
    if len(times) == len(raws):
        return {os.path.basename(f): t for f, t in zip(raws, times)}
    return {os.path.basename(f): None for f in raws}


def dedup_frames(frames_dir: str, threshold: float = 8, window: int = 4,
                 max_frames: int = 150,
                 dropped_dir: str | None = None,
                 times: dict[str, float | None] | None = None) -> tuple[int, list[dict], dict[str, float | None]]:
    """Drop near-duplicate frames by real pixel difference (downscaled grayscale,
    like videostil's pixelmatch approach — more faithful than a perceptual hash,
    which goes blind on flat colours and brightness-only changes) against a
    sliding window of the last `window` kept frames. The window also catches
    A-B-A alternation — a shot the model has already seen doesn't come back
    just because a different frame sat in between. `threshold` is the percent
    of pixels that must change for a frame to count as new.
    `times` maps each raw frame to its video timestamp; it rides through the
    dedup + rename so callers get {frame_NNN.jpg: seconds}.
    Returns (kept_count, per-frame records for the optional report, kept_times)."""
    times = times or {}
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    try:
        from PIL import Image
    except ImportError:
        return len(frames), [], {os.path.basename(f): times.get(os.path.basename(f)) for f in frames}

    def sig(path: str, size: int = 16) -> list[tuple[int, int, int]]:
        # RGB, not grayscale: hues with equal luma (a red→green cut) must not
        # look identical to the comparator
        return list(Image.open(path).convert("RGB").resize((size, size)).getdata())

    def pct_diff(a: list, b: list, tol: int = 25) -> float:
        changed = sum(max(abs(x[0] - y[0]), abs(x[1] - y[1]), abs(x[2] - y[2])) > tol
                      for x, y in zip(a, b))
        return 100.0 * changed / len(a)

    kept: list[str] = []
    recent: list[list[int]] = []  # signatures of the last `window` kept frames
    records: list[dict] = []
    for f in frames:
        h = sig(f)
        dist = min((pct_diff(h, k) for k in recent), default=None)
        if dist is None or dist > threshold:
            kept.append(f)
            recent.append(h)
            if len(recent) > window:
                recent.pop(0)
            records.append({"name": os.path.basename(f), "dist": dist, "kept": True,
                            "t": times.get(os.path.basename(f))})
        else:
            if dropped_dir:
                os.makedirs(dropped_dir, exist_ok=True)
                shutil.move(f, os.path.join(dropped_dir, os.path.basename(f)))
            else:
                os.remove(f)
            records.append({"name": os.path.basename(f), "dist": dist, "kept": False,
                            "t": times.get(os.path.basename(f))})

    # cap: thin uniformly *after* dedup so the survivors stay spread across the video
    if len(kept) > max_frames:
        step = len(kept) / max_frames
        keep_idx = {int(i * step) for i in range(max_frames)}
        for i, f in enumerate(list(kept)):
            if i not in keep_idx:
                kept.remove(f)
                os.remove(f)
                for rec in records:
                    if rec["name"] == os.path.basename(f):
                        rec["kept"] = False
                        rec["capped"] = True

    renames = {}
    for i, f in enumerate(sorted(kept), 1):
        renames[os.path.basename(f)] = f"frame_{i:03d}.jpg"
        os.rename(f, os.path.join(frames_dir, f"tmp_{i:03d}.jpg"))
    for f in sorted(os.listdir(frames_dir)):
        if f.startswith("tmp_"):
            os.rename(os.path.join(frames_dir, f), os.path.join(frames_dir, "frame_" + f[4:]))
    for rec in records:
        if rec["kept"]:
            rec["name"] = renames.get(rec["name"], rec["name"])
    kept_times = {frame_name: times.get(raw_name)
                  for raw_name, frame_name in renames.items()}
    return len(kept), records, kept_times


def write_report(out_dir: str, records: list[dict], threshold: float, window: int) -> str:
    """Self-contained report.html showing every extracted frame — kept or
    dropped — with its hash distance, so you can eyeball whether the threshold
    is too tight or too loose (videostil's Analysis Viewer, minus the server)."""
    kept_n = sum(1 for r in records if r["kept"])
    rows = []
    for r in records:
        src = f"frames/{r['name']}" if r["kept"] else f"dropped/{r['name']}"
        why = "capped" if r.get("capped") else ("kept" if r["kept"] else "dropped")
        dist = "first" if r["dist"] is None else f"{r['dist']:.1f}%"
        ts = _fmt_ts(r.get("t"))
        rows.append(
            f'<figure class="{why}"><img src="{src}" loading="lazy">'
            f'<figcaption>{r["name"]} · {ts}<br>dist {dist} · {why}</figcaption></figure>')
    html = f"""<!doctype html><meta charset="utf-8"><title>crv dedup report</title>
<style>
body{{font:14px system-ui;margin:20px;background:#111;color:#ddd}}
.grid{{display:flex;flex-wrap:wrap;gap:10px}}
figure{{margin:0;width:200px}}img{{width:100%;border-radius:4px}}
figcaption{{font-size:11px;color:#999;padding:2px 0}}
.dropped img{{opacity:.35;outline:2px solid #a33}}
.capped img{{opacity:.35;outline:2px solid #a80}}
.kept img{{outline:2px solid #3a6}}
</style>
<h2>crv dedup report</h2>
<p>threshold {threshold} · window {window} · kept {kept_n} / {len(records)}
(green kept · red duplicate · orange removed by --max-frames cap)</p>
<div class="grid">{''.join(rows)}</div>
"""
    path = os.path.join(out_dir, "report.html")
    open(path, "w", encoding="utf-8").write(html)
    return path


def _has_subtitle_stream(video: str) -> bool:
    r = _run(["ffprobe", "-v", "error", "-select_streams", "s",
              "-show_entries", "stream=index", "-of", "csv=p=0", video])
    return bool(r.stdout.strip())


def _subs_to_cues(sub_path: str) -> list[tuple[float, str]]:
    """Parse an .srt/.vtt subtitle file into [(start_seconds, text), ...],
    keeping the start timecode of every cue. The old version threw the timecodes
    away, which is exactly why a spoken line could no longer be tied to the frame
    it belongs with; here we keep them so the manifest can build a real timeline."""
    try:
        raw = open(sub_path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return []
    raw = raw.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    cues: list[tuple[float, str]] = []
    for block in re.split(r"\n\s*\n", raw):
        start: float | None = None
        text: list[str] = []
        for ln in block.splitlines():
            s = ln.strip()
            if not s or s == "WEBVTT" or s.isdigit():
                continue
            if "-->" in s:
                start = _parse_ts(s.split("-->")[0])
                continue
            s = re.sub(r"<[^>]+>", "", s)  # strip vtt inline tags like <v ->
            if s:
                text.append(s)
        joined = " ".join(text).strip()
        if start is not None and joined:
            cues.append((start, joined))
    return cues


def _write_transcript(cues: list[tuple[float, str]], out_txt: str) -> str | None:
    """Write cues to transcript.txt as '[MM:SS] text' — human-readable and,
    unlike the old plain dump, still carrying when each line was spoken."""
    if not cues:
        return None
    with open(out_txt, "w", encoding="utf-8") as fh:
        for t, text in cues:
            fh.write(f"[{_fmt_ts(t)}] {text}\n")
    return out_txt


def existing_subtitles(src: str, video: str, out_dir: str) -> tuple[str | None, list[tuple[float, str]]]:
    """Use subtitles the video already ships with, instead of re-transcribing.
    Checks (1) a sidecar .srt/.vtt next to a local source file, then
    (2) an embedded subtitle stream. Returns (transcript_path, cues) — cues carry
    the timecodes so the caller can build a timeline; ("", []) / (None, []) if none.
    This is faster and more accurate than Whisper when captions already exist."""
    dst = os.path.join(out_dir, "transcript.txt")
    # 1) sidecar file next to the original source (local files only)
    if not src.startswith(("http://", "https://")):
        base = os.path.splitext(src)[0]
        for ext in (".srt", ".vtt"):
            cand = base + ext
            if os.path.exists(cand):
                cues = _subs_to_cues(cand)
                if _write_transcript(cues, dst):
                    return dst, cues
    # 2) embedded subtitle stream
    if _has_subtitle_stream(video):
        raw = os.path.join(out_dir, "_embedded.srt")
        _run(["ffmpeg", "-y", "-i", video, "-map", "0:s:0", raw,
              "-hide_banner", "-loglevel", "error"])
        if os.path.exists(raw):
            cues = _subs_to_cues(raw)
            try:
                os.remove(raw)
            except OSError:
                pass
            if _write_transcript(cues, dst):
                return dst, cues
    return None, []


def extract_full_audio(video: str, out_dir: str) -> str | None:
    """Save the complete original soundtrack (music + speech + effects) so an
    audio-capable model can actually *hear* the video — not just read the words.
    Copies the stream losslessly when the codec allows, else re-encodes to AAC."""
    if not _has_audio(video):
        return None
    dst = os.path.join(out_dir, "audio.m4a")
    # try a lossless stream copy first (works for AAC/ALAC sources)
    _run(["ffmpeg", "-y", "-i", video, "-vn", "-c:a", "copy", dst,
          "-hide_banner", "-loglevel", "error"])
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    # fallback: re-encode (e.g. opus/vorbis sources) at a high bitrate
    _run(["ffmpeg", "-y", "-i", video, "-vn", "-c:a", "aac", "-b:a", "192k", dst,
          "-hide_banner", "-loglevel", "error"])
    return dst if os.path.exists(dst) and os.path.getsize(dst) > 0 else None


def transcribe(video: str, out_dir: str, lang: str | None, model: str = "base") -> tuple[str | None, list[tuple[float, str]]]:
    """Optional: extract audio + run Whisper if the `whisper` CLI is installed.
    Asks Whisper for .srt (timestamped) instead of .txt, so the words keep the
    times that let them line up with the frames. Returns (transcript_path, cues)."""
    if not _have("whisper"):
        return None, []
    wav = os.path.join(out_dir, "audio.wav")
    _run(["ffmpeg", "-i", video, "-vn", "-ar", "16000", "-ac", "1", wav,
          "-hide_banner", "-loglevel", "error"])
    if not os.path.exists(wav):
        return None, []
    cmd = ["whisper", wav, "--model", model, "--output_format", "srt", "--output_dir", out_dir]
    if lang and lang != "auto":
        cmd += ["--language", lang]
    _run(cmd)
    srt = os.path.join(out_dir, "audio.srt")
    dst = os.path.join(out_dir, "transcript.txt")
    if os.path.exists(srt):
        cues = _subs_to_cues(srt)
        if _write_transcript(cues, dst):
            return dst, cues
    return None, []


def make_grids(frames_dir: str, out_dir: str, cols: int = 3, rows: int = 3,
               cell_width: int = 480, times: dict | None = None) -> list[str]:
    """Tile the kept frames, in order, into contact-sheet grids. A model reading
    consecutive frames side by side in one image follows motion and progression
    far better than the same frames seen one at a time. Each cell is labelled with
    its timestamp too, so the model can cite when a moment happens straight off
    the sheet — no cross-referencing the manifest."""
    from PIL import Image, ImageDraw
    times = times or {}
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frames:
        return []
    grids_dir = os.path.join(out_dir, "grids")
    os.makedirs(grids_dir, exist_ok=True)
    per = cols * rows
    sheets = []
    label_h = 22
    for gi in range(0, len(frames), per):
        batch = frames[gi:gi + per]
        first = Image.open(batch[0])
        cw = cell_width
        ch = int(first.height * cw / first.width) + label_h
        sheet = Image.new("RGB", (cols * cw, rows * ch), "black")
        draw = ImageDraw.Draw(sheet)
        for i, f in enumerate(batch):
            im = Image.open(f)
            im = im.resize((cw, ch - label_h))
            x, y = (i % cols) * cw, (i // cols) * ch
            sheet.paste(im, (x, y + label_h))
            name = os.path.basename(f)
            label = f"{name}   {_fmt_ts(times.get(name))}" if name in times else name
            draw.text((x + 6, y + 4), label, fill="white")
        dest = os.path.join(grids_dir, f"grid_{gi // per + 1:02d}.jpg")
        sheet.save(dest, quality=85)
        sheets.append(dest)
    return sheets


def save_to_kb(kb_dir: str, manifest_path: str, src: str) -> str:
    """Copy the analysis into a knowledge-base folder as a dated markdown note,
    so it lives next to the user's other notes instead of dying in ./crv-out."""
    import datetime, re as _re
    os.makedirs(kb_dir, exist_ok=True)
    slug = _re.sub(r"[^A-Za-z0-9一-鿿]+", "-", os.path.basename(src.rstrip("/")))[:60].strip("-") or "video"
    dest = os.path.join(kb_dir, f"{datetime.date.today().isoformat()}-{slug}.md")
    body = open(manifest_path, encoding="utf-8").read()
    open(dest, "w", encoding="utf-8").write(f"# Video analysis — {src}\n\n```\n{body}\n```\n")
    return dest


def process(src: str, out_dir: str, *, scene: float = 0.30, fps_floor: float = 1.0,
            max_frames: int = 150, lang: str | None = "auto", cookies: str | None = None,
            do_transcribe: bool = True, dedup_threshold: float = 8, dedup_window: int = 4,
            keep_audio: bool = False, report: bool = False, why: str | None = None, whisper_model: str = "base", cookies_from_browser: str | None = None) -> Result:
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    video = fetch_video(src, out_dir, cookies=cookies, cookies_from_browser=cookies_from_browser)
    dur = _duration(video)
    frame_times = extract_frames(video, frames_dir, scene, fps_floor)
    extracted = len(frame_times)
    kept, records, kept_times = dedup_frames(
        frames_dir, dedup_threshold, dedup_window, max_frames,
        dropped_dir=os.path.join(out_dir, "dropped") if report else None,
        times=frame_times)
    report_path = write_report(out_dir, records, dedup_threshold, dedup_window) if report else None

    # Text for the LLM: prefer subtitles the video already has (faster + more
    # accurate); only fall back to Whisper when there are none. Be honest about
    # *why* there's no transcript — a silent video is not a missing whisper install.
    transcript = None
    cues: list[tuple[float, str]] = []
    if not do_transcribe:
        note = "(skipped: --no-transcribe)"
    else:
        transcript, cues = existing_subtitles(src, video, out_dir)
        if transcript:
            note = f"{transcript} (from the video's own subtitles)"
        elif not _have("whisper"):
            note = "(none — no existing subtitles; install whisper to transcribe: pip install openai-whisper)"
        elif not _has_audio(video):
            note = "(none — this video has no subtitles and no audio track)"
        else:
            transcript, cues = transcribe(video, out_dir, lang, model=whisper_model)
            note = f"{transcript} (transcribed by whisper)" if transcript else "(none — transcription failed)"

    # Optionally keep the full original soundtrack (music + speech + effects) for
    # models that can listen to audio directly — the transcript only has the words.
    audio_path = extract_full_audio(video, out_dir) if keep_audio else None

    manifest = os.path.join(out_dir, "MANIFEST.txt")
    lines = []
    if why:
        # The reader's job, stated up front: focus the analysis instead of a
        # wandering summary. This line is for the LLM that reads this manifest.
        lines += [f"viewing intent: {why}",
                  "(reader: analyse the frames and transcript with this intent as the lens — "
                  "surface what serves it first, skip what doesn't)", ""]
    lines += [
        f"source: {src}",
        f"duration: {dur}s | frames: {kept} (scene-change + density floor, "
        f"deduped from {extracted} extracted)",
        f"frames dir: {frames_dir}",
        f"transcript: {note}",
    ]
    if keep_audio:
        lines.append(f"audio: {audio_path or '(none — this video has no audio track)'}")

    # Unified timeline: every kept frame and every spoken cue on ONE time axis,
    # so the reader can see which words belong with which visual change. This is
    # the whole point of the fork — order-only lists left the model to guess the
    # alignment. Frames sort before speech at an equal timestamp; anything whose
    # time we couldn't recover sinks to the bottom (still listed, never dropped).
    events: list[tuple[float | None, int, str]] = []
    for name, t in kept_times.items():
        events.append((t, 0, f"frame  {name}"))
    for t, text in cues:
        events.append((t, 1, f"speech {text}"))
    events.sort(key=lambda e: (e[0] is None, e[0] if e[0] is not None else 0.0, e[1]))
    lines.append("--- timeline (video changes <-> speech, by timestamp) ---")
    for t, _kind, payload in events:
        lines.append(f"[{_fmt_ts(t)}] {payload}")

    lines.append("--- transcript ---")
    if transcript and os.path.exists(transcript):
        lines.append(open(transcript, encoding="utf-8").read().strip())
    open(manifest, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    return Result(out_dir=out_dir, video=video, duration=dur, frames_dir=frames_dir,
                  frame_count=kept, extracted_frames=extracted,
                  transcript_path=transcript, manifest_path=manifest,
                  transcript_note=note, audio_path=audio_path, report_path=report_path,
                  frame_times=kept_times)
