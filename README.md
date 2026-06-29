# claude-real-video

**Let Claude (or any LLM) actually watch a video.**

Point it at a URL or a local file. It pulls the video, extracts the frames that
*actually matter* (every scene change, not a fixed quota), throws away the
near-duplicates, transcribes the audio, and hands you a clean folder an LLM can
read.

```bash
crv "https://www.youtube.com/watch?v=..."
# → crv-out/frames/*.jpg  +  crv-out/transcript.txt  +  crv-out/MANIFEST.txt
```

Then drop the frames + `MANIFEST.txt` into Claude (or paste them) and ask away.

---

## Why not just sample frames?

Most "let an LLM watch a video" scripts grab a **fixed number of frames**
(e.g. 30s → 30 frames). That over-samples a static screencast and under-samples
a fast-cut reel. claude-real-video is smarter:

| | fixed-quota extractors | **claude-real-video** |
|---|---|---|
| Input | local file only | **URL (yt-dlp) or file** |
| Frame selection | every N seconds | **scene-change detection** + density floor |
| Static slide (10 min) | ~100 near-identical frames | **collapses to 1** (dedup) |
| Audio | sometimes | Whisper transcript w/ language detect |

You end up feeding the model *fewer, more meaningful* frames — cheaper context,
better understanding.

---

## Install

```bash
pip install claude-real-video          # core (frames + dedup)
pip install "claude-real-video[whisper]"   # + audio transcription
```

System requirements (not pip-installable):

- **ffmpeg / ffprobe** — `brew install ffmpeg` (macOS) / `apt install ffmpeg` (Linux)
- Transcription uses the `whisper` CLI (installed by the `[whisper]` extra, or `pip install openai-whisper`).

---

## Usage

```bash
# A YouTube / Instagram / TikTok / ... link
crv "https://www.instagram.com/reel/XXXX/"

# A local file, English transcript, output to ./out
crv lecture.mp4 -o out --lang en

# Frames only, no transcription
crv clip.mp4 --no-transcribe

# A login-gated video (your own / authorised use): pass a Netscape cookie file
crv "https://..." --cookies cookies.txt
```

### Options

| flag | default | meaning |
|---|---|---|
| `-o, --out` | `crv-out` | output directory |
| `--scene` | `0.30` | scene-change sensitivity (lower = more frames) |
| `--fps-floor` | `1.0` | at least one frame every N seconds |
| `--max-frames` | `150` | hard cap on total frames |
| `--lang` | `auto` | Whisper language (`en`, `zh`, `auto`, ...) |
| `--dedup-threshold` | `8` | higher = fewer frames kept |
| `--no-transcribe` | off | skip audio |
| `--cookies` | – | Netscape cookie file for login-gated sources |

---

## Use it from Python

```python
from claude_real_video import process

r = process("https://youtu.be/...", "out", lang="en")
print(r.frame_count, r.transcript_path)
```

---

## How it works

1. **Fetch** — `yt-dlp` for URLs (optional cookies), or copy a local file.
2. **Scene frames** — `ffmpeg select='gt(scene,…)'` grabs each visual change.
3. **Density floor** — also samples every `--fps-floor` seconds so nothing slips through.
4. **Dedup** — average-hash drops near-identical frames (a static screen → one frame).
5. **Transcribe** — extract audio + Whisper (optional).
6. **Manifest** — `MANIFEST.txt` summarises everything for the model.

---

## Notes

- Only download content you have the right to. The `--cookies` option is for
  your own, authorised access — don't ship credentials in a repo.
- Output is deterministic-ish; re-running overwrites the output dir.

## License

MIT
