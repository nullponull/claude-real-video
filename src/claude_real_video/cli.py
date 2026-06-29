"""Command-line interface for claude-real-video."""
import argparse
import sys

from .core import process


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="claude-real-video",
        description="Let Claude (or any LLM) actually watch a video: scene-aware, "
                    "deduplicated frames + a transcript, from a URL or a local file.",
    )
    ap.add_argument("source", help="Video URL (YouTube, Instagram, ...) or a local file path")
    ap.add_argument("-o", "--out", default="crv-out", help="Output directory (default: ./crv-out)")
    ap.add_argument("--scene", type=float, default=0.30,
                    help="Scene-change sensitivity 0-1, lower = more frames (default: 0.30)")
    ap.add_argument("--fps-floor", type=float, default=1.0,
                    help="Guarantee at least one frame every N seconds (default: 1.0)")
    ap.add_argument("--max-frames", type=int, default=150, help="Cap total frames (default: 150)")
    ap.add_argument("--lang", default="auto", help="Whisper language, e.g. en / zh / auto (default: auto)")
    ap.add_argument("--cookies", default=None,
                    help="Netscape cookie file for sites that need login (your own, authorised use only)")
    ap.add_argument("--no-transcribe", action="store_true", help="Skip audio transcription")
    ap.add_argument("--dedup-threshold", type=int, default=8,
                    help="Frame dedup sensitivity, higher = fewer frames kept (default: 8)")
    args = ap.parse_args()

    try:
        r = process(
            args.source, args.out,
            scene=args.scene, fps_floor=args.fps_floor, max_frames=args.max_frames,
            lang=args.lang, cookies=args.cookies,
            do_transcribe=not args.no_transcribe, dedup_threshold=args.dedup_threshold,
        )
    except Exception as e:  # noqa: BLE001 — surface a clean message to the user
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n✓ Done → {r.out_dir}")
    print(f"  {r.frame_count} frames  (scene {r.scene_frames} + floor, deduped)  in {r.frames_dir}")
    print(f"  manifest:   {r.manifest_path}")
    if r.transcript_path:
        print(f"  transcript: {r.transcript_path}")
    else:
        print("  transcript: skipped (install the whisper CLI / `pip install openai-whisper` to enable)")


if __name__ == "__main__":
    main()
