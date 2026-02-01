#!/usr/bin/env python3
"""
HoopTriage — AI-powered basketball clip triage tool.
Sort, score, and organise tournament footage fast.
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Run: pip install numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".mts", ".webm"}
CONTACT_SHEET_FRAMES = 4
FRAME_WIDTH = 480  # px per frame in contact sheet


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def check_ffmpeg():
    """Ensure ffmpeg and ffprobe are available."""
    for cmd in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([cmd, "-version"], capture_output=True, check=True)
        except FileNotFoundError:
            print(f"Error: {cmd} not found. Install it first (brew install ffmpeg).")
            sys.exit(1)


def get_duration(clip_path: str) -> float:
    """Return clip duration in seconds."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", clip_path],
        capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"].get("duration", 0))


def extract_audio_pcm(clip_path: str, tmp_dir: str) -> str | None:
    """Extract audio as 16-bit mono WAV. Returns path or None on failure."""
    out_path = os.path.join(tmp_dir, "audio.wav")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", clip_path,
            "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            "-f", "wav", out_path,
        ],
        capture_output=True,
    )
    if result.returncode != 0 or not os.path.exists(out_path):
        return None
    return out_path


def extract_frames(clip_path: str, output_dir: str, num_frames: int = CONTACT_SHEET_FRAMES) -> list[str]:
    """Extract evenly-spaced frames from a clip. Returns list of frame paths."""
    duration = get_duration(clip_path)
    if duration <= 0:
        return []

    frame_paths = []
    for i in range(num_frames):
        # Spread frames across 10%-90% of the clip to avoid black frames
        t = duration * (0.1 + 0.8 * i / max(num_frames - 1, 1))
        out_path = os.path.join(output_dir, f"frame_{i:02d}.jpg")
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", clip_path,
                "-vframes", "1", "-q:v", "3",
                "-vf", f"scale={FRAME_WIDTH}:-1",
                out_path,
            ],
            capture_output=True,
        )
        if os.path.exists(out_path):
            frame_paths.append(out_path)

    return frame_paths


def build_contact_sheet(frame_paths: list[str], output_path: str):
    """Stitch frames horizontally into a single contact sheet image."""
    if not frame_paths:
        return
    inputs = []
    for fp in frame_paths:
        inputs.extend(["-i", fp])

    filter_parts = []
    for i in range(len(frame_paths)):
        filter_parts.append(f"[{i}]scale={FRAME_WIDTH}:-1:force_original_aspect_ratio=decrease,pad={FRAME_WIDTH}:ih:(ow-iw)/2[f{i}];")

    hstack = "".join(f"[f{i}]" for i in range(len(frame_paths)))
    hstack += f"hstack=inputs={len(frame_paths)}"

    filter_str = "".join(filter_parts) + hstack

    subprocess.run(
        ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_str, "-q:v", "3", output_path],
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------

def analyse_audio(wav_path: str) -> dict:
    """Analyse a WAV file and return audio metrics."""
    with wave.open(wav_path, "rb") as wf:
        n_frames = wf.getnframes()
        if n_frames == 0:
            return {"rms": 0, "peak": 0, "dynamic_range": 0}
        raw = wf.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0

    rms = float(np.sqrt(np.mean(samples ** 2)))
    peak = float(np.max(np.abs(samples)))

    # Compute RMS in windows to measure dynamic range
    window_size = 1600  # 100ms at 16kHz
    if len(samples) > window_size:
        n_windows = len(samples) // window_size
        windowed = samples[:n_windows * window_size].reshape(n_windows, window_size)
        window_rms = np.sqrt(np.mean(windowed ** 2, axis=1))
        dynamic_range = float(np.max(window_rms) - np.min(window_rms))
        peak_window_rms = float(np.max(window_rms))
    else:
        dynamic_range = 0
        peak_window_rms = rms

    return {
        "rms": rms,
        "peak": peak,
        "dynamic_range": dynamic_range,
        "peak_window_rms": peak_window_rms,
    }


def score_excitement(metrics: dict) -> int:
    """Convert audio metrics to a 1-5 excitement score."""
    # Combine RMS, peak window RMS, and dynamic range
    # Higher values in all = more exciting (crowd reactions, whistles, cheering)
    combined = (
        metrics.get("peak_window_rms", 0) * 0.5
        + metrics.get("dynamic_range", 0) * 0.3
        + metrics.get("rms", 0) * 0.2
    )

    # Map to 1-5 scale (thresholds tuned for typical basketball gym audio)
    if combined > 0.25:
        return 5
    elif combined > 0.15:
        return 4
    elif combined > 0.08:
        return 3
    elif combined > 0.03:
        return 2
    else:
        return 1


# ---------------------------------------------------------------------------
# Jersey / team colour detection
# ---------------------------------------------------------------------------

def detect_dominant_colours(frame_path: str) -> list[tuple[int, int, int]]:
    """Detect dominant non-court colours from a frame using simple histogram analysis."""
    # Extract a small version of the frame as raw RGB
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", frame_path,
            "-vf", "scale=64:48",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        return []

    pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 3)

    # Filter out court-like colours (browns, tans, greys)
    # Keep saturated colours that are likely jerseys
    # Convert to simple saturation metric
    max_c = pixels.max(axis=1).astype(np.float64)
    min_c = pixels.min(axis=1).astype(np.float64)
    saturation = np.where(max_c > 0, (max_c - min_c) / max_c, 0)

    # Keep pixels with decent saturation (coloured jerseys)
    mask = saturation > 0.3
    coloured = pixels[mask]

    if len(coloured) < 10:
        return []

    # Simple quantisation: reduce to 8 levels per channel
    quantised = (coloured // 32) * 32 + 16

    # Find most common colours
    unique, counts = np.unique(quantised, axis=0, return_counts=True)
    top_idx = np.argsort(counts)[::-1][:3]

    return [tuple(int(c) for c in unique[i]) for i in top_idx]


def colour_name(rgb: tuple[int, int, int]) -> str:
    """Convert RGB to a rough colour name."""
    r, g, b = rgb
    if r > 180 and g < 100 and b < 100:
        return "red"
    if r < 100 and g > 150 and b < 100:
        return "green"
    if r < 100 and g < 100 and b > 150:
        return "blue"
    if r > 180 and g > 180 and b < 100:
        return "yellow"
    if r > 180 and g > 100 and b < 80:
        return "orange"
    if r > 100 and g < 80 and b > 150:
        return "purple"
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r < 60 and g < 60 and b < 60:
        return "black"
    if r > 150 and g > 150 and b > 150:
        return "grey"
    return f"rgb({r},{g},{b})"


def detect_team_colour(clip_path: str, tmp_dir: str) -> str | None:
    """Detect the dominant jersey colour from a clip's middle frame."""
    duration = get_duration(clip_path)
    if duration <= 0:
        return None

    mid_frame = os.path.join(tmp_dir, "team_frame.jpg")
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", f"{duration * 0.5:.2f}", "-i", clip_path,
            "-vframes", "1", "-q:v", "5", "-vf", "scale=320:-1", mid_frame,
        ],
        capture_output=True,
    )

    if not os.path.exists(mid_frame):
        return None

    colours = detect_dominant_colours(mid_frame)
    if colours:
        return colour_name(colours[0])
    return None


# ---------------------------------------------------------------------------
# Process a single clip
# ---------------------------------------------------------------------------

def process_clip(clip_path: str, output_dir: str, detect_teams: bool = True) -> dict:
    """Process a single clip: audio analysis, contact sheet, optional team detection."""
    clip_name = os.path.basename(clip_path)
    clip_stem = Path(clip_path).stem
    clip_output_dir = os.path.join(output_dir, "clips", clip_stem)
    os.makedirs(clip_output_dir, exist_ok=True)

    result = {
        "filename": clip_name,
        "path": os.path.abspath(clip_path),
        "duration": 0,
        "score": 1,
        "audio_metrics": {},
        "team_colour": None,
        "contact_sheet": None,
    }

    try:
        result["duration"] = get_duration(clip_path)
    except Exception:
        return result

    # Audio analysis
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = extract_audio_pcm(clip_path, tmp_dir)
        if wav_path:
            metrics = analyse_audio(wav_path)
            result["audio_metrics"] = metrics
            result["score"] = score_excitement(metrics)

        # Team colour detection
        if detect_teams:
            result["team_colour"] = detect_team_colour(clip_path, tmp_dir)

    # Contact sheet
    with tempfile.TemporaryDirectory() as tmp_dir:
        frames = extract_frames(clip_path, tmp_dir)
        if frames:
            sheet_path = os.path.join(clip_output_dir, "contact_sheet.jpg")
            build_contact_sheet(frames, sheet_path)
            if os.path.exists(sheet_path):
                result["contact_sheet"] = os.path.relpath(sheet_path, output_dir)

    return result


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_report(clips: list[dict], output_dir: str):
    """Generate an HTML report for reviewing clips."""
    clips_sorted = sorted(clips, key=lambda c: c["score"], reverse=True)

    # Collect unique team colours
    team_colours = sorted(set(c["team_colour"] for c in clips if c["team_colour"]))

    score_colours = {
        5: "#22c55e",  # green
        4: "#84cc16",  # lime
        3: "#eab308",  # yellow
        2: "#f97316",  # orange
        1: "#ef4444",  # red
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HoopTriage Report — {len(clips)} clips</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }}
    h1 {{ font-size: 28px; margin-bottom: 8px; }}
    .subtitle {{ color: #94a3b8; margin-bottom: 24px; font-size: 14px; }}
    .filters {{ display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; align-items: center; }}
    .filter-btn {{ padding: 6px 16px; border-radius: 20px; border: 1px solid #334155; background: #1e293b; color: #e2e8f0; cursor: pointer; font-size: 13px; transition: all 0.15s; }}
    .filter-btn:hover {{ background: #334155; }}
    .filter-btn.active {{ background: #3b82f6; border-color: #3b82f6; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 16px; }}
    .clip {{ background: #1e293b; border-radius: 12px; overflow: hidden; transition: transform 0.15s; }}
    .clip:hover {{ transform: translateY(-2px); }}
    .clip.hidden {{ display: none; }}
    .contact-sheet {{ width: 100%; display: block; }}
    .contact-sheet img {{ width: 100%; display: block; }}
    .clip-info {{ padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; }}
    .clip-name {{ font-size: 13px; font-weight: 500; word-break: break-all; flex: 1; margin-right: 12px; }}
    .clip-meta {{ display: flex; gap: 8px; align-items: center; flex-shrink: 0; }}
    .score-badge {{ display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; border-radius: 8px; font-weight: 700; font-size: 16px; color: #0f172a; }}
    .duration {{ font-size: 12px; color: #94a3b8; }}
    .team-dot {{ width: 14px; height: 14px; border-radius: 50%; border: 2px solid #334155; flex-shrink: 0; }}
    .stars {{ font-size: 14px; letter-spacing: 1px; }}
    .no-sheet {{ padding: 40px; text-align: center; color: #475569; font-size: 13px; background: #0f172a; }}
    video {{ width: 100%; max-height: 360px; background: #000; }}
    .play-btn {{ font-size: 12px; padding: 4px 12px; border-radius: 6px; background: #334155; color: #e2e8f0; border: none; cursor: pointer; }}
    .play-btn:hover {{ background: #475569; }}
    .summary {{ display: flex; gap: 24px; margin-bottom: 24px; flex-wrap: wrap; }}
    .stat {{ background: #1e293b; padding: 16px 20px; border-radius: 10px; }}
    .stat-value {{ font-size: 24px; font-weight: 700; }}
    .stat-label {{ font-size: 12px; color: #94a3b8; margin-top: 2px; }}
</style>
</head>
<body>
<h1>🏀 HoopTriage Report</h1>
<p class="subtitle">{len(clips)} clips analysed</p>

<div class="summary">
    <div class="stat">
        <div class="stat-value">{len(clips)}</div>
        <div class="stat-label">Total clips</div>
    </div>
    <div class="stat">
        <div class="stat-value">{len([c for c in clips if c['score'] >= 4])}</div>
        <div class="stat-label">Hot clips (4-5)</div>
    </div>
    <div class="stat">
        <div class="stat-value">{len([c for c in clips if c['score'] <= 2])}</div>
        <div class="stat-label">Likely skip (1-2)</div>
    </div>
    <div class="stat">
        <div class="stat-value">{sum(c['duration'] for c in clips) / 60:.0f}m</div>
        <div class="stat-label">Total footage</div>
    </div>
</div>

<div class="filters">
    <span style="color:#94a3b8;font-size:13px;">Score:</span>
    <button class="filter-btn active" onclick="filterScore(0)">All</button>
    {"".join(f'<button class="filter-btn" onclick="filterScore({s})" style="border-color:{score_colours[s]}50">{s}★</button>' for s in [5,4,3,2,1])}
    {"".join(f'''
    <span style="color:#94a3b8;font-size:13px;margin-left:12px;">Team:</span>
    <button class="filter-btn active" onclick="filterTeam(null)">All</button>
    ''' + "".join(f'<button class="filter-btn" onclick="filterTeam(&apos;{tc}&apos;)"><span class="team-dot" style="background:{tc};display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;"></span>{tc}</button>' for tc in team_colours)) if team_colours else ""}
</div>

<div class="grid" id="clip-grid">
"""

    for clip in clips_sorted:
        score = clip["score"]
        score_bg = score_colours.get(score, "#64748b")
        stars = "★" * score + "☆" * (5 - score)
        duration_str = f"{clip['duration']:.1f}s"
        team_attr = f'data-team="{clip["team_colour"]}"' if clip["team_colour"] else 'data-team=""'
        team_dot = f'<span class="team-dot" style="background:{clip["team_colour"]}"></span>' if clip["team_colour"] else ""

        if clip["contact_sheet"]:
            sheet_html = f'<div class="contact-sheet"><img src="{clip["contact_sheet"]}" loading="lazy" alt="{clip["filename"]}"></div>'
        else:
            sheet_html = '<div class="no-sheet">No frames extracted</div>'

        html += f"""
    <div class="clip" data-score="{score}" {team_attr}>
        {sheet_html}
        <div class="clip-info">
            <span class="clip-name">{clip["filename"]}</span>
            <div class="clip-meta">
                {team_dot}
                <span class="duration">{duration_str}</span>
                <span class="stars" style="color:{score_bg}">{stars}</span>
                <span class="score-badge" style="background:{score_bg}">{score}</span>
                <button class="play-btn" onclick="playClip(this, '{clip['path']}')">▶ Play</button>
            </div>
        </div>
    </div>
"""

    html += """
</div>

<script>
let currentScoreFilter = 0;
let currentTeamFilter = null;

function filterScore(score) {
    currentScoreFilter = score;
    applyFilters();
    document.querySelectorAll('.filters .filter-btn').forEach(b => {
        if (b.onclick?.toString().includes('filterScore'))
            b.classList.toggle('active', b.textContent.includes(score ? score + '★' : 'All'));
    });
}

function filterTeam(team) {
    currentTeamFilter = team;
    applyFilters();
}

function applyFilters() {
    document.querySelectorAll('.clip').forEach(el => {
        const scoreMatch = !currentScoreFilter || el.dataset.score == currentScoreFilter;
        const teamMatch = !currentTeamFilter || el.dataset.team === currentTeamFilter;
        el.classList.toggle('hidden', !(scoreMatch && teamMatch));
    });
}

function playClip(btn, path) {
    const clipEl = btn.closest('.clip');
    let video = clipEl.querySelector('video');
    if (video) {
        video.remove();
        btn.textContent = '▶ Play';
        return;
    }
    video = document.createElement('video');
    video.src = 'file://' + path;
    video.controls = true;
    video.autoplay = true;
    video.style.width = '100%';
    clipEl.insertBefore(video, clipEl.querySelector('.clip-info'));
    btn.textContent = '✕ Close';
}
</script>
</body>
</html>"""

    report_path = os.path.join(output_dir, "index.html")
    with open(report_path, "w") as f:
        f.write(html)

    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_clips(input_dir: str) -> list[str]:
    """Find all supported video clips in directory (recursive)."""
    clips = []
    for root, _, files in os.walk(input_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS and not f.startswith("."):
                clips.append(os.path.join(root, f))
    return clips


def main():
    parser = argparse.ArgumentParser(
        description="🏀 HoopTriage — Sort, score, and organise basketball clips fast.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Folder containing video clips")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: <input>/hooptriage_report)")
    parser.add_argument("--min-score", type=int, default=0, help="Only include clips with score >= N in report")
    parser.add_argument("--no-teams", action="store_true", help="Skip jersey colour detection (faster)")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers (default: 4)")

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    output_dir = args.output or os.path.join(input_dir, "hooptriage_report")
    os.makedirs(output_dir, exist_ok=True)

    check_ffmpeg()

    print(f"🏀 HoopTriage")
    print(f"   Input:   {input_dir}")
    print(f"   Output:  {output_dir}")
    print()

    clips = find_clips(input_dir)
    if not clips:
        print("No video clips found. Supported formats: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        sys.exit(1)

    print(f"Found {len(clips)} clips. Analysing...\n")

    results = []
    detect_teams = not args.no_teams

    # Process clips (sequential for now — ffmpeg is already parallel-ish internally)
    for i, clip_path in enumerate(clips, 1):
        clip_name = os.path.basename(clip_path)
        print(f"  [{i}/{len(clips)}] {clip_name}", end="", flush=True)

        result = process_clip(clip_path, output_dir, detect_teams)

        stars = "★" * result["score"] + "☆" * (5 - result["score"])
        team_str = f" [{result['team_colour']}]" if result["team_colour"] else ""
        print(f"  →  {stars}{team_str}")

        results.append(result)

    # Filter by min score
    if args.min_score > 0:
        results = [r for r in results if r["score"] >= args.min_score]
        print(f"\nFiltered to {len(results)} clips with score >= {args.min_score}")

    # Generate report
    print(f"\nGenerating report...")
    report_path = generate_report(results, output_dir)

    # Summary
    scores = [r["score"] for r in results]
    print(f"\n{'='*50}")
    print(f"🏀 HoopTriage Complete!")
    print(f"{'='*50}")
    print(f"   Clips analysed:  {len(results)}")
    print(f"   Hot clips (4-5): {len([s for s in scores if s >= 4])}")
    print(f"   Medium (3):      {len([s for s in scores if s == 3])}")
    print(f"   Likely skip (≤2):{len([s for s in scores if s <= 2])}")
    print(f"\n   Report: {report_path}")
    print(f"\n   Open it:  open \"{report_path}\"")


if __name__ == "__main__":
    main()
