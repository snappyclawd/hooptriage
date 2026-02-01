#!/usr/bin/env python3
"""
HoopTriage — AI-powered basketball clip triage tool.
Sort, score, and organise tournament footage fast.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import wave
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


def get_clip_info(clip_path: str) -> dict:
    """Return clip duration and basic info."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", clip_path],
        capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    duration = float(info.get("format", {}).get("duration", 0))
    size_bytes = int(info.get("format", {}).get("size", 0))

    # Get video dimensions
    width, height = 0, 0
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            width = stream.get("width", 0)
            height = stream.get("height", 0)
            break

    return {
        "duration": duration,
        "size_bytes": size_bytes,
        "width": width,
        "height": height,
    }


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


def extract_poster(clip_path: str, output_path: str, duration: float):
    """Extract a single poster frame from 25% into the clip."""
    t = max(duration * 0.25, 0.1)
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", clip_path,
            "-vframes", "1", "-q:v", "4",
            "-vf", "scale=640:-1",
            output_path,
        ],
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
            return {"rms": 0, "peak": 0, "dynamic_range": 0, "peak_window_rms": 0}
        raw = wf.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0

    rms = float(np.sqrt(np.mean(samples ** 2)))
    peak = float(np.max(np.abs(samples)))

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
    combined = (
        metrics.get("peak_window_rms", 0) * 0.5
        + metrics.get("dynamic_range", 0) * 0.3
        + metrics.get("rms", 0) * 0.2
    )

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
# Clip discovery
# ---------------------------------------------------------------------------

def find_clips(input_dir: str) -> list[str]:
    """Find all supported video clips in directory (recursive)."""
    clips = []
    for root, _, files in os.walk(input_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS and not f.startswith("."):
                clips.append(os.path.join(root, f))
    return clips


# ---------------------------------------------------------------------------
# Phase 1: Quick scan — generate HTML immediately
# ---------------------------------------------------------------------------

def quick_scan(clips: list[str]) -> list[dict]:
    """Quick scan: just get durations and paths. No heavy processing."""
    results = []
    for i, clip_path in enumerate(clips, 1):
        clip_name = os.path.basename(clip_path)
        print(f"  Scanning [{i}/{len(clips)}] {clip_name}", end="\r", flush=True)

        try:
            info = get_clip_info(clip_path)
        except Exception:
            info = {"duration": 0, "size_bytes": 0, "width": 0, "height": 0}

        results.append({
            "filename": clip_name,
            "path": os.path.abspath(clip_path),
            "duration": info["duration"],
            "size_bytes": info["size_bytes"],
            "width": info["width"],
            "height": info["height"],
            "score": 0,  # 0 = not yet analysed
            "audio_metrics": {},
            "poster": None,
        })

    print(f"  Scanned {len(clips)} clips.{' ' * 30}")
    return results


# ---------------------------------------------------------------------------
# Phase 2: Audio triage
# ---------------------------------------------------------------------------

def run_triage(clips: list[dict], output_dir: str):
    """Run audio analysis on all clips and update scores."""
    data_path = os.path.join(output_dir, "triage_data.json")

    for i, clip in enumerate(clips, 1):
        clip_name = clip["filename"]
        print(f"  [{i}/{len(clips)}] {clip_name}", end="", flush=True)

        # Audio analysis
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = extract_audio_pcm(clip["path"], tmp_dir)
            if wav_path:
                metrics = analyse_audio(wav_path)
                clip["audio_metrics"] = metrics
                clip["score"] = score_excitement(metrics)
            else:
                clip["score"] = 1

        # Extract poster frame
        poster_dir = os.path.join(output_dir, "posters")
        os.makedirs(poster_dir, exist_ok=True)
        poster_path = os.path.join(poster_dir, f"{Path(clip_name).stem}.jpg")
        extract_poster(clip["path"], poster_path, clip["duration"])
        if os.path.exists(poster_path):
            clip["poster"] = os.path.relpath(poster_path, output_dir)

        stars = "★" * clip["score"] + "☆" * (5 - clip["score"])
        print(f"  →  {stars}")

        # Write progress to JSON after each clip
        with open(data_path, "w") as f:
            json.dump(clips, f, indent=2)

    return clips


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_report(clips: list[dict], output_dir: str, input_dir: str) -> str:
    """Generate the HTML report with hover-scrub, manual ratings, and grid controls."""

    clips_json = json.dumps(clips, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🏀 HoopTriage — {len(clips)} clips</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }}

    /* Top bar */
    .topbar {{ position: sticky; top: 0; z-index: 100; background: #0f172aee; backdrop-filter: blur(12px); border-bottom: 1px solid #1e293b; padding: 12px 24px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
    .topbar h1 {{ font-size: 20px; white-space: nowrap; }}
    .stats {{ display: flex; gap: 16px; font-size: 13px; color: #94a3b8; }}
    .stats span {{ white-space: nowrap; }}
    .stats .num {{ color: #e2e8f0; font-weight: 600; }}

    /* Controls */
    .controls {{ display: flex; gap: 8px; align-items: center; margin-left: auto; flex-wrap: wrap; }}
    .btn {{ padding: 5px 14px; border-radius: 6px; border: 1px solid #334155; background: #1e293b; color: #e2e8f0; cursor: pointer; font-size: 12px; transition: all 0.15s; white-space: nowrap; }}
    .btn:hover {{ background: #334155; }}
    .btn.active {{ background: #3b82f6; border-color: #3b82f6; }}
    .size-slider {{ width: 100px; accent-color: #3b82f6; }}
    .sort-select {{ background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px; padding: 5px 8px; font-size: 12px; }}

    /* Filter bar */
    .filterbar {{ position: sticky; top: 56px; z-index: 99; background: #0f172add; backdrop-filter: blur(12px); padding: 8px 24px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; border-bottom: 1px solid #1e293b; }}
    .filter-label {{ font-size: 12px; color: #64748b; margin-right: 4px; }}

    /* Grid */
    .grid-container {{ padding: 16px 24px; }}
    .grid {{ display: grid; gap: 12px; }}

    /* Clip card */
    .clip {{ background: #1e293b; border-radius: 10px; overflow: hidden; transition: transform 0.1s; position: relative; }}
    .clip:hover {{ transform: scale(1.01); }}
    .clip.hidden {{ display: none; }}

    /* Video container with hover scrub */
    .vid-wrap {{ position: relative; width: 100%; aspect-ratio: 16/9; background: #000; cursor: crosshair; overflow: hidden; }}
    .vid-wrap video {{ width: 100%; height: 100%; object-fit: cover; pointer-events: none; }}
    .vid-wrap img.poster {{ width: 100%; height: 100%; object-fit: cover; position: absolute; top: 0; left: 0; }}
    .scrub-bar {{ position: absolute; bottom: 0; left: 0; height: 3px; background: #3b82f6; transition: width 0.05s; pointer-events: none; }}
    .time-indicator {{ position: absolute; bottom: 8px; right: 8px; background: #000a; color: #fff; font-size: 11px; padding: 2px 6px; border-radius: 4px; pointer-events: none; opacity: 0; transition: opacity 0.15s; font-variant-numeric: tabular-nums; }}
    .vid-wrap:hover .time-indicator {{ opacity: 1; }}

    /* Triage badge */
    .triage-status {{ position: absolute; top: 8px; left: 8px; font-size: 10px; padding: 2px 8px; border-radius: 4px; background: #334155; color: #94a3b8; pointer-events: none; }}
    .triage-status.done {{ background: transparent; color: transparent; }}

    /* Clip info */
    .clip-info {{ padding: 8px 12px; display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
    .clip-name {{ font-size: 12px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; }}
    .clip-meta {{ display: flex; gap: 6px; align-items: center; flex-shrink: 0; }}
    .duration {{ font-size: 11px; color: #64748b; font-variant-numeric: tabular-nums; }}

    /* Star rating - clickable */
    .stars {{ display: inline-flex; gap: 1px; cursor: pointer; }}
    .stars .star {{ font-size: 16px; color: #334155; transition: color 0.1s; user-select: none; }}
    .stars .star.filled {{ color: #eab308; }}
    .stars .star:hover {{ color: #facc15; }}
    .stars.manual .star.filled {{ color: #22d3ee; }}

    /* Score badge */
    .score-badge {{ min-width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; border-radius: 6px; font-weight: 700; font-size: 13px; color: #0f172a; }}
    .score-badge.pending {{ background: #334155; color: #64748b; font-size: 10px; }}

    /* Expand to play */
    .expanded .vid-wrap {{ aspect-ratio: auto; }}
    .expanded video {{ pointer-events: auto; }}

    /* Score colours */
    .score-5 {{ background: #22c55e; }}
    .score-4 {{ background: #84cc16; }}
    .score-3 {{ background: #eab308; }}
    .score-2 {{ background: #f97316; }}
    .score-1 {{ background: #ef4444; }}
    .score-0 {{ background: #334155; color: #64748b; }}
</style>
</head>
<body>

<div class="topbar">
    <h1>🏀 HoopTriage</h1>
    <div class="stats">
        <span><span class="num" id="stat-total">{len(clips)}</span> clips</span>
        <span><span class="num" id="stat-hot">-</span> hot</span>
        <span><span class="num" id="stat-skip">-</span> skip</span>
        <span><span class="num" id="stat-duration">{sum(c['duration'] for c in clips) / 60:.0f}m</span> footage</span>
        <span id="triage-progress"></span>
    </div>
    <div class="controls">
        <label style="font-size:12px;color:#64748b;">Grid:</label>
        <input type="range" class="size-slider" min="1" max="5" value="3" id="grid-size">
        <select class="sort-select" id="sort-by">
            <option value="score-desc">Score ↓</option>
            <option value="score-asc">Score ↑</option>
            <option value="name-asc">Name A-Z</option>
            <option value="name-desc">Name Z-A</option>
            <option value="duration-desc">Longest</option>
            <option value="duration-asc">Shortest</option>
        </select>
    </div>
</div>

<div class="filterbar">
    <span class="filter-label">Score:</span>
    <button class="btn active" data-filter-score="all">All</button>
    <button class="btn" data-filter-score="5">5★</button>
    <button class="btn" data-filter-score="4">4★</button>
    <button class="btn" data-filter-score="3">3★</button>
    <button class="btn" data-filter-score="2">2★</button>
    <button class="btn" data-filter-score="1">1★</button>
    <button class="btn" data-filter-score="0" style="margin-left:4px">⏳ Pending</button>
    <span class="filter-label" style="margin-left:12px;">Show:</span>
    <button class="btn active" data-filter-type="all">All</button>
    <button class="btn" data-filter-type="manual">✋ Manually rated</button>
</div>

<div class="grid-container">
    <div class="grid" id="clip-grid"></div>
</div>

<script>
// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
const CLIPS = {clips_json};
const manualRatings = JSON.parse(localStorage.getItem('hooptriage_ratings') || '{{}}');

// Apply any saved manual ratings
CLIPS.forEach(c => {{
    if (manualRatings[c.filename] !== undefined) {{
        c.score = manualRatings[c.filename];
        c.manual = true;
    }}
}});

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let filterScore = 'all';
let filterType = 'all';

// ---------------------------------------------------------------------------
// Grid sizing
// ---------------------------------------------------------------------------
const gridSizes = {{ 1: '1fr', 2: 'repeat(2, 1fr)', 3: 'repeat(3, 1fr)', 4: 'repeat(4, 1fr)', 5: 'repeat(5, 1fr)' }};
const gridEl = document.getElementById('clip-grid');
const sizeSlider = document.getElementById('grid-size');

sizeSlider.addEventListener('input', () => {{
    gridEl.style.gridTemplateColumns = gridSizes[sizeSlider.value];
}});
gridEl.style.gridTemplateColumns = gridSizes[sizeSlider.value];

// ---------------------------------------------------------------------------
// Render clips
// ---------------------------------------------------------------------------
function renderClips() {{
    gridEl.innerHTML = '';

    // Sort
    const sortBy = document.getElementById('sort-by').value;
    const sorted = [...CLIPS].sort((a, b) => {{
        switch(sortBy) {{
            case 'score-desc': return (b.score || 0) - (a.score || 0);
            case 'score-asc': return (a.score || 0) - (b.score || 0);
            case 'name-asc': return a.filename.localeCompare(b.filename);
            case 'name-desc': return b.filename.localeCompare(a.filename);
            case 'duration-desc': return b.duration - a.duration;
            case 'duration-asc': return a.duration - b.duration;
        }}
    }});

    sorted.forEach((clip, idx) => {{
        // Filter
        if (filterScore !== 'all' && String(clip.score) !== filterScore) return;
        if (filterType === 'manual' && !clip.manual) return;

        const card = document.createElement('div');
        card.className = 'clip';
        card.dataset.idx = idx;
        card.dataset.filename = clip.filename;

        const scoreClass = clip.score > 0 ? `score-${{clip.score}}` : 'score-0';
        const stars = renderStars(clip.score, clip.manual);
        const dur = formatDuration(clip.duration);
        const triageLabel = clip.score === 0 ? '<span class="triage-status">⏳ pending</span>' : '<span class="triage-status done"></span>';
        const posterSrc = clip.poster ? clip.poster : '';
        const posterImg = posterSrc ? `<img class="poster" src="${{posterSrc}}" alt="">` : '';

        card.innerHTML = `
            <div class="vid-wrap" data-path="${{clip.path}}" data-duration="${{clip.duration}}">
                ${{posterImg}}
                <div class="scrub-bar"></div>
                <div class="time-indicator">0:00</div>
                ${{triageLabel}}
            </div>
            <div class="clip-info">
                <span class="clip-name" title="${{clip.filename}}">${{clip.filename}}</span>
                <div class="clip-meta">
                    <span class="duration">${{dur}}</span>
                    <span class="stars ${{clip.manual ? 'manual' : ''}}" data-filename="${{clip.filename}}">${{stars}}</span>
                    <span class="score-badge ${{scoreClass}}">${{clip.score || '?'}}</span>
                </div>
            </div>
        `;

        gridEl.appendChild(card);
    }});

    // Attach events
    attachScrubEvents();
    attachStarEvents();
    updateStats();
}}

function renderStars(score, manual) {{
    let html = '';
    for (let i = 1; i <= 5; i++) {{
        html += `<span class="star ${{i <= score ? 'filled' : ''}}" data-value="${{i}}">★</span>`;
    }}
    return html;
}}

function formatDuration(secs) {{
    if (secs < 60) return `${{secs.toFixed(1)}}s`;
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${{m}}:${{String(s).padStart(2, '0')}}`;
}}

function updateStats() {{
    const scored = CLIPS.filter(c => c.score > 0);
    document.getElementById('stat-hot').textContent = scored.filter(c => c.score >= 4).length || '-';
    document.getElementById('stat-skip').textContent = scored.filter(c => c.score <= 2).length || '-';

    const pending = CLIPS.filter(c => c.score === 0).length;
    const prog = document.getElementById('triage-progress');
    if (pending > 0) {{
        prog.innerHTML = `<span style="color:#eab308">⏳ ${{CLIPS.length - pending}}/${{CLIPS.length}} triaged</span>`;
    }} else {{
        prog.innerHTML = '<span style="color:#22c55e">✓ All triaged</span>';
    }}
}}

// ---------------------------------------------------------------------------
// Hover scrub
// ---------------------------------------------------------------------------
function attachScrubEvents() {{
    document.querySelectorAll('.vid-wrap').forEach(wrap => {{
        let video = null;
        let isLoaded = false;

        wrap.addEventListener('mouseenter', () => {{
            if (wrap.closest('.expanded')) return;
            if (!video) {{
                video = document.createElement('video');
                video.src = 'file://' + wrap.dataset.path;
                video.muted = true;
                video.preload = 'auto';
                video.playsInline = true;
                video.style.cssText = 'width:100%;height:100%;object-fit:cover;pointer-events:none;';
                wrap.insertBefore(video, wrap.firstChild);

                video.addEventListener('loadeddata', () => {{
                    isLoaded = true;
                    const poster = wrap.querySelector('.poster');
                    if (poster) poster.style.opacity = '0';
                }});
            }}
        }});

        wrap.addEventListener('mousemove', (e) => {{
            if (!video || !isLoaded || wrap.closest('.expanded')) return;
            const rect = wrap.getBoundingClientRect();
            const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            const duration = parseFloat(wrap.dataset.duration) || 0;
            const time = pct * duration;

            video.currentTime = time;

            // Update scrub bar
            wrap.querySelector('.scrub-bar').style.width = (pct * 100) + '%';

            // Update time indicator
            const mins = Math.floor(time / 60);
            const secs = Math.floor(time % 60);
            wrap.querySelector('.time-indicator').textContent = `${{mins}}:${{String(secs).padStart(2, '0')}}`;
        }});

        wrap.addEventListener('mouseleave', () => {{
            if (wrap.closest('.expanded')) return;
            wrap.querySelector('.scrub-bar').style.width = '0';
            const poster = wrap.querySelector('.poster');
            if (poster) poster.style.opacity = '1';
        }});

        // Double-click to expand and play with controls
        wrap.addEventListener('dblclick', () => {{
            const card = wrap.closest('.clip');
            if (card.classList.contains('expanded')) {{
                card.classList.remove('expanded');
                if (video) {{
                    video.controls = false;
                    video.muted = true;
                    video.pause();
                    video.style.pointerEvents = 'none';
                }}
            }} else {{
                card.classList.add('expanded');
                if (video) {{
                    video.controls = true;
                    video.muted = false;
                    video.style.pointerEvents = 'auto';
                    video.play();
                }}
            }}
        }});
    }});
}}

// ---------------------------------------------------------------------------
// Clickable star ratings
// ---------------------------------------------------------------------------
function attachStarEvents() {{
    document.querySelectorAll('.stars').forEach(starsEl => {{
        starsEl.querySelectorAll('.star').forEach(star => {{
            star.addEventListener('click', (e) => {{
                e.stopPropagation();
                const filename = starsEl.dataset.filename;
                const value = parseInt(star.dataset.value);

                // Update data
                const clip = CLIPS.find(c => c.filename === filename);
                if (clip) {{
                    clip.score = value;
                    clip.manual = true;
                }}

                // Save to localStorage
                manualRatings[filename] = value;
                localStorage.setItem('hooptriage_ratings', JSON.stringify(manualRatings));

                // Update UI
                starsEl.classList.add('manual');
                starsEl.querySelectorAll('.star').forEach(s => {{
                    s.classList.toggle('filled', parseInt(s.dataset.value) <= value);
                }});

                // Update score badge
                const badge = starsEl.closest('.clip-meta').querySelector('.score-badge');
                badge.textContent = value;
                badge.className = `score-badge score-${{value}}`;

                updateStats();
            }});
        }});
    }});
}}

// ---------------------------------------------------------------------------
// Filter buttons
// ---------------------------------------------------------------------------
document.querySelectorAll('[data-filter-score]').forEach(btn => {{
    btn.addEventListener('click', () => {{
        filterScore = btn.dataset.filterScore;
        document.querySelectorAll('[data-filter-score]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderClips();
    }});
}});

document.querySelectorAll('[data-filter-type]').forEach(btn => {{
    btn.addEventListener('click', () => {{
        filterType = btn.dataset.filterType;
        document.querySelectorAll('[data-filter-type]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderClips();
    }});
}});

document.getElementById('sort-by').addEventListener('change', renderClips);

// ---------------------------------------------------------------------------
// Polling for triage updates
// ---------------------------------------------------------------------------
let pollInterval = null;

function pollTriageData() {{
    fetch('triage_data.json?t=' + Date.now())
        .then(r => r.ok ? r.json() : null)
        .then(data => {{
            if (!data) return;
            let changed = false;
            data.forEach(d => {{
                const clip = CLIPS.find(c => c.filename === d.filename);
                if (clip && !clip.manual && d.score > 0 && clip.score === 0) {{
                    clip.score = d.score;
                    clip.audio_metrics = d.audio_metrics;
                    clip.poster = d.poster;
                    changed = true;
                }}
            }});
            if (changed) renderClips();

            // Stop polling when all done
            if (CLIPS.every(c => c.score > 0)) {{
                clearInterval(pollInterval);
                document.getElementById('triage-progress').innerHTML = '<span style="color:#22c55e">✓ All triaged</span>';
            }}
        }})
        .catch(() => {{}});
}}

// Start polling if there are pending clips
if (CLIPS.some(c => c.score === 0)) {{
    pollInterval = setInterval(pollTriageData, 2000);
}}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------
document.addEventListener('keydown', (e) => {{
    // 1-5: rate focused/hovered clip
    if (e.key >= '1' && e.key <= '5') {{
        const hovered = document.querySelector('.clip:hover');
        if (hovered) {{
            const starsEl = hovered.querySelector('.stars');
            if (starsEl) {{
                const star = starsEl.querySelector(`[data-value="${{e.key}}"]`);
                if (star) star.click();
            }}
        }}
    }}
}});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
renderClips();
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

def main():
    parser = argparse.ArgumentParser(
        description="🏀 HoopTriage — Sort, score, and organise basketball clips fast.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", default=None, help="Folder containing video clips")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: <input>/hooptriage_report)")
    parser.add_argument("--min-score", type=int, default=0, help="Only include clips with score >= N in report")
    parser.add_argument("--scan-only", action="store_true", help="Quick scan only — skip audio triage (instant report)")
    parser.add_argument("--triage-only", action="store_true", help="Run triage on already-scanned clips")

    args = parser.parse_args()

    # Interactive mode: prompt for folder if not provided
    if args.input is None:
        print("🏀 HoopTriage")
        print()
        try:
            raw = input("Drag a folder here (or paste a path): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye! 🏀")
            sys.exit(0)

        # Clean up path (macOS drag-and-drop adds quotes and escapes spaces)
        raw = raw.strip("'\"")
        raw = raw.replace("\\ ", " ")
        args.input = raw

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

    # Phase 1: Quick scan
    print(f"Phase 1: Scanning {len(clips)} clips...")
    results = quick_scan(clips)

    # Generate report immediately (browsable before triage finishes)
    print(f"\nGenerating report...")
    report_path = generate_report(results, output_dir, input_dir)
    print(f"   Report ready: {report_path}")
    print(f"   Open it now:  open \"{report_path}\"\n")

    # Auto-open report in browser
    import platform
    if platform.system() == "Darwin":
        subprocess.Popen(["open", report_path])
    elif platform.system() == "Windows":
        os.startfile(report_path)
    else:
        subprocess.Popen(["xdg-open", report_path], stderr=subprocess.DEVNULL)

    if args.scan_only:
        print("Scan complete. Run with --triage-only to add audio scores later.")
        return

    # Phase 2: Audio triage
    print(f"Phase 2: Audio triage (scores will update in the report as they complete)...")
    results = run_triage(results, output_dir)

    # Regenerate final report with all scores
    generate_report(results, output_dir, input_dir)

    # Summary
    scores = [r["score"] for r in results if r["score"] > 0]
    print(f"\n{'='*50}")
    print(f"🏀 HoopTriage Complete!")
    print(f"{'='*50}")
    print(f"   Clips analysed:  {len(results)}")
    print(f"   Hot clips (4-5): {len([s for s in scores if s >= 4])}")
    print(f"   Medium (3):      {len([s for s in scores if s == 3])}")
    print(f"   Likely skip (≤2): {len([s for s in scores if s <= 2])}")
    print(f"\n   Report: {report_path}")
    print(f"   Open it:  open \"{report_path}\"")


if __name__ == "__main__":
    main()
