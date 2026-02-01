# 🏀 HoopTriage

AI-powered basketball clip triage tool — sort, score, and organise tournament footage fast.

## The Problem

You filmed a basketball tournament. You have 1,200+ short clips. All thumbnails look the same. Sorting them takes an entire day.

## The Solution

HoopTriage analyses your clips and:

- **Scores excitement** (1-5) based on crowd noise and audio energy
- **Generates contact sheets** (4 frames per clip) so you can see what's in each clip at a glance
- **Detects jersey colours** to auto-group clips by team
- **Outputs an HTML report** you open in your browser to review everything fast

## Requirements

- Python 3.9+
- ffmpeg (must be in your PATH)

## Install

```bash
# Clone the repo
git clone https://github.com/snappyclawd/hooptriage.git
cd hooptriage

# Install dependencies
pip install -r requirements.txt
```

### Installing ffmpeg

**Mac:**
```bash
brew install ffmpeg
```

**Windows:**
```bash
winget install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

## Usage

```bash
# Full run — scan + triage + report
python3 hooptriage.py /path/to/clips

# Quick scan — instant report, no audio analysis
python3 hooptriage.py /path/to/clips --scan-only

# Specify output directory
python3 hooptriage.py /path/to/clips --output ./report
```

After running, open `hooptriage_report/index.html` in your browser.

## Features

### Hover Scrub
Move your mouse across any clip to scrub through it. Left edge = start, right edge = end. Just like Resolve's media browser.

### Audio Scoring
Extracts audio from each clip and scores excitement (1-5) based on crowd noise energy. High energy = crowd reaction = interesting clip.

### Manual Ratings
Click the stars on any clip to override the auto-score. Hover over a clip and press 1-5 on your keyboard for quick rating. Manual ratings are saved in your browser.

### Grid Size
Use the slider to adjust how many clips per row — from 1 (full width) to 5 (compact grid).

### Sorting & Filtering
Sort by score, name, or duration. Filter by score level or show only manually-rated clips.

### Live Updates
The report opens instantly after a quick scan. Audio triage runs in the background and scores appear as clips are processed — no need to wait.

### Double-Click to Play
Double-click any clip to expand it with full playback controls and audio.

## 100% Local

No files are uploaded anywhere. Everything runs on your machine. Your footage stays yours.

## License

MIT
