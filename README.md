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
# Basic — analyse a folder of clips
python hooptriage.py /path/to/clips

# Specify output directory
python hooptriage.py /path/to/clips --output ./report

# Only show clips scoring 3+
python hooptriage.py /path/to/clips --min-score 3

# Skip jersey detection (faster)
python hooptriage.py /path/to/clips --no-teams
```

After running, open `report/index.html` in your browser to review your clips.

## How It Works

1. **Audio Analysis** — Extracts audio from each clip, measures RMS energy and peak levels. Crowd reactions = high energy = interesting clip.
2. **Contact Sheets** — Extracts 4 evenly-spaced frames from each clip. Way more useful than a single thumbnail.
3. **Jersey Detection** — Samples frames and identifies dominant non-court colours to group clips by team.
4. **HTML Report** — Everything in a fast, sortable, filterable browser interface. Click to play any clip.

## 100% Local

No files are uploaded anywhere. Everything runs on your machine. Your footage stays yours.

## License

MIT
