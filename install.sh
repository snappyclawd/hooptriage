#!/bin/bash
# HoopTriage installer — sets up the 'triage' command

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENTRY="$SCRIPT_DIR/hooptriage.py"

echo "🏀 HoopTriage Installer"
echo ""

# Check dependencies
if ! command -v python3 &>/dev/null; then
    echo "❌ python3 not found. Install it first (brew install python)."
    exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
    echo "❌ ffmpeg not found. Install it first (brew install ffmpeg)."
    exit 1
fi

# Install Python deps
echo "Installing Python dependencies..."
pip3 install -r "$SCRIPT_DIR/requirements.txt" -q

# Make executable
chmod +x "$ENTRY"

# Create symlink
LINK="/usr/local/bin/triage"
if [ -L "$LINK" ] || [ -f "$LINK" ]; then
    echo "Updating existing triage command..."
    sudo rm "$LINK"
fi

sudo ln -s "$ENTRY" "$LINK"

echo ""
echo "✅ Done! You can now run:"
echo ""
echo "   triage              — interactive (drag a folder)"
echo "   triage /path/to/clips  — direct"
echo "   triage --scan-only /path  — instant, no audio analysis"
echo ""
echo "🏀 Let's go!"
