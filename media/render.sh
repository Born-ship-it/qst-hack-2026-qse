#!/usr/bin/env bash
# Render all media animations to GIF.
#
# Usage:
#     bash media/render.sh              # render all three
#     bash media/render.sh workflow     # render one
#
# Requires: manim, ffmpeg, imageio[ffmpeg]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="output"
mkdir -p "$OUT_DIR"

# Low quality for fast iteration during development;
# override with QUALITY=-qh for 1080p60 on the final render.
QUALITY="${QUALITY:--ql}"
FPS="${FPS:-15}"
WIDTH="${WIDTH:-800}"

render_one() {
    local name="$1"
    echo "── Rendering $name ──"
    manim "$QUALITY" --format=mp4 -o "$name" "$name.py" "${name^}" 2>&1 | tail -5

    local mp4
    mp4=$(find media/videos -name "${name}.mp4" -newer "$name.py" | head -1)
    if [[ -z "$mp4" ]]; then
        echo "  ERROR: no mp4 produced for $name" >&2
        return 1
    fi

    local gif="$OUT_DIR/$name.gif"
    echo "  Converting to $gif"

    # Two-pass palette for clean GIFs with limited colours
    ffmpeg -y -loglevel error \
        -i "$mp4" \
        -vf "fps=$FPS,scale=$WIDTH:-1:flags=lanczos,palettegen" \
        /tmp/palette.png
    ffmpeg -y -loglevel error \
        -i "$mp4" -i /tmp/palette.png \
        -lavfi "fps=$FPS,scale=$WIDTH:-1:flags=lanczos[x];[x][1:v]paletteuse" \
        "$gif"

    local size
    size=$(stat -c%s "$gif" 2>/dev/null || stat -f%z "$gif")
    echo "  $gif  ($(( size / 1024 )) KB)"
}

if [[ $# -gt 0 ]]; then
    for name in "$@"; do
        render_one "$name"
    done
else
    render_one workflow
    render_one sqd_animation
    render_one qse_animation
fi

echo
echo "Done. GIFs written to $OUT_DIR/"