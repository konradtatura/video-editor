"""
Renders captions.json into an .ass subtitle file and burns it into the
video via ffmpeg/libass. captions.json is hand-authored (by reading
make_captions.py's word output against glossary.json) -- this script only
does the mechanical part: layout, styling, and burn-in.

Per-word highlighting is done with inline ASS override tags rather than a
second style, so a single caption card can mix default-colored and
highlighted words.

Usage:
    python burn_captions.py <captions.json> <output.mp4>
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

from ffmpeg_util import find_ffmpeg

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
DEFAULT_FONT = "Montserrat SemiBold"
DEFAULT_FONT_FILE = "Montserrat-SemiBold.ttf"
# Proxima Nova Semibold looked closest to the target reference style, but
# (a) it's a commercial font -- can't be defaulted-to or redistributed
# without your own license, and (b) the specific file available this
# session failed to load via libass/fontsdir entirely (see "Font
# resolution can fail silently" in SKILL.md) -- possibly connected to its
# unverified source, not just the licensing question. If you have a
# legitimately-licensed copy, drop the .otf into assets/ and override
# "font"/"font_file" locally via captions.json's "style" key (see below);
# verify it actually resolves (SKILL.md) before trusting any render.

DEFAULT_STYLE = {
    "font": DEFAULT_FONT,
    "font_file": DEFAULT_FONT_FILE,  # override per-project via captions.json's
                                       # "style" key -- e.g. to point at a
                                       # locally-licensed commercial font
                                       # without changing the shared default
    "font_size": 40,
    "primary_color": "#FFFFFF",
    "outline_color": "#666666",  # used for both outline and shadow (they
                                   # share the same ASS color field) --
                                   # lightened from black so the shadow
                                   # reads as a soft gray, not a dark ring
    "outline": 0.5,        # kept minimal, not zero -- a hairline helps edge
                            # definition against light backgrounds, but the
                            # reference has essentially no visible hard stroke
    "shadow": 0.3,
    "blur": 2.8,           # kept high on purpose -- low blur made the thin
                            # outline+shadow read as a hard dark ring; high
                            # blur diffuses it into a soft glow instead.
                            # Blur softens spread, shadow controls how dark/
                            # far it reaches -- they're not redundant, tune
                            # both independently rather than assuming one
                            # implies the other
    "letter_spacing": -1.5,  # ASS \fsp, pixels -- ports the "negative
                              # tracking" look confirmed from the reference
                              # (CapCut "character: -1" / Canva "-50")
    "alignment": 2,       # ASS numpad alignment: 2 = bottom-center
    "margin_v": 259,      # px from the bottom edge -- see SKILL.md "Vertical
                           # position" for how to recompute this per video
    "default_highlight": "#FE2C55",  # TikTok red, used if a word is
                                       # highlighted but no color was set
}


def hex_to_ass_color(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H00{b}{g}{r}".upper()


def get_video_resolution(ffprobe, path):
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "csv=p=0", path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    w, h = result.stdout.strip().split(",")
    return int(w), int(h)


def build_ass(captions, style, res_w, res_h):
    primary = hex_to_ass_color(style["primary_color"])
    outline_c = hex_to_ass_color(style["outline_color"])
    default_highlight = hex_to_ass_color(style["default_highlight"])

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_w}
PlayResY: {res_h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style['font']},{style['font_size']},{primary},{primary},{outline_c},&H00000000,0,0,0,0,100,100,0,0,1,{style['outline']},{style['shadow']},{style['alignment']},40,40,{style['margin_v']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def ass_time(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    lines = [header]
    for card in captions["cards"]:
        parts = []
        for w in card["words"]:
            color = hex_to_ass_color(w["highlight"]) if isinstance(w.get("highlight"), str) else (default_highlight if w.get("highlight") else None)
            text = w["text"]
            if color:
                parts.append(f"{{\\c{color}}}{text}{{\\c{primary}}}")
            else:
                parts.append(text)
        text = " ".join(parts)
        override = ""
        if style.get("blur"):
            override += f"\\blur{style['blur']}"
        if style.get("letter_spacing"):
            override += f"\\fsp{style['letter_spacing']}"
        if override:
            text = f"{{{override}}}{text}"
        lines.append(f"Dialogue: 0,{ass_time(card['start'])},{ass_time(card['end'])},Default,,0,0,0,,{text}")

    return "\n".join(lines)


def cache_key(captions_path, video_path, font_path):
    stat_c = os.stat(captions_path)
    stat_v = os.stat(video_path)
    stat_f = os.stat(font_path)
    key = f"{stat_c.st_mtime}|{stat_v.st_size}|{stat_v.st_mtime}|{stat_f.st_mtime}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("captions_json")
    parser.add_argument("output_video")
    parser.add_argument("--out", default=None, help="defaults to <output_video>_captioned.mp4")
    args = parser.parse_args()

    with open(args.captions_json, encoding="utf-8") as f:
        captions = json.load(f)

    style = {**DEFAULT_STYLE, **captions.get("style", {})}
    font_path = os.path.join(ASSETS_DIR, style["font_file"])

    ffmpeg = find_ffmpeg("ffmpeg")
    ffprobe = find_ffmpeg("ffprobe")
    res_w, res_h = get_video_resolution(ffprobe, args.output_video)

    ass_content = build_ass(captions, style, res_w, res_h)
    ass_path = os.path.splitext(args.captions_json)[0] + ".ass"
    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.write(ass_content)
    print(f"[burn_captions] wrote {ass_path}", file=sys.stderr)

    out_path = args.out or (os.path.splitext(args.output_video)[0] + "_captioned.mp4")

    # ffmpeg's ass filter needs forward slashes and escaped colons/backslashes
    # in the filter-graph string, even on Windows.
    ass_filter_path = ass_path.replace("\\", "/").replace(":", "\\:")
    fontsdir_filter_path = ASSETS_DIR.replace("\\", "/").replace(":", "\\:")

    cmd = [
        ffmpeg, "-y", "-i", args.output_video,
        "-vf", f"ass='{ass_filter_path}':fontsdir='{fontsdir_filter_path}'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr)
        raise RuntimeError("ffmpeg burn-in failed")

    print(f"[burn_captions] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
