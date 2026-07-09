"""
Generates a self-contained review.html for a project: plays the rendered
output in a real browser (via a local static server, e.g. `.claude/launch.json`'s
"review-console" config -- python -m http.server), shows a clickable timeline
strip marking every cut join and clip, and lets the user drop timestamped
flags while watching instead of writing prose the way they had to before.

Why a real local server instead of an inline chat widget: an inline widget
runs in a sandboxed context with no access to files on disk, so it cannot
play a local video file. A local static server can. The tradeoff: flags are
handed back via a "copy to clipboard, paste into chat" button rather than a
one-click send -- still eliminates all timestamp-guessing (the actual
problem), just costs one paste instead of zero.

Scope, deliberately: this DOES let the user mark problems while watching.
This does NOT let them trim/edit/re-render from the page -- that would mean
rebuilding a video editor and abandoning cutlist.json as the single source
of truth, which is what makes the render/verify loop fast and auditable.
Editing stays with Claude, reading the flags back and translating them into
cutlist.json edits (the same way prose feedback always was).

Usage:
    python review_console.py <project_dir> [--video output_review.mp4] [--out review.html]

Then serve it (e.g. `python -m http.server 8420 --directory projects` from
the repo root) and open http://localhost:8420/<project_name>/review.html
"""

import argparse
import html
import json
import os
import sys

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def build_cutmap(cutlist, cuts):
    starts = [0.0] + list(cuts[:-1])
    cutmap = []
    for i, (clip, start, end) in enumerate(zip(cutlist["keep"], starts, cuts)):
        note = clip.get("note", "")
        cutmap.append({
            "clip": i + 1,
            "output_start": start,
            "output_end": end,
            "raw_start": clip["start"],
            "raw_end": clip["end"],
            "note_preview": (note[:100] + "...") if len(note) > 100 else note,
        })
    return cutmap


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Review -- {project_name}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px; background: #14161a; color: #e8e8ea;
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
  }}
  h1 {{ font-size: 18px; font-weight: 600; margin: 0 0 4px; color: #fff; }}
  .sub {{ color: #8a8d95; font-size: 13px; margin-bottom: 20px; }}
  .layout {{ display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }}
  .video-col {{ flex: 1 1 260px; max-width: 320px; }}
  video {{
    width: 100%; max-height: 78vh; border-radius: 10px; background: #000; display: block;
    object-fit: contain; margin: 0 auto;
  }}
  .timeline {{
    position: relative; height: 46px; margin-top: 10px; border-radius: 6px;
    background: #1f232b; cursor: pointer; overflow: hidden; border: 1px solid #2a2f3a;
  }}
  .tl-clip {{
    position: absolute; top: 0; bottom: 0; border-right: 1px solid #383e4a;
  }}
  .tl-clip:nth-child(odd) {{ background: #232833; }}
  .tl-clip:nth-child(even) {{ background: #1c2029; }}
  .tl-clip-label {{
    position: absolute; top: 2px; left: 4px; font-size: 10px; color: #6d7280;
    pointer-events: none; white-space: nowrap;
  }}
  .tl-playhead {{
    position: absolute; top: 0; bottom: 0; width: 2px; background: #ff5a5f;
    pointer-events: none; z-index: 5;
  }}
  .tl-flag {{
    position: absolute; bottom: 2px; width: 10px; height: 10px; border-radius: 50%;
    transform: translateX(-50%); border: 1px solid #14161a; z-index: 4; cursor: pointer;
  }}
  .controls-col {{ flex: 1 1 320px; min-width: 300px; }}
  .now {{ font-variant-numeric: tabular-nums; font-size: 22px; color: #fff; margin-bottom: 14px; }}
  .now span {{ color: #6d7280; font-size: 13px; }}
  .flag-buttons {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
  .flag-buttons button {{
    background: #262b35; color: #e8e8ea; border: 1px solid #383e4a; border-radius: 8px;
    padding: 9px 14px; font-size: 13px; cursor: pointer; transition: background .1s;
  }}
  .flag-buttons button:hover {{ background: #323947; }}
  .flag-buttons button.active {{ outline: 2px solid #5b8cff; }}
  textarea {{
    width: 100%; background: #1c2029; color: #e8e8ea; border: 1px solid #383e4a;
    border-radius: 8px; padding: 8px; font-size: 13px; resize: vertical; min-height: 40px;
  }}
  .add-btn {{
    margin-top: 8px; background: #5b8cff; color: #fff; border: none; border-radius: 8px;
    padding: 9px 16px; font-size: 13px; cursor: pointer; font-weight: 600;
  }}
  .add-btn:disabled {{ background: #384260; cursor: default; }}
  .flag-list {{ margin-top: 18px; border-top: 1px solid #2a2f3a; padding-top: 12px; }}
  .flag-item {{
    display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;
    background: #1c2029; border: 1px solid #2a2f3a; border-radius: 8px; padding: 8px 10px;
    margin-bottom: 6px; font-size: 13px;
  }}
  .flag-item .meta {{ color: #8a8d95; font-size: 11px; }}
  .flag-item button {{ background: none; border: none; color: #ff5a5f; cursor: pointer; font-size: 13px; }}
  .copy-btn {{
    margin-top: 14px; width: 100%; background: #232833; color: #e8e8ea; border: 1px solid #383e4a;
    border-radius: 8px; padding: 11px; font-size: 13px; cursor: pointer; font-weight: 600;
  }}
  .copy-btn.copied {{ background: #1e3a2a; border-color: #2f6b46; color: #7fe0a3; }}
  .empty {{ color: #6d7280; font-size: 13px; font-style: italic; }}
</style>
</head>
<body>

<h1>{project_name} -- review</h1>
<div class="sub">Watch, click a flag type at the moment you notice something (pauses the video), optionally add a short note, then copy the list into chat. No timestamps to guess.</div>

<div class="layout">
  <div class="video-col">
    <video id="v" src="{video_src}" controls></video>
    <div class="timeline" id="timeline"></div>
  </div>

  <div class="controls-col">
    <div class="now"><span>at</span> <span id="nowTime">0:00.0</span></div>

    <div class="flag-buttons" id="flagButtons">
      <button data-type="too-early">Cut too early</button>
      <button data-type="too-late">Cut too late</button>
      <button data-type="repeat">Repeat left in</button>
      <button data-type="cut-this">Cut this</button>
      <button data-type="other">Other</button>
    </div>
    <textarea id="note" placeholder="optional note (a few words)"></textarea>
    <button class="add-btn" id="addBtn" disabled>Pick a flag type above</button>

    <div class="flag-list">
      <div id="flagList"><div class="empty">No flags yet.</div></div>
      <button class="copy-btn" id="copyBtn">Copy flags for chat</button>
    </div>
  </div>
</div>

<script>
const CUTMAP = {cutmap_json};
const TOTAL_DURATION = {total_duration};

const video = document.getElementById('v');
const timeline = document.getElementById('timeline');
const nowTime = document.getElementById('nowTime');
const flagButtons = document.getElementById('flagButtons');
const noteBox = document.getElementById('note');
const addBtn = document.getElementById('addBtn');
const flagListEl = document.getElementById('flagList');
const copyBtn = document.getElementById('copyBtn');

let selectedType = null;
let pendingTime = null;
let flags = [];

function fmt(t) {{
  const m = Math.floor(t / 60);
  const s = (t - m * 60).toFixed(1);
  return m + ':' + (s < 10 ? '0' : '') + s;
}}

function renderTimeline() {{
  timeline.innerHTML = '';
  CUTMAP.forEach(c => {{
    const div = document.createElement('div');
    div.className = 'tl-clip';
    const leftPct = (c.output_start / TOTAL_DURATION) * 100;
    const widthPct = ((c.output_end - c.output_start) / TOTAL_DURATION) * 100;
    div.style.left = leftPct + '%';
    div.style.width = widthPct + '%';
    div.title = 'clip ' + c.clip + ': ' + c.note_preview;
    const label = document.createElement('div');
    label.className = 'tl-clip-label';
    label.textContent = c.clip;
    div.appendChild(label);
    timeline.appendChild(div);
  }});
  const playhead = document.createElement('div');
  playhead.className = 'tl-playhead';
  playhead.id = 'playhead';
  timeline.appendChild(playhead);
}}

function updatePlayhead() {{
  const pct = (video.currentTime / TOTAL_DURATION) * 100;
  const ph = document.getElementById('playhead');
  if (ph) ph.style.left = pct + '%';
  nowTime.textContent = fmt(video.currentTime);
}}

function clipAt(t) {{
  const c = CUTMAP.find(c => t >= c.output_start && t < c.output_end);
  return c ? c.clip : null;
}}

function renderFlagMarkers() {{
  document.querySelectorAll('.tl-flag').forEach(el => el.remove());
  const colors = {{'too-early':'#ffb020','too-late':'#ffb020','repeat':'#ff5a5f','cut-this':'#ff5a5f','other':'#5b8cff'}};
  flags.forEach((f, i) => {{
    const dot = document.createElement('div');
    dot.className = 'tl-flag';
    dot.style.left = ((f.time / TOTAL_DURATION) * 100) + '%';
    dot.style.background = colors[f.type] || '#5b8cff';
    dot.title = f.type + ' @ ' + fmt(f.time);
    dot.onclick = (e) => {{ e.stopPropagation(); video.currentTime = f.time; }};
    timeline.appendChild(dot);
  }});
}}

function renderFlagList() {{
  if (flags.length === 0) {{
    flagListEl.innerHTML = '<div class="empty">No flags yet.</div>';
    return;
  }}
  flagListEl.innerHTML = '';
  flags.forEach((f, i) => {{
    const item = document.createElement('div');
    item.className = 'flag-item';
    const clip = clipAt(f.time);
    item.innerHTML = '<div><div>' + fmt(f.time) + ' -- <b>' + f.type + '</b>' +
      (clip ? ' (clip ' + clip + ')' : '') +
      (f.note ? '<br><span class="meta">' + f.note.replace(/</g,'&lt;') + '</span>' : '') + '</div></div>';
    const rm = document.createElement('button');
    rm.textContent = 'remove';
    rm.onclick = () => {{ flags.splice(i, 1); renderFlagList(); renderFlagMarkers(); }};
    item.appendChild(rm);
    flagListEl.appendChild(item);
  }});
}}

timeline.addEventListener('click', (e) => {{
  const rect = timeline.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  video.currentTime = Math.max(0, Math.min(TOTAL_DURATION, pct * TOTAL_DURATION));
}});

video.addEventListener('timeupdate', updatePlayhead);
video.addEventListener('loadedmetadata', updatePlayhead);

flagButtons.querySelectorAll('button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    selectedType = btn.dataset.type;
    pendingTime = video.currentTime;
    video.pause();
    flagButtons.querySelectorAll('button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    addBtn.disabled = false;
    addBtn.textContent = 'Add flag at ' + fmt(pendingTime);
  }});
}});

addBtn.addEventListener('click', () => {{
  flags.push({{ time: pendingTime, type: selectedType, note: noteBox.value.trim() }});
  flags.sort((a, b) => a.time - b.time);
  noteBox.value = '';
  selectedType = null;
  pendingTime = null;
  flagButtons.querySelectorAll('button').forEach(b => b.classList.remove('active'));
  addBtn.disabled = true;
  addBtn.textContent = 'Pick a flag type above';
  renderFlagList();
  renderFlagMarkers();
}});

function showManualCopyFallback(text) {{
  // navigator.clipboard.writeText can fail for reasons outside our control
  // (lost document focus, a denied permission prompt, an older browser) --
  // confirmed directly that it fails silently if nothing catches the
  // rejected promise, which would strand the user with no way to get their
  // flags out. Always have a manual fallback, not just a "copied!" toast
  // that may not be true.
  let box = document.getElementById('manualCopyBox');
  if (!box) {{
    box = document.createElement('textarea');
    box.id = 'manualCopyBox';
    box.style.width = '100%';
    box.style.marginTop = '10px';
    box.style.minHeight = '90px';
    box.style.background = '#1c2029';
    box.style.color = '#e8e8ea';
    box.style.border = '1px solid #383e4a';
    box.style.borderRadius = '8px';
    box.style.padding = '8px';
    box.style.fontSize = '12px';
    copyBtn.insertAdjacentElement('afterend', box);
  }}
  box.value = text;
  box.style.display = 'block';
  box.focus();
  box.select();
}}

copyBtn.addEventListener('click', () => {{
  if (flags.length === 0) return;
  const lines = flags.map(f => {{
    const clip = clipAt(f.time);
    return '- ' + fmt(f.time) + (clip ? ' (clip ' + clip + ')' : '') + ' [' + f.type + ']' + (f.note ? ': ' + f.note : '');
  }});
  const text = 'Review flags for {project_name}:\\n' + lines.join('\\n');

  if (!navigator.clipboard || !navigator.clipboard.writeText) {{
    showManualCopyFallback(text);
    return;
  }}
  navigator.clipboard.writeText(text).then(() => {{
    copyBtn.textContent = 'Copied -- paste into chat';
    copyBtn.classList.add('copied');
    setTimeout(() => {{ copyBtn.textContent = 'Copy flags for chat'; copyBtn.classList.remove('copied'); }}, 2000);
  }}).catch(() => {{
    showManualCopyFallback(text);
  }});
}});

renderTimeline();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir")
    parser.add_argument("--video", default="output_review.mp4",
                         help="video file to embed, relative to the project dir (default: annotate_cuts.py's review copy)")
    parser.add_argument("--out", default="review.html")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    project_name = os.path.basename(project_dir)

    with open(os.path.join(project_dir, "cutlist.json"), encoding="utf-8") as f:
        cutlist = json.load(f)
    with open(os.path.join(project_dir, "timeline.json"), encoding="utf-8") as f:
        timeline = json.load(f)

    cuts = timeline["cuts"]
    cutmap = build_cutmap(cutlist, cuts)
    total_duration = cuts[-1]

    video_path = os.path.join(project_dir, args.video)
    if not os.path.exists(video_path):
        print(f"[review_console] warning: {args.video} not found in {project_dir} -- "
              f"run annotate_cuts.py first, or pass --video output.mp4 to use the plain render", file=sys.stderr)

    page = PAGE_TEMPLATE.format(
        project_name=html.escape(project_name),
        video_src=html.escape(args.video),
        cutmap_json=json.dumps(cutmap),
        total_duration=total_duration,
    )

    out_path = os.path.join(project_dir, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"[review_console] wrote {out_path}", file=sys.stderr)
    print(f"[review_console] serve projects/ (see .claude/launch.json's 'review-console' config) and open "
          f"http://localhost:8420/{project_name}/{args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
