"""
Generates a self-contained review.html for a project: video + timeline in
sync, arrow-key frame-nudging, and an in/out marker that deletes the marked
range for real -- writes straight to cutlist.json and re-renders (via
range_server.py's POST /api/delete-range), with no re-verification step.

This is deliberately narrower than earlier versions of this tool. Two prior
designs were built and abandoned in the same project:
1. A flag-and-copy console (mark a problem, copy a list, paste into chat)
   -- worked, but still cost a round trip through Claude per fix.
2. A drag-to-resize timeline trim editor -- rejected by the user after
   hands-on testing: no undo, can't split, "pain in the ass."
This version is explicitly scoped to what the user asked for instead:
"only the timeline... the marker so I can mark the timestamps to delete...
precise navigating... I don't want reverifying, no checking if now it's
correct. You are doing your work one time [the automatic pass]. Then manual
tagging the timestamps to delete, and you are deleting them." So: no resize
handles (delete-only, never stretches a clip back out past its current
boundary), no flag types/notes/copy-to-chat, and the delete endpoint does
not call verify.py or regenerate captions -- it only edits cutlist.json and
re-renders. Re-verification and recaptioning happen once, separately, when
the user says they're done (see SKILL.md's "Iterating after review").

Usage:
    python review_console.py <project_dir> [--video output.mp4] [--out review.html]

Then serve it (`.claude/launch.json`'s "review-console" config, which runs
range_server.py -- not `python -m http.server`, see range_server.py's own
docstring for why) and open http://localhost:8420/<project_name>/review.html
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
    max-width: 720px; margin-left: auto; margin-right: auto;
  }}
  h1 {{ font-size: 18px; font-weight: 600; margin: 0 0 4px; color: #fff; }}
  .sub {{ color: #8a8d95; font-size: 13px; margin-bottom: 20px; line-height: 1.5; }}
  .sub b {{ color: #ccc; }}
  video {{
    width: 100%; max-height: 70vh; border-radius: 10px; background: #000; display: block;
    object-fit: contain; margin: 0 auto;
  }}
  .timeline {{
    position: relative; height: 52px; margin-top: 10px; border-radius: 6px;
    background: #1f232b; cursor: pointer; overflow: hidden; border: 1px solid #2a2f3a;
  }}
  .tl-clip {{
    position: absolute; top: 0; bottom: 0; border-right: 1px solid #383e4a; pointer-events: none;
  }}
  .tl-clip:nth-child(odd) {{ background: #232833; }}
  .tl-clip:nth-child(even) {{ background: #1c2029; }}
  .tl-playhead {{
    position: absolute; top: 0; bottom: 0; width: 2px; background: #ff5a5f;
    pointer-events: none; z-index: 5;
  }}
  .tl-range {{
    position: absolute; top: 4px; bottom: 4px; background: rgba(91, 140, 255, 0.35);
    border: 1px solid #5b8cff; border-radius: 3px; z-index: 3; pointer-events: none;
  }}
  .tl-range.set {{ background: rgba(255, 90, 95, 0.4); border-color: #ff5a5f; }}
  .now {{ font-variant-numeric: tabular-nums; font-size: 26px; color: #fff; margin: 14px 0 4px; }}
  .now span {{ color: #6d7280; font-size: 13px; }}
  .step-row {{ display: flex; gap: 8px; align-items: center; margin-bottom: 16px; }}
  .step-row button {{
    background: #262b35; color: #e8e8ea; border: 1px solid #383e4a; border-radius: 8px;
    padding: 8px 12px; font-size: 13px; cursor: pointer;
  }}
  .step-row button:hover {{ background: #323947; }}
  .step-row .hint {{ color: #6d7280; font-size: 12px; }}
  .mark-row {{ display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }}
  .mark-row button {{
    background: #262b35; color: #e8e8ea; border: 1px solid #383e4a; border-radius: 8px;
    padding: 10px 14px; font-size: 13px; cursor: pointer;
  }}
  .mark-row button:hover {{ background: #323947; }}
  .mark-row button.armed {{ outline: 2px solid #5b8cff; }}
  .mark-row button:disabled {{ opacity: 0.4; cursor: default; }}
  .delete-btn {{
    background: #7a2020 !important; border-color: #a33 !important; font-weight: 600;
  }}
  .delete-btn:hover {{ background: #8f2626 !important; }}
  .mark-status {{ font-size: 13px; color: #8a8d95; min-height: 18px; margin-bottom: 6px; }}
  .mark-status.ok {{ color: #7fe0a3; }}
  .mark-status.err {{ color: #ff8a8a; }}
  .mark-status.busy {{ color: #e0c97f; }}
</style>
</head>
<body>

<h1>{project_name} -- review</h1>
<div class="sub">
  Space/click to play. <b>Left/Right</b> arrows nudge {step}s (hold <b>Shift</b> for {big_step}s).
  Press <b>I</b> at the start of a bad part, <b>O</b> at the end, then <b>Delete marked range</b> --
  it cuts that exact span out of the video for real (re-renders, no re-check). Marks only ever
  remove time, never add it back.
</div>

<video id="v" src="{video_src}" controls></video>
<div class="timeline" id="timeline"></div>
<div class="now"><span>at</span> <span id="nowTime">0:00.00</span></div>

<div class="mark-row">
  <button id="markStartBtn" title="Shortcut: I">Mark start (I)</button>
  <button id="markEndBtn" disabled title="Shortcut: O">Mark end (O)</button>
  <button id="deleteBtn" class="delete-btn" disabled>Delete marked range</button>
  <button id="cancelMarkBtn" disabled>Cancel mark</button>
</div>
<div class="mark-status" id="markStatus"></div>

<script>
let CUTMAP = {cutmap_json};
let TOTAL_DURATION = {total_duration};
const PROJECT_NAME = {project_name_json};
const STEP = {step};
const BIG_STEP = {big_step};

const video = document.getElementById('v');
const timeline = document.getElementById('timeline');
const nowTime = document.getElementById('nowTime');
const markStartBtn = document.getElementById('markStartBtn');
const markEndBtn = document.getElementById('markEndBtn');
const deleteBtn = document.getElementById('deleteBtn');
const cancelMarkBtn = document.getElementById('cancelMarkBtn');
const markStatus = document.getElementById('markStatus');

let markStart = null;
let markEnd = null;

function fmt(t) {{
  const m = Math.floor(t / 60);
  const s = (t - m * 60).toFixed(2);
  return m + ':' + (s < 10 ? '0' : '') + s;
}}

function renderTimeline() {{
  timeline.innerHTML = '';
  CUTMAP.forEach(c => {{
    const div = document.createElement('div');
    div.className = 'tl-clip';
    div.style.left = ((c.output_start / TOTAL_DURATION) * 100) + '%';
    div.style.width = (((c.output_end - c.output_start) / TOTAL_DURATION) * 100) + '%';
    div.title = c.note_preview;
    timeline.appendChild(div);
  }});
  const playhead = document.createElement('div');
  playhead.className = 'tl-playhead';
  playhead.id = 'playhead';
  timeline.appendChild(playhead);
  renderMarkRange();
}}

function renderMarkRange() {{
  const old = document.getElementById('markRangeBar');
  if (old) old.remove();
  if (markStart === null) return;
  const end = markEnd !== null ? markEnd : video.currentTime;
  const bar = document.createElement('div');
  bar.id = 'markRangeBar';
  bar.className = 'tl-range' + (markEnd !== null ? ' set' : '');
  bar.style.left = ((markStart / TOTAL_DURATION) * 100) + '%';
  bar.style.width = ((Math.max(0, end - markStart) / TOTAL_DURATION) * 100) + '%';
  timeline.appendChild(bar);
}}

function updatePlayhead() {{
  const pct = (video.currentTime / TOTAL_DURATION) * 100;
  const ph = document.getElementById('playhead');
  if (ph) ph.style.left = pct + '%';
  nowTime.textContent = fmt(video.currentTime);
  if (markStart !== null && markEnd === null) renderMarkRange();
}}

function setStatus(text, cls) {{
  markStatus.textContent = text;
  markStatus.className = 'mark-status' + (cls ? ' ' + cls : '');
}}

function markRangeStart() {{
  markStart = video.currentTime;
  markEnd = null;
  video.pause();
  markStartBtn.classList.add('armed');
  markEndBtn.disabled = false;
  deleteBtn.disabled = true;
  cancelMarkBtn.disabled = false;
  setStatus('in point set at ' + fmt(markStart) + ' -- play/scrub/nudge to the end, then press O');
  renderMarkRange();
}}

function markRangeEnd() {{
  if (markStart === null) return;
  video.pause();
  if (video.currentTime <= markStart) {{
    setStatus('end must be after the in point (' + fmt(markStart) + ')', 'err');
    return;
  }}
  markEnd = video.currentTime;
  markStartBtn.classList.remove('armed');
  markEndBtn.disabled = true;
  deleteBtn.disabled = false;
  setStatus('marked ' + fmt(markStart) + '–' + fmt(markEnd) + ' -- press Delete marked range to cut it, or Cancel');
  renderMarkRange();
}}

function cancelMark() {{
  markStart = null;
  markEnd = null;
  markStartBtn.classList.remove('armed');
  markEndBtn.disabled = true;
  deleteBtn.disabled = true;
  cancelMarkBtn.disabled = true;
  setStatus('');
  renderMarkRange();
}}

async function deleteMarkedRange() {{
  if (markStart === null || markEnd === null) return;
  deleteBtn.disabled = true;
  markStartBtn.disabled = true;
  setStatus('cutting ' + fmt(markStart) + '–' + fmt(markEnd) + ' and re-rendering...', 'busy');
  try {{
    const resp = await fetch('/api/delete-range', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{project: PROJECT_NAME, start: markStart, end: markEnd}}),
    }});
    const data = await resp.json();
    if (!data.ok) {{
      setStatus('failed: ' + (data.error || 'unknown error'), 'err');
      deleteBtn.disabled = false;
      markStartBtn.disabled = false;
      return;
    }}
    CUTMAP = data.cutmap;
    TOTAL_DURATION = data.total_duration;
    const cutAt = markStart;
    cancelMark();
    renderTimeline();
    video.src = video.getAttribute('src').split('?')[0] + '?t=' + Date.now();
    video.load();
    video.addEventListener('loadedmetadata', () => {{ video.currentTime = Math.min(cutAt, TOTAL_DURATION); }}, {{once: true}});
    setStatus('deleted -- new duration ' + fmt(TOTAL_DURATION), 'ok');
  }} catch (err) {{
    setStatus('request failed: ' + err, 'err');
  }} finally {{
    deleteBtn.disabled = true;
    markStartBtn.disabled = false;
  }}
}}

markStartBtn.addEventListener('click', markRangeStart);
markEndBtn.addEventListener('click', markRangeEnd);
deleteBtn.addEventListener('click', deleteMarkedRange);
cancelMarkBtn.addEventListener('click', cancelMark);

document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'i' || e.key === 'I') {{ e.preventDefault(); markRangeStart(); }}
  else if (e.key === 'o' || e.key === 'O') {{ e.preventDefault(); markRangeEnd(); }}
  else if (e.key === 'ArrowLeft') {{
    e.preventDefault();
    video.pause();
    video.currentTime = Math.max(0, video.currentTime - (e.shiftKey ? BIG_STEP : STEP));
  }} else if (e.key === 'ArrowRight') {{
    e.preventDefault();
    video.pause();
    video.currentTime = Math.min(TOTAL_DURATION, video.currentTime + (e.shiftKey ? BIG_STEP : STEP));
  }} else if (e.code === 'Space' || e.key === ' ') {{
    // Handled explicitly (not left to the browser default) so it works
    // the same regardless of what last had focus -- e.g. without this,
    // pressing space right after clicking "Mark start" would re-trigger
    // that button via the browser's native space-activates-focused-
    // button behavior instead of toggling playback.
    e.preventDefault();
    if (video.paused) video.play(); else video.pause();
  }}
}});

timeline.addEventListener('click', (e) => {{
  const rect = timeline.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  video.currentTime = Math.max(0, Math.min(TOTAL_DURATION, pct * TOTAL_DURATION));
}});

video.addEventListener('timeupdate', updatePlayhead);
video.addEventListener('loadedmetadata', updatePlayhead);

renderTimeline();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir")
    parser.add_argument("--video", default="output.mp4",
                         help="video file to embed, relative to the project dir (default: the plain render, "
                              "so deletes show up immediately)")
    parser.add_argument("--out", default="review.html")
    parser.add_argument("--step", type=float, default=0.2, help="arrow-key nudge size in seconds")
    parser.add_argument("--big-step", type=float, default=1.0, help="shift+arrow nudge size in seconds")
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
              f"run render.py first", file=sys.stderr)

    page = PAGE_TEMPLATE.format(
        project_name=html.escape(project_name),
        video_src=html.escape(args.video),
        cutmap_json=json.dumps(cutmap),
        total_duration=total_duration,
        project_name_json=json.dumps(project_name),
        step=args.step,
        big_step=args.big_step,
    )

    out_path = os.path.join(project_dir, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"[review_console] wrote {out_path}", file=sys.stderr)
    print(f"[review_console] serve projects/ (see .claude/launch.json's 'review-console' config) and open "
          f"http://localhost:8420/{project_name}/{args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
