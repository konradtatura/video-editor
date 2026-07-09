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

Also lets the user drag a clip's raw start/end boundary directly (click a
clip on the timeline to open its trim panel, scrub the raw footage, "set
start/end here", Save & re-render) -- confirmed user preference: describing
a cut point in words to be translated back into a timestamp is slower and
less reliable than nudging it by ear/eye directly, so the flag-list flow
above is for coarse/structural notes ("repeat left in") while trimming is
for fine boundary placement. This still writes through cutlist.json (via
range_server.py's POST /api/trim, which edits the file and re-renders) --
it is not a UI-only preview, cutlist.json stays the single source of truth,
it's just edited from the browser now instead of by Claude relaying a
timestamp back into the file by hand.

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
  .tl-handle {{
    position: absolute; top: 0; bottom: 0; width: 10px; z-index: 6; cursor: ew-resize;
  }}
  .tl-handle.start {{ left: 0; }}
  .tl-handle.end {{ right: 0; }}
  .tl-handle::after {{
    content: ''; position: absolute; top: 6px; bottom: 6px; left: 4px; width: 2px;
    background: #5b8cff; border-radius: 2px; opacity: 0; transition: opacity .1s;
  }}
  .tl-handle:hover::after, .tl-handle.dragging::after {{ opacity: 1; }}
  .tl-clip.dragging {{ outline: 2px solid #5b8cff; outline-offset: -1px; z-index: 3; }}
  .tl-drag-label {{
    position: absolute; bottom: 100%; margin-bottom: 4px; transform: translateX(-50%);
    background: #262b35; border: 1px solid #383e4a; border-radius: 4px; padding: 2px 6px;
    font-size: 11px; white-space: nowrap; pointer-events: none; z-index: 7;
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
  .tl-clip.editing {{ outline: 2px solid #5b8cff; outline-offset: -1px; }}
  .trim-panel {{
    margin-top: 14px; background: #1c2029; border: 1px solid #2a2f3a; border-radius: 8px;
    padding: 12px; font-size: 13px;
  }}
  .trim-panel h3 {{ margin: 0 0 8px; font-size: 13px; color: #fff; }}
  .trim-panel .note-preview {{ color: #8a8d95; font-size: 12px; margin-bottom: 10px; }}
  .trim-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
  .trim-row label {{ width: 42px; color: #8a8d95; }}
  .trim-row input[type=number] {{
    width: 90px; background: #14161a; color: #e8e8ea; border: 1px solid #383e4a;
    border-radius: 6px; padding: 5px 7px; font-size: 13px;
  }}
  .trim-row button {{
    background: #262b35; color: #e8e8ea; border: 1px solid #383e4a; border-radius: 6px;
    padding: 5px 9px; font-size: 12px; cursor: pointer;
  }}
  .trim-row button:hover {{ background: #323947; }}
  #rawPreview {{ width: 100%; max-height: 30vh; border-radius: 6px; background: #000; margin: 8px 0; display: block; }}
  .trim-actions {{ display: flex; gap: 8px; margin-top: 10px; }}
  .save-btn {{
    flex: 1; background: #2f8a4e; color: #fff; border: none; border-radius: 8px;
    padding: 9px; font-size: 13px; cursor: pointer; font-weight: 600;
  }}
  .save-btn:disabled {{ background: #2a3b30; color: #7a8a80; cursor: default; }}
  .cancel-btn {{
    background: #262b35; color: #e8e8ea; border: 1px solid #383e4a; border-radius: 8px;
    padding: 9px 14px; font-size: 13px; cursor: pointer;
  }}
  .trim-status {{ font-size: 12px; margin-top: 8px; color: #8a8d95; }}
  .trim-status.err {{ color: #ff8a8a; }}
  .trim-status.ok {{ color: #7fe0a3; }}
</style>
</head>
<body>

<h1>{project_name} -- review</h1>
<div class="sub">Watch, click a flag type at the moment you notice something (pauses the video), optionally add a short note, then copy the list into chat. Drag a clip's left/right edge on the timeline to trim it directly, or click the middle of a clip to fine-tune with a raw-footage preview.</div>

<div class="layout">
  <div class="video-col">
    <video id="v" src="{video_src}" controls></video>
    <div class="timeline" id="timeline"></div>
    <div class="trim-panel" id="trimPanel" style="display:none;"></div>
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
let CUTMAP = {cutmap_json};
let TOTAL_DURATION = {total_duration};
const PROJECT_NAME = {project_name_json};
const RAW_SRC = {raw_src_json};

const video = document.getElementById('v');
const timeline = document.getElementById('timeline');
const nowTime = document.getElementById('nowTime');
const flagButtons = document.getElementById('flagButtons');
const noteBox = document.getElementById('note');
const addBtn = document.getElementById('addBtn');
const flagListEl = document.getElementById('flagList');
const copyBtn = document.getElementById('copyBtn');
const trimPanel = document.getElementById('trimPanel');

let selectedType = null;
let pendingTime = null;
let flags = [];
let editingClipIndex = null;

function fmt(t) {{
  const m = Math.floor(t / 60);
  const s = (t - m * 60).toFixed(1);
  return m + ':' + (s < 10 ? '0' : '') + s;
}}

function renderTimeline() {{
  timeline.innerHTML = '';
  CUTMAP.forEach((c, i) => {{
    const div = document.createElement('div');
    div.className = 'tl-clip' + (i === editingClipIndex ? ' editing' : '');
    const leftPct = (c.output_start / TOTAL_DURATION) * 100;
    const widthPct = ((c.output_end - c.output_start) / TOTAL_DURATION) * 100;
    div.style.left = leftPct + '%';
    div.style.width = widthPct + '%';
    div.title = 'clip ' + c.clip + ': ' + c.note_preview + ' (drag an edge to trim, click to fine-tune)';
    div.addEventListener('click', (e) => {{ e.stopPropagation(); openTrimPanel(i); }});

    const leftHandle = document.createElement('div');
    leftHandle.className = 'tl-handle start';
    leftHandle.addEventListener('mousedown', (e) => {{ startDrag(e, i, 'start'); }});
    div.appendChild(leftHandle);

    const rightHandle = document.createElement('div');
    rightHandle.className = 'tl-handle end';
    rightHandle.addEventListener('mousedown', (e) => {{ startDrag(e, i, 'end'); }});
    div.appendChild(rightHandle);

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

function fmtRaw(t) {{ return t.toFixed(2) + 's'; }}

function startDrag(e, clipIndex, edge) {{
  e.preventDefault();
  e.stopPropagation();
  const c = CUTMAP[clipIndex];
  const clipDiv = timeline.children[clipIndex];
  const handle = clipDiv.querySelector('.tl-handle.' + edge);
  const startClientX = e.clientX;
  const origRawStart = c.raw_start;
  const origRawEnd = c.raw_end;
  const origOutputStart = c.output_start;
  const origOutputEnd = c.output_end;

  clipDiv.classList.add('dragging');
  handle.classList.add('dragging');
  const label = document.createElement('div');
  label.className = 'tl-drag-label';
  clipDiv.appendChild(label);

  let liveStart = origRawStart;
  let liveEnd = origRawEnd;

  function onMove(ev) {{
    const rect = timeline.getBoundingClientRect();
    const deltaPx = ev.clientX - startClientX;
    const deltaSec = (deltaPx / rect.width) * TOTAL_DURATION;
    const MIN_DUR = 0.2;
    if (edge === 'start') {{
      liveStart = Math.max(0, Math.min(origRawEnd - MIN_DUR, origRawStart + deltaSec));
      const newLeftPct = ((origOutputStart + (liveStart - origRawStart)) / TOTAL_DURATION) * 100;
      clipDiv.style.left = newLeftPct + '%';
      clipDiv.style.width = (((origOutputEnd - origOutputStart) - (liveStart - origRawStart)) / TOTAL_DURATION) * 100 + '%';
      label.style.left = '0%';
      label.textContent = 'start ' + fmtRaw(liveStart);
    }} else {{
      liveEnd = Math.max(origRawStart + MIN_DUR, origRawEnd + deltaSec);
      clipDiv.style.width = (((origOutputEnd - origOutputStart) + (liveEnd - origRawEnd)) / TOTAL_DURATION) * 100 + '%';
      label.style.left = '100%';
      label.textContent = 'end ' + fmtRaw(liveEnd);
    }}
  }}

  function onUp() {{
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    label.remove();
    clipDiv.classList.remove('dragging');
    handle.classList.remove('dragging');
    if (Math.abs(liveStart - origRawStart) < 0.01 && Math.abs(liveEnd - origRawEnd) < 0.01) {{
      renderTimeline();
      renderFlagMarkers();
      return; // no real movement, just a click-through -- don't trigger a render
    }}
    saveTrim(clipIndex, liveStart, liveEnd);
  }}

  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}}

async function saveTrim(clipIndex, newStart, newEnd) {{
  if (!(newEnd > newStart)) {{
    renderTimeline();
    renderFlagMarkers();
    return;
  }}
  const statusEl = document.getElementById('trimStatus');
  const saveBtn = document.getElementById('saveTrimBtn');
  if (saveBtn) {{ saveBtn.disabled = true; saveBtn.textContent = 'Re-rendering...'; }}
  if (statusEl) {{ statusEl.textContent = 're-rendering...'; statusEl.className = 'trim-status'; }}
  try {{
    const resp = await fetch('/api/trim', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{project: PROJECT_NAME, clip_index: clipIndex, start: newStart, end: newEnd}}),
    }});
    const data = await resp.json();
    if (!data.ok) {{
      if (statusEl) {{ statusEl.textContent = 'failed: ' + (data.error || 'unknown error'); statusEl.className = 'trim-status err'; }}
      renderTimeline();
      renderFlagMarkers();
      return;
    }}
    CUTMAP = data.cutmap;
    TOTAL_DURATION = data.total_duration;
    if (editingClipIndex !== null) {{
      openTrimPanel(editingClipIndex); // refresh the panel's inputs to the saved values
    }} else {{
      renderTimeline();
      renderFlagMarkers();
    }}
    video.src = video.getAttribute('src').split('?')[0] + '?t=' + Date.now();
    video.load();
    if (statusEl) {{ statusEl.textContent = 'saved and re-rendered.'; statusEl.className = 'trim-status ok'; }}
  }} catch (err) {{
    if (statusEl) {{ statusEl.textContent = 'request failed: ' + err; statusEl.className = 'trim-status err'; }}
    renderTimeline();
    renderFlagMarkers();
  }} finally {{
    if (saveBtn) {{ saveBtn.disabled = false; saveBtn.textContent = 'Save & re-render'; }}
  }}
}}

function openTrimPanel(i) {{
  editingClipIndex = i;
  renderTimeline();
  renderFlagMarkers();
  const c = CUTMAP[i];
  trimPanel.style.display = 'block';
  trimPanel.innerHTML =
    '<h3>Clip ' + c.clip + ' -- raw footage boundaries</h3>' +
    '<div class="note-preview">' + c.note_preview.replace(/</g,'&lt;') + '</div>' +
    '<video id="rawPreview" src="' + RAW_SRC + '" controls></video>' +
    '<div class="trim-row"><label>start</label>' +
      '<input type="number" id="startInput" step="0.05" value="' + c.raw_start.toFixed(2) + '">' +
      '<button id="seekStartBtn">jump to it</button>' +
      '<button id="setStartBtn">set = preview time</button></div>' +
    '<div class="trim-row"><label>end</label>' +
      '<input type="number" id="endInput" step="0.05" value="' + c.raw_end.toFixed(2) + '">' +
      '<button id="seekEndBtn">jump to it</button>' +
      '<button id="setEndBtn">set = preview time</button></div>' +
    '<div class="trim-actions">' +
      '<button class="save-btn" id="saveTrimBtn">Save &amp; re-render</button>' +
      '<button class="cancel-btn" id="cancelTrimBtn">Close</button>' +
    '</div>' +
    '<div class="trim-status" id="trimStatus"></div>';

  const rawPreview = document.getElementById('rawPreview');
  const startInput = document.getElementById('startInput');
  const endInput = document.getElementById('endInput');
  const trimStatus = document.getElementById('trimStatus');

  document.getElementById('seekStartBtn').addEventListener('click', () => {{
    rawPreview.currentTime = Math.max(0, parseFloat(startInput.value) - 1.0);
  }});
  document.getElementById('seekEndBtn').addEventListener('click', () => {{
    rawPreview.currentTime = Math.max(0, parseFloat(endInput.value) - 1.0);
  }});
  document.getElementById('setStartBtn').addEventListener('click', () => {{
    startInput.value = rawPreview.currentTime.toFixed(2);
  }});
  document.getElementById('setEndBtn').addEventListener('click', () => {{
    endInput.value = rawPreview.currentTime.toFixed(2);
  }});
  document.getElementById('cancelTrimBtn').addEventListener('click', closeTrimPanel);

  document.getElementById('saveTrimBtn').addEventListener('click', () => {{
    const newStart = parseFloat(startInput.value);
    const newEnd = parseFloat(endInput.value);
    if (!(newEnd > newStart)) {{
      trimStatus.textContent = 'end must be after start';
      trimStatus.className = 'trim-status err';
      return;
    }}
    saveTrim(i, newStart, newEnd);
  }});
}}

function closeTrimPanel() {{
  editingClipIndex = null;
  trimPanel.style.display = 'none';
  trimPanel.innerHTML = '';
  renderTimeline();
  renderFlagMarkers();
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
    parser.add_argument("--video", default="output.mp4",
                         help="video file to embed, relative to the project dir (default: the plain render, "
                              "so trim edits show up immediately without re-running annotate_cuts.py)")
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
              f"run render.py first, or pass --video output_review.mp4 for the numbered-overlay copy", file=sys.stderr)

    page = PAGE_TEMPLATE.format(
        project_name=html.escape(project_name),
        video_src=html.escape(args.video),
        cutmap_json=json.dumps(cutmap),
        total_duration=total_duration,
        project_name_json=json.dumps(project_name),
        raw_src_json=json.dumps(cutlist["source"]),
    )

    out_path = os.path.join(project_dir, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"[review_console] wrote {out_path}", file=sys.stderr)
    print(f"[review_console] serve projects/ (see .claude/launch.json's 'review-console' config) and open "
          f"http://localhost:8420/{project_name}/{args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
