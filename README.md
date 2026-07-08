# video-editor

Two [Claude Code](https://claude.com/claude-code) skills for turning a raw talking-head recording into a captioned, filler-free short-form video — invoked from within a Claude Code session, not run standalone.

- **[`rough-cut`](.claude/skills/rough-cut/SKILL.md)** — transcribes raw footage, removes filler words/dead air/retakes via an editorial cutlist (read and decided by Claude, not a rules engine), and renders with a hash-cached segment pipeline so re-editing one cut only re-encodes that one clip.
- **[`captions`](.claude/skills/captions/SKILL.md)** — burns word-level, TikTok-style captions onto an already-edited video, with a glossary system for fixing words Whisper reliably mishears (acronyms, product names, numbers) and per-word highlight styling.

Typical flow: raw video → `rough-cut` → edited `output.mp4` → `captions` → final captioned video.

## Setup (one-time, per machine)

Both skills need ffmpeg on PATH and a Python environment with `faster-whisper` (CPU-only `torch`). They can share one venv.

**macOS / Linux**
```bash
brew install ffmpeg   # Linux: use your package manager instead
python3 -m venv venv-whisper
source venv-whisper/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r .claude/skills/rough-cut/requirements.txt
pip install -r .claude/skills/captions/requirements.txt
```

**Windows**
```powershell
winget install Gyan.FFmpeg   # restart the shell afterward so ffmpeg is on PATH
python -m venv venv-whisper
venv-whisper\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r .claude\skills\rough-cut\requirements.txt
pip install -r .claude\skills\captions\requirements.txt
```

Installing the CPU-only `torch` line *first*, separately, matters — otherwise pip may grab the much larger CUDA build by default.

No WSL, no WhisperX, no Adobe/Creative Cloud — both skills use `faster-whisper` directly (not the full `whisperx` package) specifically to avoid a Windows Smart App Control issue with one of whisperx's dependencies, and every bundled font is free/OFL-licensed (no commercial font included).

## Transcription backend: local vs. Groq

Both skills transcribe with local `faster-whisper` (`large-v3`, CPU) by default — accurate, but minutes per clip. There's an optional cloud backend using [Groq](https://groq.com)'s API, which runs the same `whisper-large-v3` model on their hardware and returns results in seconds instead of minutes, at roughly $0.03–0.05 per hour of audio (with a free tier).

**Tradeoff:** with the Groq backend, the extracted audio track leaves your machine and is sent to Groq's API. Local mode keeps everything on-device. Use local for anything sensitive or offline; use Groq when you want fast turnaround and don't mind the audio going over the network.

To use it:
1. Get an API key at [console.groq.com/keys](https://console.groq.com/keys).
2. Set it in your shell: `export GROQ_API_KEY=gsk_...` (or put it in a local, gitignored `.env` and `source .env`). Never commit this key.
3. Set `TRANSCRIBE_BACKEND=groq` before running any transcription step (`chunk_transcribe.py`, `transcribe2.py`, `make_captions.py`). Default is `local` if unset.

If `TRANSCRIBE_BACKEND=groq` is set but `GROQ_API_KEY` is missing, or a Groq API call fails (rate limit, network, invalid key), each script prints a clear message and falls back to local `faster-whisper` rather than crashing.

## Using it in Claude Code

This repo *is* the `.claude/skills/` layout Claude Code expects. Either:
- Clone this repo as your project root (skills are picked up automatically from `.claude/skills/`), or
- Copy `.claude/skills/rough-cut/` and `.claude/skills/captions/` into an existing project's `.claude/skills/` directory.

**Restart Claude Code** after adding the skills — they're only discovered at session start. Once restarted, invoke them with `/rough-cut` or `/captions`, or just describe what you want ("cut the filler out of this recording") and Claude will pick the right skill.

Every script needs to run through the venv's Python specifically (e.g. `venv-whisper/bin/python .claude/skills/rough-cut/scripts/chunk_transcribe.py ...`), not whatever `python` resolves to globally.

## Per-project setup

Each skill has editable, per-project config files (copy the `.example.json` templates):
- `rough-cut`: `cutlist.json` (built by Claude reading the transcript, not hand-written from scratch)
- `captions`: `glossary.example.json` → `glossary.json` (confirmed ASR-mishearing corrections), `domain.example.json` → `domain.json` (content niche/jargon, context for judgment)

See each skill's `SKILL.md` for the full pipeline, known limitations, and the reasoning behind non-obvious design choices (word timestamps near pauses are unreliable, no ASR transcript is fully trustworthy for repeats, font resolution can fail silently with no error, etc.) — those are worth reading before extending either skill, not just using it.
