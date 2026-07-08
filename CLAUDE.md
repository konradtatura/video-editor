# Default workflow for this project

The user is a marketer, not an engineer. Do not explain scripts, cutlists, transcripts,
or ask them to make technical choices. They send raw video(s); they want cut +
captioned video(s) back. Nothing else.

## When the user gives you one or more raw video files (any way — dropped in the repo,
a path, a folder of clips):

For **each** video, run the full pipeline yourself, back to back, with no pause for
confirmation between steps:

1. Set up `projects/<name>/` (pick `<name>` from the filename) and copy the raw file in.
2. Run the `rough-cut` skill fully: transcribe (both passes), read the transcript(s)
   yourself and build the cutlist (this is an editorial judgment call — make it, don't
   ask the user to review it), render, then verify.
3. Immediately run the `captions` skill on the rough-cut output to burn in captions,
   using whatever `glossary.json`/`domain.json` already exist in that skill's folder.
4. Copy the final captioned file to `output/<name>_final.mp4` (create `output/` if
   missing).

Do this for every video in the batch without stopping in between. Only interrupt the
user for one specific case: a retake where the *content* changed (different numbers,
claims, or facts between takes) — that's their editorial call, everything else
(fillers, dead air, which take to keep otherwise, caption styling) is yours to decide.

When all videos in the batch are done, give a short summary: one line per video,
where the final file landed, and a flag for anything you had to judge-call or any
retake-with-different-content question you're waiting on.

Do not ask the user about language, folder structure, model choice, or caption style
— detect/assume sensibly and proceed. If something is genuinely broken (e.g. ffmpeg
missing, a file won't transcribe), say so plainly and simply, without jargon.
