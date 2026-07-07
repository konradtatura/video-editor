"""
Surfaces low-ASR-confidence words that need a human decision before
captions are finalized.

This exists because cross-checking multiple ASR passes against each other
does not catch a mishearing every pass makes the same way (see SKILL.md:
"ROAS" was consistently misheard the same way across every transcription
pass tried). The only reliable catch for that class of error is a human who
knows the actual content -- this script's job is to point at candidates,
not decide anything.

NOTE on what this deliberately does NOT do: an earlier version tried fuzzy
text-similarity matching against domain.json's vocabulary (e.g. flagging a
word that's textually close to "ROAS"). Calibrating it against real cases
from this project showed it doesn't work: genuine mishearings ("0az" vs
"ROAS" -> 0.29, "5x0x" vs "ROAS" -> 0.00) scored LOWER than coincidental
false positives ("tak" vs "CTA" -> 0.67). ASR mishearings are driven by how
a word SOUNDS, not how it's spelled, so character-edit-distance is the
wrong tool -- no threshold fixes this, since real cases and noise overlap
in score. There's no reliable Polish phonetic-matching library to reach for
here, so this is a documented limitation, not a bug: domain.json's value is
as CONTEXT for a human/Claude reading the transcript with domain knowledge
in mind, not as an automated pre-filter. When reading a transcript, read
domain.json first and actively watch for places where a number, acronym,
or name from that vocabulary plausibly belongs but the transcript reads
vague or ungrammatical instead -- that's a human judgment call, and it
should become a question to the user, not a silent substitution.

Usage:
    python flag_ambiguous.py <words.json> [--min-score 0.6]
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("words_json")
    parser.add_argument("--min-score", type=float, default=0.6)
    args = parser.parse_args()

    with open(args.words_json, encoding="utf-8") as f:
        words = json.load(f)["words"]

    candidates = [w for w in words if w["score"] < args.min_score]

    print(f"[flag_ambiguous] {len(candidates)} word(s) scored below {args.min_score} confidence:\n")
    for w in candidates:
        print(f"  {w['start']:.2f}s  '{w['word']}'  (score {w['score']:.2f})")
    if not candidates:
        print("  none")
    print(
        "\n[flag_ambiguous] low confidence alone doesn't mean wrong -- but read each of these "
        "against domain.json's context before finalizing. If a number, acronym, or name from the "
        "domain vocabulary plausibly belongs here and the transcript reads vague/ungrammatical "
        "instead, ask the user rather than guess a replacement."
    )


if __name__ == "__main__":
    main()
