"""
Produces a DRAFT captions.json from make_captions.py's word output and a
glossary: applies glossary substitutions mechanically (find a mishearing
phrase, replace with the canonical term, tag it for highlighting) and
groups words into short caption cards, preferring to break at punctuation.

This is a draft, the same way chunk_transcribe.py's output is a draft
transcript, not a finished cutlist -- read it over before rendering.
Glossary matching here is exact-phrase (case-insensitive, punctuation-
stripped), so it will miss mishearings you haven't listed yet; anything
that looks like a name, acronym, or number that Whisper likely mangled is
worth a manual second look even if this script didn't flag it.

Usage:
    python group_cards.py <words.json> <glossary.json> <output_captions.json> [--max-words 3]
"""

import argparse
import json
import re


def normalize(text):
    return re.sub(r"[.,!?;:]+$", "", text.strip().lower())


def apply_glossary(words, glossary):
    """Scans the word sequence for glossary mishearings and merges any
    match into a single word entry carrying the canonical text + highlight
    color, spanning the original timing."""
    terms = []
    for term in glossary.get("terms", []):
        for mishearing in term["mishearings"]:
            mis_words = mishearing.split()
            terms.append((mis_words, term["canonical"], term.get("highlight")))
        # the canonical form itself should also be highlighted wherever it
        # already appears correctly transcribed
        terms.append((term["canonical"].split(), term["canonical"], term.get("highlight")))
    # longest phrase first, so multi-word matches win over partial ones
    terms.sort(key=lambda t: -len(t[0]))

    result = []
    i = 0
    while i < len(words):
        matched = False
        for mis_words, canonical, highlight in terms:
            n = len(mis_words)
            if i + n > len(words):
                continue
            window = [normalize(w["word"]) for w in words[i:i + n]]
            if window == [normalize(w) for w in mis_words]:
                trailing_punct = ""
                m = re.search(r"[.,!?;:]+$", words[i + n - 1]["word"])
                if m:
                    trailing_punct = m.group(0)
                result.append({
                    "word": canonical + trailing_punct,
                    "start": words[i]["start"],
                    "end": words[i + n - 1]["end"],
                    "highlight": highlight,
                })
                i += n
                matched = True
                break
        if not matched:
            result.append({**words[i], "highlight": None})
            i += 1
    return result


def group_into_cards(words, max_words, cuts=None):
    """cuts: sorted list of hard-cut times in this video's own timeline (see
    timeline.json from the rough-cut skill's render.py). A card is never
    allowed to span a cut -- straddling one means the caption keeps
    displaying words from a shot that has already ended, or shows a word
    from the new shot before its card technically started, which reads as
    the caption dying mid-sentence right when the scene changes."""
    cuts = cuts or []
    cut_i = 0

    cards = []
    current = []
    for w in words:
        # ASR word timestamps are approximate; a cut point is exact (it's
        # where render.py actually spliced two segments). If a cut falls
        # inside this word's own reported span, trust the cut and clamp the
        # word to start there -- otherwise the word gets grouped with the
        # wrong side of the edit and the card-break logic below never
        # triggers (this was a real bug: found via a card that visibly
        # straddled a known cut when it shouldn't have).
        for cut in cuts:
            if w["start"] < cut < w["end"]:
                w = {**w, "start": cut}
                break

        # if a cut happened between the previous word and this one, close
        # out the current card first regardless of word count/punctuation
        while cut_i < len(cuts) and cuts[cut_i] <= w["start"]:
            if current:
                cards.append(current)
                current = []
            cut_i += 1

        current.append(w)
        ends_clause = bool(re.search(r"[.,!?]$", w["word"]))
        if len(current) >= max_words or ends_clause:
            cards.append(current)
            current = []
    if current:
        cards.append(current)

    result = [
        {
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "words": [{"text": w["word"], "highlight": w.get("highlight")} for w in group],
        }
        for group in cards
    ]

    # close gaps so a caption is always on screen between the first and
    # last word -- extending a card's end to the next card's start is
    # always safe re: cut boundaries, since the next card's start is
    # already guaranteed to be on the correct side of any cut.
    for i in range(len(result) - 1):
        result[i]["end"] = result[i + 1]["start"]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("words_json")
    parser.add_argument("glossary_json")
    parser.add_argument("output_captions_json")
    parser.add_argument("--max-words", type=int, default=4)
    parser.add_argument("--timeline", default=None, help="path to timeline.json (from rough-cut's render.py) -- if given, cards never span a hard cut")
    args = parser.parse_args()

    with open(args.words_json, encoding="utf-8") as f:
        words_data = json.load(f)
    with open(args.glossary_json, encoding="utf-8") as f:
        glossary = json.load(f)

    cuts = None
    if args.timeline:
        with open(args.timeline, encoding="utf-8") as f:
            cuts = json.load(f)["cuts"]

    merged_words = apply_glossary(words_data["words"], glossary)
    cards = group_into_cards(merged_words, args.max_words, cuts=cuts)

    out = {"cards": cards}
    with open(args.output_captions_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    n_highlighted = sum(1 for c in cards for w in c["words"] if w.get("highlight"))
    print(f"[group_cards] wrote {len(cards)} cards, {n_highlighted} glossary matches highlighted -> {args.output_captions_json}")
    print("[group_cards] this is a DRAFT -- review card breaks and glossary coverage before rendering")


if __name__ == "__main__":
    main()
