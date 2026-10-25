"""
Lightweight (stdlib-only) utilities for extracting background-only checklist items from SceneShifter prompts.

This module intentionally avoids heavy deps so it can be imported in unit tests without needing
torch/decord/cv2/etc.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional


_CLOTHING_OR_FOREGROUND_TOKENS = {
    # clothing / subject descriptors
    "jeans",
    "jean",
    "trousers",
    "pants",
    "skirt",
    "dress",
    "sweater",
    "hoodie",
    "jacket",
    "coat",
    "shirt",
    "top",
    "t-shirt",
    "tee",
    "blouse",
    "shorts",
    "ankle-length",
    "long-sleeve",
    "long-sleeved",
    "knit",
    # subject words we never want in background items
    "man",
    "woman",
    "person",
    "face",
    "hair",
    "skin",
    "clothing",
    "subject",
}

_ATMOSPHERE_OR_ADJECTIVE_TOKENS = {
    # vibes / subjective descriptors, often incomplete when split by commas ("a casual.")
    "casual",
    "lived-in",
    "atmosphere",
    "vibe",
    "feel",
    "feeling",
}

_LIGHTING_OR_SHADING_TOKENS = {
    # user requested to exclude shading/lighting-related gt items
    "lighting",
    "sun",
    "sunlight",
    "sunlit",
    "shade",
    "shaded",
    "shadow",
    "shadows",
    "highlight",
    "highlights",
    "ambient",
    "warm",
    "late",
    "afternoon",
    "daytime",
    "night",
    "dusk",
    "dawn",
    "patch",
    "patches",
}

# Common non-background action/camera phrases that sometimes leak into parsing.
_ACTION_OR_CAMERA_TOKENS = {
    "continue",
    "continues",
    "continuing",
    "speak",
    "speaks",
    "speaking",
    "camera",
    "prepared",
    "silent",
    "remain",
    "remaining",
    "appearing",
    "lips",
    "closed",
}

# A small allowlist of tangible background nouns / structures.
# We only keep "with ..." parts that contain at least one of these tokens.
_BACKGROUND_OBJECT_ALLOWLIST = {
    "table",
    "tables",
    "chair",
    "chairs",
    "bench",
    "benches",
    "planter",
    "planters",
    "window",
    "windows",
    "lamp",
    "lamps",
    "light",
    "lights",
    "pendant",
    "floor",
    "wall",
    "ceiling",
    "pavement",
    "paved",
    "courtyard",
    "cafe",
    "café",
    "restaurant",
    "station",
    "concourse",
    "street",
    "road",
    "sidewalk",
    "kitchen",
    "bedroom",
    "living",
    "room",
}

_VALID_DATASET_POSES = {"standing", "sitting", "squatting"}


def strip_accents(s: str) -> str:
    s_norm = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s_norm if not unicodedata.combining(ch))


def extract_location_phrase(prompt: str) -> str:
    """
    Extract a short location noun phrase. Avoid trailing lighting/time clauses.
    """
    p = strip_accents(prompt).lower()

    # Prefer sentence form that appears in many scene0 prompts:
    #   "... speaks to the camera. The <location> in <lighting/time> ..."
    m = re.search(r"\.\s*the\s+(?P<loc>[^,\.]+?)\s+in\s+", p)
    if m:
        loc = m.group("loc").strip(" ,.")
        loc = re.sub(r"\s+", " ", loc)
        return loc

    # Common patterns: "... <pose> ... in <location> (in <lighting/time>) ..."
    patterns = [
        # "<pose> ... in <location> in <lighting/time>"
        r"\b(?:stand(?:s|ing)?|sit(?:s|ting)?|squat(?:s|ting)?)\b.*?\bin\s+(?:a |an |the )?(?P<loc>[^,\.]+?)\s+in\s+",
        # "<pose> ... in <location>, ..."
        r"\b(?:stand(?:s|ing)?|sit(?:s|ting)?|squat(?:s|ting)?)\b.*?\bin\s+(?:a |an |the )?(?P<loc>[^,\.]+?)(?:,|\.)",
        # generic "in <location> in <time>"
        r"\bin\s+(?:a |an |the )?(?P<loc>[^,\.]+?)\s+in\s+",
        # generic "in <location>, ..."
        r"\bin\s+(?:a |an |the )?(?P<loc>[^,\.]+?)(?:,|\.)",
    ]
    for pat in patterns:
        m = re.search(pat, p)
        if not m:
            continue
        loc = m.group("loc").strip(" ,.")
        loc = re.sub(r"\s+", " ", loc)
        # Guard against action leakage in weird sentence splits.
        if _contains_any_token(loc, _ACTION_OR_CAMERA_TOKENS):
            continue
        return loc
    return ""


def _contains_any_token(text: str, tokens: set[str]) -> bool:
    # simple word boundary-ish matching: compare on whitespace/punctuation split
    words = re.findall(r"[a-zA-Z]+(?:-[a-zA-Z]+)?", text.lower())
    return any(w in tokens for w in words)


def _has_background_object_keyword(text: str) -> bool:
    words = re.findall(r"[a-zA-Z]+(?:-[a-zA-Z]+)?", text.lower())
    return any(w in _BACKGROUND_OBJECT_ALLOWLIST for w in words)


def background_checklist_from_prompt(prompt: str, max_items: int = 5) -> List[str]:
    """
    Return an ordered list of background-only checklist statements.
    Excludes:
    - foreground clothing/subject terms
    - atmosphere/vibe adjectives (e.g. "a casual", "lived-in atmosphere")
    - shading/lighting cues (e.g. "sun patches", "shade")
    """
    p = strip_accents(prompt).strip()
    p_l = p.lower()

    items: List[str] = []

    loc = extract_location_phrase(p)
    if loc:
        # If location phrase itself is lighting-only, ignore; otherwise keep.
        if not _contains_any_token(loc, _LIGHTING_OR_SHADING_TOKENS):
            items.append(f"The background location is {loc}.")

    # Parse background object list tail (avoid the foreground clothing "with ...").
    # We only accept "with ..." that occurs *after* the location clause, or comma-separated
    # lists that follow the location/lighting clause.
    tail = ""

    # 1) Prefer "... in <loc> in <time> with <tail>."
    m = re.search(
        r"\bin\s+(?:a |an |the )?.+?\s+in\s+[^\.]+?\s+with\s+(?P<tail>.+?)(?:\.|$)",
        p_l,
    )
    if m:
        tail = m.group("tail")
    else:
        # 2) Or "... in <loc> in <time>, <tail>."
        m2 = re.search(
            r"\bin\s+(?:a |an |the )?.+?\s+in\s+[^\.]+?,\s*(?P<tail>.+?)(?:\.|$)",
            p_l,
        )
        if m2:
            tail = m2.group("tail")

    if tail:
        raw_parts = re.split(r",|\band\b", tail)
        for part in raw_parts:
            part = part.strip(" .")
            part = re.sub(r"\s+", " ", part)
            if not part:
                continue
            # reject action/camera phrases that sometimes leak in via earlier "with ..."
            if _contains_any_token(part, _ACTION_OR_CAMERA_TOKENS):
                continue
            # reject obviously incomplete fragments (e.g. "a casual")
            if len(part.split()) <= 2 and _contains_any_token(part, _ATMOSPHERE_OR_ADJECTIVE_TOKENS):
                continue
            # reject anything that looks like foreground clothing/person descriptors
            if _contains_any_token(part, _CLOTHING_OR_FOREGROUND_TOKENS):
                continue
            # reject atmosphere/vibe phrases
            if _contains_any_token(part, _ATMOSPHERE_OR_ADJECTIVE_TOKENS):
                continue
            # reject shading/lighting phrases
            if _contains_any_token(part, _LIGHTING_OR_SHADING_TOKENS):
                continue
            # keep only if it contains a tangible background keyword
            if not _has_background_object_keyword(part):
                continue
            items.append(f"The background contains {part}.")
            if len(items) >= max_items:
                break

    if not items:
        items = ["The background scene matches the described setting."]
    return items[:max_items]


def expected_pose_from_prompt(prompt: str) -> Optional[str]:
    p = prompt.lower()
    # Dataset poses: standing / sitting / squatting only.
    # Support both adjective and verb forms commonly used in prompts.
    # Walking clips are evaluated with the standing-vs-sitting-vs-squatting pose question.
    # Treat "walk along" as the standing pose class instead of leaving it unscored.
    if re.search(r"\bwalk(?:s|ing)?\s+along\b", p):
        return "standing"
    if re.search(r"\bstand(?:s|ing)?\b", p):
        return "standing"
    if re.search(r"\bsit(?:s|ting)?\b", p):
        return "sitting"
    if re.search(r"\bsquat(?:s|ting)?\b", p):
        return "squatting"
    return None


def expected_support_object_from_prompt(prompt: str) -> Optional[str]:
    p = prompt.lower()
    # Conservative list of support nouns that appear in SceneShifter prompts.
    # Note: this function only *reads* the prompt; it does not invent objects.
    for obj in ["bench", "chair", "stool", "step", "seawall", "railing"]:
        if obj in p:
            return obj
    return None

