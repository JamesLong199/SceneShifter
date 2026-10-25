#!/usr/bin/env python3
"""
Generate the *full questionnaire* for all subjects (all metadata prompts), including:
- background_semantics:
  - part1_presence: simple yes/no presence checks for each gt_item
  - part2_perceptual: perceptual questions that probe realism/plausibility of the same concepts
    (with "explain briefly if unsatisfactory")
- interaction_naturalness: pose + subject-environment plausibility questions

Important: this script does NOT run any VLM. It only produces the exact question strings.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


_SCRIPT_DIR = Path(__file__).resolve().parent
_MISC_DIR = _SCRIPT_DIR / "misc"
if _MISC_DIR.exists() and str(_MISC_DIR) not in sys.path:
    sys.path.insert(0, str(_MISC_DIR))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_segment_prompts(metadata_path: Path) -> List[str]:
    try:
        data = _read_json(metadata_path)
    except Exception:
        return []
    routing = data.get("frame_prompt_routing", {})
    prompts = routing.get("segment_prompts", [])
    if not isinstance(prompts, list):
        return []
    out: List[str] = []
    for p in prompts:
        out.append(p if isinstance(p, str) else "")
    return out


def _contains_any(text: str, toks: Sequence[str]) -> bool:
    t = text.lower()
    return any(tok in t for tok in toks)


def infer_subject_phrase_from_prompt(prompt: str) -> str:
    text = prompt.lower()
    if "the man and the woman" in text:
        return "the man and the woman"
    if "the woman and the man" in text:
        return "the woman and the man"
    if "the woman" in text:
        return "the woman"
    if "the man" in text:
        return "the man"
    return "the subject"


def _subject_possessive(subject_phrase: str) -> str:
    phrase = subject_phrase.strip() or "the subject"
    return f"{phrase}'" if phrase.endswith("s") else f"{phrase}'s"


def _subject_display(subject_phrase: str) -> str:
    return subject_phrase.strip() or "the subject"


def _subject_be(subject_phrase: str) -> str:
    return "are" if " and " in subject_phrase.lower() else "is"


def _subject_be_cap(subject_phrase: str) -> str:
    return "Are" if " and " in subject_phrase.lower() else "Is"


_ACTION_PATTERNS: List[tuple[str, str]] = [
    (r"\bstands?\s+up\b", "stand up"),
    (r"\bsits?\s+down\b", "sit down"),
    (r"\bsquats?\s+down\b", "squat down"),
    (r"\bwalks?\s+along\b", "walk along"),
    (r"\bstands?\b", "stand"),
    (r"\bsits?\b", "sit"),
    (r"\bsquats?\b", "squat"),
]


def infer_action_info_from_first_prompt(prompt: str) -> Dict[str, Any]:
    text = prompt.lower()
    actions: List[str] = []
    for pattern, canonical in _ACTION_PATTERNS:
        if re.search(pattern, text) and canonical not in actions:
            if canonical == "stand" and "stand up" in actions:
                continue
            if canonical == "sit" and "sit down" in actions:
                continue
            if canonical == "squat" and "squat down" in actions:
                continue
            actions.append(canonical)
    return {
        "action_mode": "multi_action" if len(actions) > 1 else "single_action",
        "actions": actions,
        "combo_action_text": " and ".join(actions) if len(actions) > 1 else (actions[0] if actions else ""),
    }


def build_combo_action_question(subject_phrase: str, combo_action_text: str) -> Dict[str, Any]:
    subject = _subject_display(subject_phrase)
    action = combo_action_text.strip()
    return {
        "id": "combo_action_performed",
        "question": f"Does {subject} perform the full combo action '{action}' in the video? Answer yes or no. If no, briefly explain (<=1 sentence).",
        "answer_format": "yes_no",
        "scope": "whole_video",
    }



def build_background_semantics_questionnaire(
    *,
    prompt: str,
    gt_items: List[str],
    presence_only: bool = True,
) -> Dict[str, Any]:
    """Presence-only background checks. Perceptual follow-ups are not generated."""
    del prompt, presence_only
    part1_presence = [
        {
            "gt_item": gi,
            "question": f"Is this statement true in the background? '{gi}' Answer with yes or no only.",
        }
        for gi in gt_items
    ]
    return {
        "background_semantics": {
            "part1_presence": part1_presence,
            "part2_perceptual": [],
            "presence_only": True,
            "gating_rule": "Perceptual questions are disabled in presence-only mode.",
            "answer_format_rule": "Answer each presence question with yes/no.",
        }
    }


def build_interaction_naturalness_questionnaire(
    *,
    prompt: str,
    expected_pose: str,
    expected_support: Optional[str],
    subject_phrase: str = "the subject",
) -> Dict[str, Any]:
    """
    Interaction questions aligned with the current evaluation script's intent:
      1) pose classification
      2) physical plausibility using GT pose
      3) conditional seating integration (when sitting + support object known)
      4) optional squatting stability check
    """
    pose = expected_pose.strip().lower() if expected_pose else ""
    support = expected_support.strip().lower() if isinstance(expected_support, str) and expected_support.strip() else None
    if pose == "standing":
        support = None

    gt_pose_phrase = pose if pose in {"standing", "sitting", "squatting"} else "posed"
    subject = _subject_display(subject_phrase)
    subject_possessive = _subject_possessive(subject)
    subject_be = _subject_be(subject)
    subject_be_cap = _subject_be_cap(subject)

    # We encode sequential gating explicitly so the evaluator can skip follow-ups if an earlier step fails.
    # Each question is an object to support "depends_on" and "forfeit_followups_if_no".
    qs: List[Dict[str, Any]] = []

    # Q1: pose classification (always asked)
    qs.append(
        {
            "id": "pose_classification",
            "question": f"What pose {subject_be} {subject} performing? Choose exactly one word: standing, sitting, or squatting.",
            "answer_format": "one_word",
            "allowed_answers": ["standing", "sitting", "squatting"],
        }
    )

    if pose == "sitting":
        # Q2: is the subject sitting on the expected support?
        if support:
            q2 = f"{subject_be_cap} {subject} sitting on a '{support}' (as intended by the prompt)? Answer yes or no. If no, briefly explain (<=1 sentence)."
        else:
            q2 = f"{subject_be_cap} {subject} sitting on a plausible seat/surface (as intended by the prompt)? Answer yes or no. If no, briefly explain (<=1 sentence)."

        qs.append(
            {
                "id": "sitting_on_expected_support",
                "depends_on": {"question_id": "pose_classification", "expected_answer": "sitting"},
                "question": q2,
                "answer_format": "yes_no",
                "forfeit_followups_if_no": ["sitting_integration"],
            }
        )

        # Q3: seating integration (only if Q2 is yes)
        if support:
            q3 = f"Does the sitting by {subject} look naturally supported by the '{support}' (hips/thighs contact the seat surface; body not floating above or sinking into it)? Answer yes/no. If no, briefly explain (<=1 sentence)."
        else:
            q3 = f"Does the sitting by {subject} look naturally supported by a plausible surface (even if partially occluded), without floating or severe intersection? Answer yes/no. If no, briefly explain (<=1 sentence)."

        qs.append(
            {
                "id": "sitting_integration",
                "depends_on": {"question_id": "sitting_on_expected_support", "expected_answer": "yes"},
                "question": q3,
                "answer_format": "yes_no",
            }
        )

    elif pose == "standing":
        qs.append(
            {
                "id": "standing_plausibility",
                "depends_on": {"question_id": "pose_classification", "expected_answer": "standing"},
                "question": f"Does the standing by {subject} look physically plausible (feet grounded, no obvious floating, no severe body-environment intersection)? Answer yes/no. If no, briefly explain (<=1 sentence).",
                "answer_format": "yes_no",
            }
        )

    elif pose == "squatting":
        qs.append(
            {
                "id": "squatting_plausibility",
                "depends_on": {"question_id": "pose_classification", "expected_answer": "squatting"},
                "question": f"Does the squatting by {subject} look physically plausible and stably grounded (feet contact the ground plausibly; no obvious hovering, sliding, or severe ground intersection)? Answer yes/no. If no, briefly explain (<=1 sentence).",
                "answer_format": "yes_no",
            }
        )

    return {
        "interaction_naturalness": {
            "expected_pose": pose or None,
            "expected_support": support,
            "subject_phrase": subject,
            "questions": qs,
            "sequential_rule": "For sitting: ask pose_classification -> sitting_on_expected_support -> sitting_integration. If sitting_on_expected_support is unsatisfactory (no), forfeit sitting_integration.",
            "answer_format_rule": "Answer each question with yes/no. If no, provide a concise explanation (<=1 sentence) where requested.",
        }
    }
