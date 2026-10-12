"""Subject-domain gating.

pixel-curator curates training data for any subject — people, objects,
animals, scenes, styles.  Some signals are inherently human-specific
(face detection, identity embeddings, pose/anatomy heuristics); others are
domain-agnostic (sharpness, exposure, duplicates, resolution, diversity).

``domain`` in the project file selects the signal set:

- ``human``  — full signal set (the original behavior).
- ``object`` / ``animal`` / ``general`` — domain-agnostic signals only;
  human-only stages explain themselves and exit instead of crashing or
  silently producing empty results.

Domain-agnostic stages (extract, photos, prefilter, basic curate, cluster,
caption, build, package, duplicates, health, insights, viz) never consult
this module — they work for every domain unchanged.
"""
from __future__ import annotations

from .config import VALID_DOMAINS

__all__ = ["domain_of", "is_human", "require_human", "HUMAN_ONLY_FEATURES"]

#: Features that need faces / human anatomy and therefore refuse to run
#: for non-human domains.
HUMAN_ONLY_FEATURES = {
    "anchor": "identity anchor (face embeddings)",
    "rescore": "identity rescoring",
    "likeness": "ArcFace likeness scoring",
    "pick-epoch": "epoch likeness ranking",
    "tune-stack": "stack weight tuning by likeness",
    "lookbook": "identity x outfit lookbook",
    "face-fix": "face consistency studio",
    "recommend-stack": "identity-aware stack recommendation",
    "smart-extract": "identity-aware smart extraction tiers",
}


def domain_of(prj) -> str:
    """Normalized domain string (unknown values fall back to 'human' with
    the doctor command flagging them)."""
    d = str(getattr(prj, "domain", "human") or "human").strip().lower()
    return d if d in VALID_DOMAINS else "human"


def is_human(prj) -> bool:
    return domain_of(prj) == "human"


def require_human(prj, feature: str) -> bool:
    """Gate a human-only stage.  Returns True when it may run; otherwise
    prints why not (with the generic alternative) and returns False."""
    if is_human(prj):
        return True
    what = HUMAN_ONLY_FEATURES.get(feature, feature)
    print(f"'{feature}' needs human-specific signals ({what}) but this "
          f"project's domain is '{domain_of(prj)}'.")
    print("Domain-agnostic pipeline that works for any subject:")
    print("  pixel-curator -p PROJECT extract  ->  curate  ->  caption  "
          "->  build  ->  package")
    print("Set domain = \"human\" in the project file to enable "
          "identity/face features.")
    return False
