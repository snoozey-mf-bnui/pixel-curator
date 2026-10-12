#!/usr/bin/env python3
"""
nsfw_router.py
Central NSFW routing engine.
Uses nsfw_vocab.py to intelligently route images into specialized NSFW recipes.
"""

try:
    from pixel_curator import nsfw_vocab
except ImportError:
    nsfw_vocab = None


def get_nsfw_category(tags: list) -> str:
    """Return the best NSFW category for a list of tags."""
    if not nsfw_vocab:
        return "general_nsfw"

    tag_set = set(t.lower() for t in tags)

    # Priority order
    priority = [
        ("pussy_focus", nsfw_vocab.NSFW_CATEGORY_ROUTING["pussy_focus"]),
        ("anal_focus", nsfw_vocab.NSFW_CATEGORY_ROUTING["anal_focus"]),
        ("cock_focus", nsfw_vocab.NSFW_CATEGORY_ROUTING["cock_focus"]),
        ("tits_focus", nsfw_vocab.NSFW_CATEGORY_ROUTING["tits_focus"]),
        ("kink_focus", nsfw_vocab.NSFW_CATEGORY_ROUTING["kink_focus"]),
        ("hair_focus", nsfw_vocab.NSFW_CATEGORY_ROUTING["hair_focus"]),
        ("fluids_focus", nsfw_vocab.NSFW_CATEGORY_ROUTING["fluids_focus"]),
    ]

    for category, keywords in priority:
        if any(kw in tag_set for kw in keywords):
            return category

    return "general_nsfw"


def should_use_nsfw_recipe(tags: list) -> bool:
    """Quick check if this image should go through NSFW routing."""
    if not nsfw_vocab:
        return False
    return any(nsfw_vocab.is_nsfw_tag(t) for t in tags)


def get_routing_boosters(category: str) -> list:
    """Return positive prompt boosters for a given NSFW category."""
    if not nsfw_vocab:
        return []
    return nsfw_vocab.POSITIVE_BOOSTERS
