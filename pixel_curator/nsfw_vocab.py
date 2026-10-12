#!/usr/bin/env python3
"""
nsfw_vocab.py
Comprehensive, filthy, dominant NSFW vocabulary for the LoRA Designer ecosystem.
Use this module to power captioning, routing, prompting, and specialized recipe building.
"""

# ======================== ANATOMY TAGS ========================
ANATOMY = {
    "pussy": ["pussy", "cunt", "vagina", "vulva", "labia", "clit", "clitoris", "fuckhole", "wet pussy", "dripping pussy", "spread pussy", "gaping pussy"],
    "cock": ["cock", "dick", "penis", "shaft", "hard cock", "throbbing cock", "veiny cock", "fat cock", "monster cock"],
    "tits": ["tits", "breasts", "boobs", "huge tits", "massive tits", "perky tits", "saggy tits", "nipples", "puffy nipples", "inverted nipples"],
    "ass": ["ass", "butt", "booty", "fat ass", "phat ass", "bubble butt", "asshole", "gaping asshole", "spread ass"],
    "body": ["naked", "nude", "completely naked", "bottomless", "topless", "stripped", "exposed"],
}

# ======================== HAIR TAGS ========================
HAIR = {
    "pubic": ["pubic hair", "bush", "hairy pussy", "shaved pussy", "landing strip", "neatly trimmed", "wild bush", "hairy mound"],
    "armpit": ["armpit hair", "hairy armpits", "shaved armpits", "stubbly armpits"],
    "general": ["hairy", "hirsute", "smooth", "shaved", "stubble"],
}

# ======================== FLUIDS & MESS TAGS ========================
FLUIDS = [
    "cum", "creampie", "cumshot", "facial", "cum on face", "cum in mouth", "cum on tits", "cum on ass",
    "pussy juice", "wet pussy", "dripping", "squirting", "squirted", "juicy", "sloppy", "messy",
    "precum", "drooling", "saliva", "spit", "throat pie", "overflowing cum"
]

# ======================== SEX ACTS ========================
ACTS = [
    "fucking", "hard fucking", "rough sex", "pounding", "deep penetration", "balls deep",
    "creampie", "internal creampie", "breeding", "impregnation", "gangbang", "double penetration",
    "anal", "ass fucking", "throat fucking", "face fucking", "titfuck", "paizuri",
    "69", "facesitting", "rimming", "eating pussy", "cunnilingus", "blowjob", "deepthroat",
    "handjob", "footjob", "masturbating", "fingering", "fisting"
]

# ======================== KINKS & FETISHES ========================
KINKS = [
    "ahegao", "rolled eyes", "tongue out", "mind break", "broken", "corrupted",
    "bondage", "tied up", "restrained", "shibari", "rope marks",
    "bdsm", "spanking", "choking", "hair pulling", "slapping",
    "watersports", "pissing", "golden shower",
    "pregnant", "pregnancy", "impregnated", "breeding kink",
    "lactation", "milking", "breast milk",
    "exhibitionism", "public", "risky",
    "degradation", "humiliation", "name calling", "slut", "whore", "cumdump",
    "pet play", "collar", "leash",
    "age play", "ddlg", "brat"
]

# ======================== POSITIVE NSFW PROMPT BOOSTERS ========================
POSITIVE_BOOSTERS = [
    "highly detailed pussy", "perfect anatomy", "wet and glistening", "aroused",
    "explicit details", "obscene", "lewd", "filthy", "depraved", "perverted",
    "detailed fluids", "stringy saliva", "cum strings", "pussy juice strings",
    "ahegao face", "ecstatic expression", "orgasm face"
]

# ======================== NEGATIVE PROMPT (Anti-SFW) ========================
NEGATIVE_BOOSTERS = [
    "clothed", "censored", "bar censor", "mosaic censor", "underwear", "bra", "panties",
    "safe for work", "sfw", "modest", "conservative", "covered", "blurred"
]

# ======================== CATEGORY ROUTING FOR SPECIALIZED RECIPES ========================
NSFW_CATEGORY_ROUTING = {
    "pussy_focus": ["pussy", "vulva", "labia", "clit", "spread pussy", "hairy pussy", "shaved pussy", "creampie", "pussy juice"],
    "cock_focus": ["cock", "dick", "penis", "hard cock", "cumshot", "creampie"],
    "tits_focus": ["tits", "breasts", "nipples", "titfuck", "lactation"],
    "anal_focus": ["ass", "asshole", "anal", "gaping asshole", "ass fucking"],
    "hair_focus": ["pubic hair", "bush", "hairy pussy", "armpit hair", "hirsute"],
    "kink_focus": ["ahegao", "bondage", "pregnant", "watersports", "degradation"],
    "fluids_focus": ["cum", "creampie", "squirting", "pussy juice", "saliva"],
    "general_nsfw": ["naked", "nude", "explicit", "lewd", "obscene", "sexual"]
}


def get_nsfw_tags(category: str = None) -> list:
    """Return NSFW tags. If category is given, return focused list."""
    if category and category in NSFW_CATEGORY_ROUTING:
        return NSFW_CATEGORY_ROUTING[category]
    # Return everything if no category
    all_tags = []
    for group in [ANATOMY, HAIR]:
        for tags in group.values():
            all_tags.extend(tags)
    all_tags.extend(FLUIDS)
    all_tags.extend(ACTS)
    all_tags.extend(KINKS)
    return list(set(all_tags))


def is_nsfw_tag(tag: str) -> bool:
    """Quick check if a tag is considered NSFW."""
    tag = tag.lower()
    all_nsfw = set()
    for d in [ANATOMY, HAIR]:
        for v in d.values():
            all_nsfw.update(v)
    all_nsfw.update(FLUIDS)
    all_nsfw.update(ACTS)
    all_nsfw.update(KINKS)
    return any(t in tag for t in all_nsfw)


if __name__ == "__main__":
    print("NSFW Vocabulary Module Loaded Successfully")
    print(f"Total anatomy terms: {sum(len(v) for v in ANATOMY.values())}")
    print(f"Total kinks: {len(KINKS)}")
    print("Ready to make things filthy.")
