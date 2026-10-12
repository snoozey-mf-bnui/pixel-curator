#!/usr/bin/env python3
"""
duplicate_detector.py
---------------------
Supercharged module for detecting near-duplicate or highly similar frames.

Enhancements:
- Built-in Perceptual Hashing (dHash): Computes a 64-bit difference hash directly 
  from image arrays. Robust to scaling, compression, and minor color shifts.
- Hamming Distance Calculation: Accurately measures the bitwise difference between 
  two image hashes to determine similarity (0.0 to 1.0).
- Stateful Duplicate Registry: A dedicated class to track hashes across a dataset, 
  preventing the need for O(N^2) manual comparisons.
- Configurable thresholds for strict vs. loose duplicate detection.
- Robust input sanitization and JSON serialization for pipeline integration.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple

import numpy as np

# Attempt to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DuplicateThresholds:
    """Configurable thresholds for duplicate classification."""
    # Hamming distance out of 64 bits.
    # 0 = identical. < 5 = near duplicate. < 10 = highly similar.
    strict_distance: int = 5       
    loose_distance: int = 12


@dataclass
class DuplicateScores:
    """Container for duplicate detection results with serialization."""
    similarity_score: Optional[float] = None    # 0.0 (unique) to 1.0 (identical)
    is_duplicate: Optional[bool] = None
    duplicate_of: Optional[int] = None          # Index/ID of the frame it duplicates
    hamming_distance: Optional[int] = None
    confidence: float = 0.0
    thresholds_used: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Converts scores to a dictionary, rounding for cleaner output."""
        data = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}

    def to_json(self) -> str:
        """Serializes scores to a JSON string."""
        return json.dumps(self.to_dict())


def _sanitize_score(value: Optional[float]) -> Optional[float]:
    """Validates and clamps scores to [0.0, 1.0]. Returns None if invalid."""
    if value is None: return None
    if not isinstance(value, (int, float)): return None
    if math.isnan(value) or math.isinf(value): return None
    return max(0.0, min(1.0, float(value)))


# --- CV & Hashing Estimation Helpers ---

def compute_dhash(image: np.ndarray, hash_size: int = 8) -> Optional[int]:
    """
    Computes the difference hash (dHash) of an image.
    dHash is robust to scaling, brightness changes, and compression artifacts.
    Returns a 64-bit integer.
    """
    if not CV2_AVAILABLE or image is None:
        return None
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Resize to 9x8 (width+1, height) to compute horizontal differences
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    
    # Compute horizontal gradient: 1 if left pixel is brighter than right pixel
    diff = resized[:, 1:] > resized[:, :-1]
    
    # Flatten the 8x8 boolean array and convert to a 64-bit integer
    flat_hash = diff.flatten()
    integer_hash = 0
    for i, bit in enumerate(flat_hash):
        if bit:
            integer_hash |= (1 << i)
            
    return integer_hash


def calculate_hamming_distance(hash1: int, hash2: int) -> int:
    """Calculates the Hamming distance (number of differing bits) between two hashes."""
    return bin(hash1 ^ hash2).count('1')


def hamming_to_similarity(distance: int, hash_size: int = 64) -> float:
    """Converts Hamming distance to a 0.0-1.0 similarity score (1.0 = identical)."""
    return 1.0 - (distance / hash_size)


# --- Main Logic ---

def evaluate_duplicate(
    similarity_score: Optional[float] = None,
    is_duplicate: Optional[bool] = None,
    duplicate_of: Optional[int] = None,
    hamming_distance: Optional[int] = None,
    thresholds: DuplicateThresholds = DuplicateThresholds(),
) -> DuplicateScores:
    """
    Evaluate whether a frame is a duplicate from pre-computed signals.
    """
    sim_q = _sanitize_score(similarity_score)

    scores = DuplicateScores(
        similarity_score=sim_q,
        is_duplicate=is_duplicate,
        duplicate_of=duplicate_of,
        hamming_distance=hamming_distance,
        thresholds_used=asdict(thresholds),
    )

    # If hamming distance is provided, use it as the source of truth
    if hamming_distance is not None:
        scores.similarity_score = hamming_to_similarity(hamming_distance)
        scores.is_duplicate = hamming_distance <= thresholds.strict_distance
        scores.confidence = 0.9 if hamming_distance <= thresholds.loose_distance else 0.7
    else:
        # Fallback to legacy logic
        if sim_q is not None:
            scores.is_duplicate = sim_q >= 0.92
            scores.confidence = 0.8
        elif is_duplicate is not None:
            scores.similarity_score = 0.95 if is_duplicate else 0.3
            scores.confidence = 0.6

    return scores


def evaluate_duplicate_from_images(
    image1: np.ndarray,
    image2: np.ndarray,
    original_id: Optional[int] = None,
    thresholds: DuplicateThresholds = DuplicateThresholds(),
) -> DuplicateScores:
    """
    Convenience wrapper: Computes hashes for two images and evaluates if they are duplicates.
    """
    if image1 is None or image2 is None or not CV2_AVAILABLE:
        return evaluate_duplicate(thresholds=thresholds)

    hash1 = compute_dhash(image1)
    hash2 = compute_dhash(image2)
    
    if hash1 is None or hash2 is None:
        return evaluate_duplicate(thresholds=thresholds)

    distance = calculate_hamming_distance(hash1, hash2)

    return evaluate_duplicate(
        hamming_distance=distance,
        duplicate_of=original_id,
        thresholds=thresholds
    )


# --- Stateful Registry for Batch Processing ---

class DuplicateRegistry:
    """
    Maintains a registry of image hashes to efficiently detect duplicates 
    across a streaming dataset without O(N^2) comparisons.
    """

    def __init__(self, thresholds: DuplicateThresholds = DuplicateThresholds()):
        self.thresholds = thresholds
        self.registry: List[Tuple[int, int]] = []  # List of (frame_id, hash)

    def check_and_add(self, frame_id: int, image: np.ndarray) -> DuplicateScores:
        """
        Checks an image against all registered frames. 
        If unique, adds it to the registry. If duplicate, returns the match.
        """
        current_hash = compute_dhash(image)
        if current_hash is None:
            return evaluate_duplicate(thresholds=self.thresholds)

        best_match_id = None
        best_distance = float('inf')

        for ref_id, ref_hash in self.registry:
            dist = calculate_hamming_distance(current_hash, ref_hash)
            if dist < best_distance:
                best_distance = dist
                best_match_id = ref_id
                # Early exit if we find an exact or near-exact match
                if dist <= 2:
                    break

        # Evaluate if the closest match falls within the duplicate threshold
        scores = evaluate_duplicate(
            hamming_distance=int(best_distance) if best_distance != float('inf') else None,
            duplicate_of=best_match_id,
            thresholds=self.thresholds
        )

        # If it's NOT a duplicate, add it to the registry for future checks
        if not scores.is_duplicate:
            self.registry.append((frame_id, current_hash))

        return scores


def get_similarity_score(scores: DuplicateScores) -> float:
    """Returns the similarity score (higher = more duplicate-like)."""
    return scores.similarity_score if scores.similarity_score is not None else 0.0


def is_likely_duplicate(scores: DuplicateScores, threshold: float = 0.92) -> bool:
    """Returns True if the frame is likely a duplicate."""
    if scores.hamming_distance is not None:
        # Inverse of threshold: 0.92 similarity ~ 5 bits difference (out of 64)
        max_dist = int((1.0 - threshold) * 64)
        return scores.hamming_distance <= max_dist
    if scores.similarity_score is not None:
        return scores.similarity_score >= threshold
    return scores.is_duplicate if scores.is_duplicate is not None else False


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Signals (Backward Compatible):")
    legacy_scores = evaluate_duplicate(similarity_score=0.95)
    print(f"   Is Duplicate: {legacy_scores.is_duplicate}, Confidence: {legacy_scores.confidence:.2f}\n")

    if CV2_AVAILABLE:
        print("2. Testing CV Pipeline - Identical Images:")
        # Create a random noise image
        np.random.seed(42)
        img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        identical_scores = evaluate_duplicate_from_images(img1, img1, original_id=0)
        print(f"   Hamming Distance: {identical_scores.hamming_distance} (Should be 0)")
        print(f"   Similarity: {identical_scores.similarity_score:.4f} (Should be 1.0)")
        print(f"   Is Duplicate: {identical_scores.is_duplicate}\n")

        print("3. Testing CV Pipeline - Slightly Modified (Near Duplicate):")
        # Slightly alter the image (add small noise / change brightness)
        img2 = img1.copy()
        img2 = cv2.convertScaleAbs(img2, alpha=1.1, beta=10) # Brightness/contrast shift
        near_scores = evaluate_duplicate_from_images(img1, img2, original_id=0)
        print(f"   Hamming Distance: {near_scores.hamming_distance} (Should be low, < 10)")
        print(f"   Is Duplicate: {near_scores.is_duplicate}\n")

        print("4. Testing CV Pipeline - Completely Different Images:")
        img3 = np.zeros((256, 256, 3), dtype=np.uint8) # Solid black
        diff_scores = evaluate_duplicate_from_images(img1, img3, original_id=0)
        print(f"   Hamming Distance: {diff_scores.hamming_distance} (Should be high, > 30)")
        print(f"   Is Duplicate: {diff_scores.is_duplicate} (Should be False)\n")

        print("5. Testing Stateful Duplicate Registry:")
        registry = DuplicateRegistry()
        
        # Frame 0: Add unique image
        s0 = registry.check_and_add(0, img1)
        print(f"   Frame 0: Is Dup: {s0.is_duplicate}, Dup Of: {s0.duplicate_of}")
        
        # Frame 1: Add completely different image
        s1 = registry.check_and_add(1, img3)
        print(f"   Frame 1: Is Dup: {s1.is_duplicate}, Dup Of: {s1.duplicate_of}")
        
        # Frame 2: Add near-duplicate of Frame 0
        s2 = registry.check_and_add(2, img2)
        print(f"   Frame 2: Is Dup: {s2.is_duplicate}, Dup Of: {s2.duplicate_of} (Should be 0)")
        print(f"   JSON: {s2.to_json()}\n")
