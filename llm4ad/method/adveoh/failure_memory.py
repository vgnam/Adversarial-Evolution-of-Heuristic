from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, asdict
from threading import Lock
from typing import List, Optional


# =============================================================================
# FailureMode — a single discovered weakness
# =============================================================================

@dataclass
class FailureMode:
    """A discovered failure mode: an abstract description of a recurring weakness.

    Attributes:
        description: Human-readable textual description of the weakness
                     (e.g. "nearest-neighbor collapse under clustered structures").
        generator_id: SHA1 of the generator function that produced the defeating instance.
        heuristic_id: SHA1 of the heuristic function that was defeated.
        generator_description: The ``algorithm`` field of the generator (LLM thought).
        heuristic_description: The ``algorithm`` field of the defeated heuristic.
        generation: The heuristic generation at which this failure was discovered.
        task: Task identifier (e.g. "tsp_construct").
        performance_delta: How much worse the heuristic performed vs. optimal/baseline.
        optimal_value: The optimal or reference value for this instance.
        severity: Qualitative severity estimate (0.0 = mild, 1.0 = critical).
        strength: How consistently this weakness defeats heuristics (0..1).
                 Updated each time a different heuristic falls to it.
        coverage: How many heuristics have been verified robust against it.
        instance_seed: The seed of the specific instance that first revealed this failure.
    """
    description: str
    generator_id: str
    heuristic_id: str
    generator_description: str = ''
    heuristic_description: str = ''
    generation: int = 0
    task: str = ''
    performance_delta: float = 0.0
    optimal_value: float = 1.0
    severity: float = 0.5
    strength: float = 0.5
    coverage: int = 0
    instance_seed: int = 0

    @property
    def normalized_delta(self) -> float:
        """Difficulty: performance_delta normalised by optimal value.
        A larger value means the heuristic performed worse relative to optimal.
        """
        if self.optimal_value is None or abs(self.optimal_value) < 1e-12:
            return self.performance_delta
        return self.performance_delta / abs(self.optimal_value)

    @property
    def signature(self) -> str:
        """Deterministic key for deduplication based on the abstract description."""
        return hashlib.sha1(
            self.description.strip().lower().encode('utf-8')
        ).hexdigest()


# =============================================================================
# FailureModeMemory — persistent shared knowledge base
# =============================================================================

class FailureModeMemory:
    """Persistent storage for discovered failure modes.

    Supports:
      - Adding new failure modes with deduplication against existing ones.
      - Querying: most threatening, most recently discovered, uncovered modes.
      - Similarity checking between a new failure description and archived modes.
      - Persistence to/from JSON.
    """

    def __init__(self, path: Optional[str] = None):
        self._modes: dict[str, FailureMode] = {}
        self._lock = Lock()
        self._path = path
        if path and os.path.isfile(path):
            self._load()

    # ------------------------------------------------------------------
    #  Core API
    # ------------------------------------------------------------------

    def add(self, mode: FailureMode) -> bool:
        """Add a failure mode, deduplicating by signature.

        Returns True if a new mode was added, False if it merged with an
        existing one.
        """
        sig = mode.signature
        with self._lock:
            existing = self._modes.get(sig)
            if existing is not None:
                # Merge: accumulate strength, bump coverage if a different heuristic
                if mode.heuristic_id != existing.heuristic_id:
                    existing.strength = min(1.0, existing.strength + 0.1)
                existing.severity = max(existing.severity, mode.severity)
                existing.generation = max(existing.generation, mode.generation)
                if mode.performance_delta > existing.performance_delta:
                    existing.performance_delta = mode.performance_delta
                    existing.generator_id = mode.generator_id
                    existing.heuristic_id = mode.heuristic_id
                self._save()
                return False

            self._modes[sig] = mode
            self._save()
            return True

    def mark_covered(self, failure_sig: str) -> None:
        """Record that a heuristic successfully survived this failure mode."""
        with self._lock:
            mode = self._modes.get(failure_sig)
            if mode is not None:
                mode.coverage += 1
                mode.strength = max(0.0, mode.strength - 0.05)
                self._save()

    def mark_uncovered(self, failure_sig: str) -> None:
        """Record that another heuristic fell to this failure mode."""
        with self._lock:
            mode = self._modes.get(failure_sig)
            if mode is not None:
                mode.strength = min(1.0, mode.strength + 0.05)
                self._save()

    # ------------------------------------------------------------------
    #  Queries
    # ------------------------------------------------------------------

    def most_threatening(self, k: int = 5) -> List[FailureMode]:
        """Return the k failure modes with highest strength (most consistently
        defeat heuristics)."""
        with self._lock:
            sorted_modes = sorted(
                self._modes.values(),
                key=lambda m: (m.strength, m.normalized_delta),
                reverse=True,
            )
            return list(sorted_modes[:k])

    def most_recent(self, k: int = 5) -> List[FailureMode]:
        """Return the k most recently discovered failure modes."""
        with self._lock:
            sorted_modes = sorted(
                self._modes.values(),
                key=lambda m: m.generation,
                reverse=True,
            )
            return list(sorted_modes[:k])

    def uncovered(self, min_strength: float = 0.3) -> List[FailureMode]:
        """Return failure modes with zero coverage and strength above threshold."""
        with self._lock:
            return [
                m for m in self._modes.values()
                if m.coverage == 0 and m.strength >= min_strength
            ]

    def get_all(self) -> List[FailureMode]:
        with self._lock:
            return list(self._modes.values())

    def get_by_signature(self, sig: str) -> Optional[FailureMode]:
        with self._lock:
            return self._modes.get(sig)

    def __len__(self) -> int:
        with self._lock:
            return len(self._modes)

    def __bool__(self) -> bool:
        return len(self) > 0

    # ------------------------------------------------------------------
    #  Similarity
    # ------------------------------------------------------------------

    def max_similarity(self, description: str) -> float:
        """Compute the maximum similarity between a new failure description
        and all archived failure modes.

        Uses simple token-overlap (Jaccard) on lowercased, non-stopword tokens.
        This is lightweight and does not require an LLM call.
        """
        tokens = self._tokenize(description)
        if not tokens:
            return 0.0

        max_sim = 0.0
        with self._lock:
            for mode in self._modes.values():
                mode_tokens = self._tokenize(mode.description)
                if not mode_tokens:
                    continue
                intersection = len(tokens & mode_tokens)
                union = len(tokens | mode_tokens)
                sim = intersection / union if union > 0 else 0.0
                if sim > max_sim:
                    max_sim = sim
        return max_sim

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Lowercase, split on non-alphanumeric, remove short/stopword tokens."""
        _STOPWORDS = frozenset({
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'just', 'because', 'but', 'and', 'or', 'if', 'while',
            'that', 'this', 'these', 'those', 'it', 'its', 'which', 'who',
            'whom', 'what',
        })
        raw = re.findall(r'[a-z]+', text.lower())
        return {t for t in raw if len(t) > 2 and t not in _STOPWORDS}

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            with open(self._path, 'r') as f:
                data = json.load(f)
            for item in data:
                mode = FailureMode(**item)
                self._modes[mode.signature] = mode
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save(self) -> None:
        if not self._path:
            return
        os.makedirs(os.path.dirname(self._path) or '.', exist_ok=True)
        data = [asdict(m) for m in self._modes.values()]
        with open(self._path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_to(self, path: str) -> None:
        self._path = path
        self._save()

    # ------------------------------------------------------------------
    #  Convenience
    # ------------------------------------------------------------------

    def get_descriptions(self, k: int = 3) -> List[str]:
        """Return the k most threatening failure descriptions for prompt injection."""
        return [m.description for m in self.most_threatening(k)]

    def novelty_vs_memory(self, description: str) -> float:
        """Novelty score: 1.0 = completely novel, 0.0 = identical to archived.

        Used in the generator reward: ``novelty_penalty = novelty_vs_memory(...)``
        """
        return 1.0 - self.max_similarity(description)

    def coverage_fraction(self, heuristic_survived_sigs: set[str]) -> float:
        """Fraction of archived failure modes that a heuristic has survived.

        Only considers modes above a minimum strength threshold.
        """
        all_modes = self.get_all()
        if not all_modes:
            return 1.0
        relevant = [m for m in all_modes if m.strength >= 0.2]
        if not relevant:
            return 1.0
        covered = sum(1 for m in relevant if m.signature in heuristic_survived_sigs)
        return covered / len(relevant)


# Lazily import re at module level (used by _tokenize)
import re  # noqa: E402
