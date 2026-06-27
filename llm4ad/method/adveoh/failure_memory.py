from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from threading import Lock
from typing import List, Optional

import numpy as np


# =============================================================================
# FailureMode — a single discovered weakness
# =============================================================================

@dataclass
class FailureMode:
    """A discovered failure mode: an abstract description of a recurring weakness.

    Attributes:
        description: LLM-generated natural-language description of the weakness.
        generator_id: SHA1 of the generator that produced the defeating instance.
        heuristic_id: SHA1 of the heuristic that was defeated.
        generator_description: The ``algorithm`` field of the generator.
        heuristic_description: The ``algorithm`` field of the defeated heuristic.
        generation: Heuristic generation at which this failure was discovered.
        task: Task identifier (e.g. "tsp_construct").
        performance_delta: How much worse the heuristic performed vs. optimal.
        optimal_value: Optimal or reference value for this instance.
        severity: Qualitative severity in [0, 1].
        strength: How consistently this weakness defeats heuristics (0..1).
                  Updated each time a different heuristic falls to it.
        coverage: How many heuristics have been verified robust against it.
        instance_seed: Seed of the specific instance that first revealed this failure.
        embedding: Cached sentence embedding for cosine similarity (not persisted).
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

    # Not persisted — recomputed on load when needed
    embedding: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def normalized_delta(self) -> float:
        if self.optimal_value is None or abs(self.optimal_value) < 1e-12:
            return self.performance_delta
        return self.performance_delta / abs(self.optimal_value)

    @property
    def signature(self) -> str:
        """Deterministic key for deduplication based on the abstract description."""
        return hashlib.sha1(
            self.description.strip().lower().encode('utf-8')
        ).hexdigest()

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop('embedding', None)  # never persist the numpy array
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'FailureMode':
        d.pop('embedding', None)
        return cls(**d)


# =============================================================================
# EmbeddingIndex — lazy singleton for sentence embeddings
# =============================================================================

class _EmbeddingIndex:
    """Singleton wrapper around a sentence-transformer model.

    Lazy-loaded on first use so that importing failure_memory never blocks.
    Falls back gracefully to Jaccard if sentence-transformers is unavailable.
    """
    _instance: Optional['_EmbeddingIndex'] = None
    _lock = Lock()
    MODEL_NAME = 'all-MiniLM-L6-v2'

    def __init__(self):
        self._model = None
        self._available = None

    @classmethod
    def get(cls) -> '_EmbeddingIndex':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_model(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from sentence_transformers import SentenceTransformer  # noqa
            self._model = SentenceTransformer(self.MODEL_NAME)
            self._available = True
        except Exception:
            self._available = False
        return self._available

    def encode(self, text: str) -> Optional[np.ndarray]:
        if not self._ensure_model():
            return None
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        # Both should already be L2-normalised by encode()
        return float(np.clip(np.dot(a, b), -1.0, 1.0))


# =============================================================================
# FailureModeMemory — persistent shared knowledge base
# =============================================================================

class FailureModeMemory:
    """Persistent storage for discovered failure modes.

    Similarity is measured via sentence-embedding cosine distance (falls back
    to Jaccard token overlap when sentence-transformers is unavailable).

    Coverage is only credited when a heuristic is evaluated on an instance
    produced by the same generator_id that originally triggered the failure
    mode (see ``try_mark_covered``).
    """

    # Cosine similarity threshold above which two descriptions are considered
    # "the same" failure mode and merged rather than stored separately.
    MERGE_THRESHOLD: float = 0.85

    def __init__(self, path: Optional[str] = None):
        self._modes: dict[str, FailureMode] = {}
        self._lock = Lock()
        self._path = path
        self._emb = _EmbeddingIndex.get()
        if path and os.path.isfile(path):
            self._load()

    # ------------------------------------------------------------------
    #  Core API
    # ------------------------------------------------------------------

    def add(self, mode: FailureMode) -> bool:
        """Add a failure mode.

        Deduplication strategy (two-level):
        1. Exact match on signature (hash of lowercased description) → merge.
        2. Semantic near-duplicate via embedding cosine ≥ MERGE_THRESHOLD → merge.

        Returns True if a genuinely new mode was added.
        """
        sig = mode.signature

        # Compute embedding once before acquiring lock
        emb = self._emb.encode(mode.description)
        mode.embedding = emb

        with self._lock:
            # --- Level 1: exact signature match ---
            if sig in self._modes:
                self._merge(self._modes[sig], mode)
                self._save()
                return False

            # --- Level 2: semantic near-duplicate ---
            if emb is not None:
                for existing in self._modes.values():
                    if existing.embedding is None:
                        existing.embedding = self._emb.encode(existing.description)
                    if existing.embedding is not None:
                        sim = self._emb.cosine(emb, existing.embedding)
                        if sim >= self.MERGE_THRESHOLD:
                            self._merge(existing, mode)
                            self._save()
                            return False

            self._modes[sig] = mode
            self._save()
            return True

    @staticmethod
    def _merge(existing: FailureMode, incoming: FailureMode) -> None:
        """Merge ``incoming`` into ``existing`` in-place."""
        if incoming.heuristic_id != existing.heuristic_id:
            existing.strength = min(1.0, existing.strength + 0.1)
        existing.severity = max(existing.severity, incoming.severity)
        existing.generation = max(existing.generation, incoming.generation)
        if incoming.performance_delta > existing.performance_delta:
            existing.performance_delta = incoming.performance_delta
            existing.generator_id = incoming.generator_id
            existing.heuristic_id = incoming.heuristic_id

    # ------------------------------------------------------------------
    #  Coverage — generator_id gated (Fix 3)
    # ------------------------------------------------------------------

    def try_mark_covered(
        self,
        heu_func_id: str,
        generator_id: str,
        heuristic_score: float,
        optimal_value: Optional[float],
        min_delta: float = 0.05,
    ) -> set[str]:
        """Credit a heuristic with surviving failure modes IF the instance
        was produced by the same generator that originally triggered each mode.

        Args:
            heu_func_id: SHA1 of the heuristic function.
            generator_id: SHA1 of the generator that produced the current instance.
            heuristic_score: Score the heuristic achieved on this instance.
            optimal_value: Reference/optimal score, if known.
            min_delta: Maximum allowed (optimal - score)/optimal to count as "surviving".

        Returns:
            Set of failure signatures newly credited to this heuristic.
        """
        newly_covered: set[str] = set()

        # Determine if heuristic performed well enough to "survive"
        if optimal_value is not None and abs(optimal_value) > 1e-12:
            ratio = (optimal_value - heuristic_score) / abs(optimal_value)
            survived = ratio < min_delta
        else:
            survived = heuristic_score > 0.5  # fallback heuristic

        if not survived:
            return newly_covered

        with self._lock:
            for sig, mode in self._modes.items():
                # KEY FIX: only credit if this instance came from the same generator
                if mode.generator_id != generator_id:
                    continue
                newly_covered.add(sig)
                mode.coverage += 1
                mode.strength = max(0.0, mode.strength - 0.05)

        if newly_covered:
            self._save()
        return newly_covered

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
        with self._lock:
            return sorted(
                self._modes.values(),
                key=lambda m: (m.strength, m.normalized_delta),
                reverse=True,
            )[:k]

    def most_recent(self, k: int = 5) -> List[FailureMode]:
        with self._lock:
            return sorted(
                self._modes.values(),
                key=lambda m: m.generation,
                reverse=True,
            )[:k]

    def uncovered(self, min_strength: float = 0.3) -> List[FailureMode]:
        with self._lock:
            return [
                m for m in self._modes.values()
                if m.coverage == 0 and m.strength >= min_strength
            ]

    def get_all(self) -> List[FailureMode]:
        with self._lock:
            return list(self._modes.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._modes)

    def __bool__(self) -> bool:
        return len(self) > 0

    # ------------------------------------------------------------------
    #  Similarity (Fix 2: embedding-based)
    # ------------------------------------------------------------------

    def max_similarity(self, description: str) -> float:
        """Maximum cosine similarity between description and all archived modes.

        Falls back to Jaccard token overlap if embeddings are unavailable.
        """
        emb = self._emb.encode(description)

        with self._lock:
            if not self._modes:
                return 0.0

            if emb is not None:
                # Embedding path
                max_sim = 0.0
                for mode in self._modes.values():
                    if mode.embedding is None:
                        mode.embedding = self._emb.encode(mode.description)
                    if mode.embedding is not None:
                        sim = self._emb.cosine(emb, mode.embedding)
                        if sim > max_sim:
                            max_sim = sim
                return max_sim
            else:
                # Fallback: Jaccard token overlap
                tokens = self._tokenize(description)
                if not tokens:
                    return 0.0
                max_sim = 0.0
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

    def novelty_vs_memory(self, description: str) -> float:
        """Novelty = 1 - max_similarity. Used in generator reward."""
        return 1.0 - self.max_similarity(description)

    def coverage_fraction(self, heuristic_survived_sigs: set[str]) -> float:
        """Fraction of archived failure modes (above min strength) survived."""
        all_modes = self.get_all()
        if not all_modes:
            return 1.0
        relevant = [m for m in all_modes if m.strength >= 0.2]
        if not relevant:
            return 1.0
        covered = sum(1 for m in relevant if m.signature in heuristic_survived_sigs)
        return covered / len(relevant)

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            with open(self._path, 'r') as f:
                data = json.load(f)
            for item in data:
                mode = FailureMode.from_dict(item)
                self._modes[mode.signature] = mode
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            pass

    def _save(self) -> None:
        if not self._path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        data = [m.to_dict() for m in self._modes.values()]
        tmp = self._path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self._path)  # atomic write

    def save_to(self, path: str) -> None:
        self._path = path
        self._save()

    # ------------------------------------------------------------------
    #  Convenience
    # ------------------------------------------------------------------

    def get_descriptions(self, k: int = 3) -> List[str]:
        return [m.description for m in self.most_threatening(k)]

    # ------------------------------------------------------------------
    #  Jaccard fallback (used only when embeddings unavailable)
    # ------------------------------------------------------------------

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

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        import re
        raw = re.findall(r'[a-z]+', text.lower())
        return {t for t in raw if len(t) > 2 and t not in cls._STOPWORDS}