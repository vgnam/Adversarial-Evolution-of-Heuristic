from __future__ import annotations

import math
from typing import Any, List, Optional

from .failure_memory import FailureMode


class FailureAnalyzer:
    """Analyzes evaluation results to extract failure-mode descriptions.

    Input: A generated problem instance, the heuristic's attempted solution,
    the ground truth / optimal solution, and the performance delta.

    Processing: Uses description-based analysis to characterize:
    - What type of structure caused the failure (clustering, symmetry, sparsity,
      deception)
    - Qualitative description of the weakness
    - Estimated severity

    Output: A structured FailureMode object.

    This implementation uses heuristic rules based on instance statistics and
    performance data, which is task-agnostic and does not require an LLM call.
    The ``refine_with_llm`` method can optionally use an LLM for richer
    descriptions when one is available.
    """

    # Pattern templates indexed by weakness category.
    # The ``{stat_desc}`` placeholder is filled with numerical context.
    PATTERNS: dict[str, str] = {
        'clustering': (
            "Nearest-neighbor collapse under clustered structure: "
            "instances group points into dense clusters, causing greedy "
            "construction heuristics to make locally optimal but globally "
            "poor choices when bridging clusters. {stat_desc}"
        ),
        'symmetry': (
            "Symmetry-induced oscillation: instances exhibit near-identical "
            "alternatives at critical decision points, causing heuristic "
            "selection to oscillate between near-equal options. {stat_desc}"
        ),
        'sparsity': (
            "Exploration failure under sparse hub configuration: instances "
            "place nodes in a hub-and-spoke layout where long edges connect "
            "tight clusters, making myopic distance-based selection fail. "
            "{stat_desc}"
        ),
        'deceptive': (
            "Deceptive local-optimum trap: instances are structured so that "
            "locally optimal decisions at early steps lead to globally poor "
            "outcomes that cannot be recovered from. {stat_desc}"
        ),
        'uniform': (
            "Near-uniform random structure that defeats heuristic by "
            "lack of exploitable patterns: heuristic overfits to specific "
            "structural assumptions that random instances violate. "
            "{stat_desc}"
        ),
        'scale': (
            "Scale-induced decision paralysis: instances with widely varying "
            "magnitudes cause threshold-based heuristics to make poor "
            "normalisation or scaling decisions. {stat_desc}"
        ),
        'adversarial': (
            "Explicitly adversarial structure: instance contains purpose-built "
            "patterns (corridors, dead ends, forced detours) that exploit "
            "the heuristic's deterministic tie-breaking rule. {stat_desc}"
        ),
    }

    CATEGORY_ORDER: list[str] = [
        'adversarial', 'deceptive', 'clustering', 'symmetry',
        'sparsity', 'scale', 'uniform',
    ]

    def __init__(self, task: str = ''):
        self._task = task

    # ================================================================
    #  Main analysis entry point
    # ================================================================

    def analyze(
        self,
        instance: Any,
        heuristic_score: float,
        optimal_value: Optional[float] = None,
        generator_description: str = '',
        heuristic_description: str = '',
        generator_id: str = '',
        heuristic_id: str = '',
        generation: int = 0,
        instance_seed: int = 0,
    ) -> Optional[FailureMode]:
        """Analyze a heuristic failure and produce a structured FailureMode.

        Args:
            instance: The problem instance (task-native format).
            heuristic_score: Score the heuristic achieved on this instance.
            optimal_value: Optimal/reference score for this instance, if known.
            generator_description: ``algorithm`` field of the generator.
            heuristic_description: ``algorithm`` field of the heuristic.
            generator_id: SHA1 of the generator function.
            heuristic_id: SHA1 of the heuristic function.
            generation: Current heuristic generation.
            instance_seed: Seed used to generate this instance.

        Returns:
            A FailureMode if the heuristic performed significantly worse
            than optimal, or None if performance was acceptable.
        """
        if heuristic_score is None:
            return None

        delta, optimal = self._compute_delta(heuristic_score, optimal_value)
        if delta <= 0:
            return None

        severity = self._estimate_severity(delta, optimal)
        category, stat_desc = self._categorize_failure(instance, heuristic_score, optimal)

        description = self._build_description(category, stat_desc)

        return FailureMode(
            description=description,
            generator_id=generator_id,
            heuristic_id=heuristic_id,
            generator_description=generator_description,
            heuristic_description=heuristic_description,
            generation=generation,
            task=self._task,
            performance_delta=delta,
            optimal_value=optimal or 1.0,
            severity=severity,
            strength=0.5,  # initial; adjusted via memory updates
            coverage=0,
            instance_seed=instance_seed,
        )

    # ================================================================
    #  Delta and severity
    # ================================================================

    @staticmethod
    def _compute_delta(
        heuristic_score: float,
        optimal_value: Optional[float],
    ) -> tuple[float, Optional[float]]:
        """Compute how much worse the heuristic performed vs optimal.

        Returns (delta, optimal) where delta > 0 means heuristic was worse.
        If no optimal value is known, a heuristic-based estimate is used.
        """
        if optimal_value is not None and abs(optimal_value) > 1e-12:
            delta = max(0.0, optimal_value - heuristic_score)
            return delta, optimal_value

        # Without optimal, use heuristic_score magnitude as baseline.
        # Scores are typically non-negative and higher-is-better.
        # A score near 0 or negative indicates failure.
        if heuristic_score < 0:
            return abs(heuristic_score), 0.0
        # If positive but small relative to typical values, treat as mild failure.
        return max(0.0, 1.0 - heuristic_score), 1.0

    @staticmethod
    def _estimate_severity(delta: float, optimal: Optional[float]) -> float:
        """Map performance delta to a severity score in [0, 1]."""
        if optimal is None or abs(optimal) < 1e-12:
            return min(1.0, delta / 10.0)

        ratio = delta / abs(optimal)
        if ratio >= 1.0:
            return 1.0
        if ratio >= 0.5:
            return 0.8
        if ratio >= 0.25:
            return 0.5
        if ratio >= 0.1:
            return 0.3
        return 0.1

    # ================================================================
    #  Failure categorization
    # ================================================================

    @staticmethod
    def _extract_instance_stats(instance: Any) -> dict:
        """Extract structural statistics from an instance for failure analysis.

        This is task-agnostic: works with numpy arrays, tuples of arrays,
        and dict-like structures.
        """
        stats = {}

        try:
            import numpy as np

            # Helper to extract coordinates from various instance formats
            def _to_float_array(data) -> Optional[np.ndarray]:
                if data is None:
                    return None
                if isinstance(data, np.ndarray):
                    return data
                if isinstance(data, (tuple, list)):
                    # Try first element if it's a tuple of arrays
                    for item in data:
                        arr = np.asarray(item)
                        if arr.ndim >= 1 and arr.size > 0:
                            return arr
                return None

            arr = _to_float_array(instance)
            if arr is None:
                return stats

            flat = arr.ravel()
            stats['n_elements'] = flat.size
            stats['n_nan'] = int(np.isnan(flat).sum())
            stats['n_inf'] = int(np.isinf(flat).sum())
            stats['finite'] = np.isfinite(flat)
            finite_vals = flat[stats['finite']]

            if finite_vals.size > 0:
                stats['min'] = float(finite_vals.min())
                stats['max'] = float(finite_vals.max())
                stats['mean'] = float(finite_vals.mean())
                stats['std'] = float(finite_vals.std())
                stats['range'] = stats['max'] - stats['min']

                # Coefficient of variation (normalised dispersion)
                if abs(stats['mean']) > 1e-12:
                    stats['cv'] = stats['std'] / abs(stats['mean'])
                else:
                    stats['cv'] = stats['std']

                # For 2D coordinates, compute clustering metrics
                if arr.ndim == 2 and arr.shape[1] == 2 and finite_vals.size >= 4:
                    coords = arr[~np.isnan(arr).any(axis=1) & ~np.isinf(arr).any(axis=1)]
                    if coords.shape[0] >= 4:
                        # Pairwise distances
                        from scipy.spatial.distance import pdist  # noqa
                        try:
                            dists = pdist(coords)
                            stats['mean_pairwise_dist'] = float(dists.mean())
                            stats['std_pairwise_dist'] = float(dists.std())
                            stats['cv_pairwise'] = (
                                stats['std_pairwise_dist'] / stats['mean_pairwise_dist']
                                if stats['mean_pairwise_dist'] > 1e-12 else 0
                            )
                            # Skewness of nearest-neighbour distances
                            from scipy.spatial import KDTree  # noqa
                            tree = KDTree(coords)
                            nn_dists, _ = tree.query(coords, k=2)
                            nn_dists = nn_dists[:, 1]  # exclude self
                            if nn_dists.std() > 0:
                                stats['nn_skew'] = float(
                                    np.mean(((nn_dists - nn_dists.mean()) / nn_dists.std()) ** 3)
                                )
                            else:
                                stats['nn_skew'] = 0.0
                        except Exception:
                            pass
        except ImportError:
            pass

        return stats

    def _categorize_failure(
        self,
        instance: Any,
        heuristic_score: float,
        optimal_value: Optional[float],
    ) -> tuple[str, str]:
        """Determine the most likely failure category and a statistical
        description string.

        Returns (category_key, stat_description).
        """
        stats = self._extract_instance_stats(instance)

        # Fallback if we have no instance stats to analyse
        if not stats:
            return 'adversarial', 'instance structure not analyzable with current statistics'

        stat_parts = []

        # --- Clustering detection: high CV of pairwise distances, negative NN skew ---
        cv_pairwise = stats.get('cv_pairwise', 0)
        nn_skew = stats.get('nn_skew', 0)
        if cv_pairwise > 0.5 and nn_skew < -0.5:
            stat_parts.append(
                f'high pairwise distance variation (CV={cv_pairwise:.2f}) '
                f'with negative NN skew ({nn_skew:.2f})'
            )
            stat_desc = '; '.join(stat_parts) if stat_parts else 'statistical clustering indicators'
            return 'clustering', stat_desc

        # --- Symmetry detection: low CV of pairwise distances + many near-equal NN ---
        if cv_pairwise < 0.2 and nn_skew is not None and abs(nn_skew) < 0.3:
            stat_parts.append(
                f'low pairwise distance variation (CV={cv_pairwise:.2f}) '
                f'with symmetric NN distribution (skew={nn_skew:.2f})'
            )
            stat_desc = '; '.join(stat_parts) if stat_parts else 'statistical symmetry indicators'
            return 'symmetry', stat_desc

        # --- Sparsity detection: high CV, positive NN skew ---
        if cv_pairwise > 0.5 and nn_skew > 0.5:
            stat_parts.append(
                f'high pairwise distance variation (CV={cv_pairwise:.2f}) '
                f'with positive NN skew ({nn_skew:.2f})'
            )
            stat_desc = '; '.join(stat_parts) if stat_parts else 'sparse hub-like structure'
            return 'sparsity', stat_desc

        # --- Scale detection: high coefficient of variation in values ---
        cv_val = stats.get('cv', 0)
        if cv_val > 2.0:
            stat_parts.append(
                f'extreme value range (min={stats.get("min", 0):.3f}, '
                f'max={stats.get("max", 0):.3f}, CV={cv_val:.2f})'
            )
            stat_desc = '; '.join(stat_parts)
            return 'scale', stat_desc

        # --- Deception: large performance gap with moderate structural variation ---
        delta, optimal = self._compute_delta(heuristic_score, optimal_value)
        ratio = delta / abs(optimal) if optimal and abs(optimal) > 1e-12 else delta
        if ratio > 0.5:
            stat_parts.append(
                f'large performance gap (heuristic={heuristic_score:.4f}, '
                f'optimal={optimal_value if optimal_value else "N/A"})'
            )
            stat_desc = '; '.join(stat_parts) if stat_parts else 'disproportionate performance gap'
            return 'deceptive', stat_desc

        # --- Default: near-uniform ---
        if stat_parts:
            stat_desc = '; '.join(stat_parts)
        else:
            stat_desc = (
                f'values in [{stats.get("min", 0):.3f}, {stats.get("max", 0):.3f}], '
                f'std={stats.get("std", 0):.3f}'
            )
        return 'uniform', stat_desc

    # ================================================================
    #  Description building
    # ================================================================

    def _build_description(self, category: str, stat_desc: str) -> str:
        pattern = self.PATTERNS.get(category, self.PATTERNS['adversarial'])
        return pattern.format(stat_desc=stat_desc)

    # ================================================================
    #  Convenience: batch analysis
    # ================================================================

    def analyze_batch(
        self,
        evaluation_results: List[dict],
        generation: int,
    ) -> List[FailureMode]:
        """Analyze a batch of evaluation results.

        Each entry in evaluation_results should be a dict with keys:
          - instance: the instance data
          - heuristic_score: score the heuristic achieved
          - optimal_value: optimal/reference score (optional)
          - generator_description (optional)
          - heuristic_description (optional)
          - generator_id (optional)
          - heuristic_id (optional)
          - instance_seed (optional)
        """
        modes = []
        for result in evaluation_results:
            mode = self.analyze(
                instance=result.get('instance'),
                heuristic_score=result.get('heuristic_score'),
                optimal_value=result.get('optimal_value'),
                generator_description=result.get('generator_description', ''),
                heuristic_description=result.get('heuristic_description', ''),
                generator_id=result.get('generator_id', ''),
                heuristic_id=result.get('heuristic_id', ''),
                generation=generation,
                instance_seed=result.get('instance_seed', 0),
            )
            if mode is not None:
                modes.append(mode)
        return modes
