from __future__ import annotations

import json
import textwrap
from typing import Any, List, Optional

from .failure_memory import FailureMode


# =============================================================================
# FailureAnalyzer — LLM-primary failure description generator
# =============================================================================

class FailureAnalyzer:
    """Analyzes evaluation results and produces LLM-generated failure descriptions.

    Fix 1: The primary description comes from an LLM call, not a pre-programmed
    template. Statistical features (clustering, sparsity, scale) are computed
    from the instance and passed as *context* to the LLM prompt. The LLM then
    produces a free-form, instance-specific description of the weakness.

    This ensures:
    - Descriptions are semantically diverse (not collapsed into 7 fixed buckets).
    - Embedding-based similarity in FailureModeMemory has meaningful signal.
    - The archive accumulates genuinely distinct discovered weaknesses.

    The LLM is invoked via a lightweight direct HTTP call to the Anthropic API
    (matching how AdvEoH already calls the LLM) rather than importing the full
    LLM wrapper, to keep this module self-contained.

    Fallback: if the LLM call fails (timeout, parse error, no API key), the
    analyzer falls back to the statistical rule-based description so that the
    FMA system degrades gracefully rather than crashing.
    """

    # Prompt template for the LLM failure description call.
    # Kept short (fits in ~200 tokens) to minimise latency and cost.
    _PROMPT_TEMPLATE = textwrap.dedent("""\
        You are analyzing why a heuristic solver failed on a combinatorial optimization instance.

        Task: {task}

        Instance structural statistics:
        {stats_summary}

        Generator strategy (what the generator tried to do):
        {generator_description}

        Heuristic strategy (what the solver tried to do):
        {heuristic_description}

        Performance gap: the heuristic achieved {heuristic_score:.4f} vs optimal {optimal_value:.4f} \
(gap = {gap_pct:.1f}%).

        In exactly ONE sentence (max 40 words), describe the specific structural weakness \
of this instance that caused the heuristic to fail. Be precise and concrete — name the \
structural pattern (e.g. clustered layout, symmetric alternatives, hub-spoke topology, \
deceptive local optima, adversarial corridors). Do not start with "The instance".
    """)

    _SYSTEM_PROMPT = (
        "You are a concise algorithmic analyst. "
        "Respond with exactly one sentence describing a heuristic failure mode. "
        "Do not add preamble, explanation, or punctuation beyond the sentence."
    )

    def __init__(
        self,
        task: str = '',
        llm_call_fn=None,
        model: str = 'claude-haiku-4-5-20251001',
        max_tokens: int = 80,
        timeout: float = 15.0,
    ):
        """
        Args:
            task: Short task description string (e.g. "TSP constructive heuristic").
            llm_call_fn: Optional callable(prompt: str) -> str. If provided, used
                instead of the built-in Anthropic API call. Useful for testing or
                when the caller already has an LLM wrapper. Signature:
                    fn(prompt: str, system: str) -> str | None
            model: Anthropic model to use for failure description generation.
                   Haiku is used by default — cheap and fast, ~1 call per failure.
            max_tokens: Maximum tokens for the description (one sentence ≈ 40 words).
            timeout: HTTP request timeout in seconds.
        """
        self._task = task
        self._llm_call_fn = llm_call_fn
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout

    # ================================================================
    #  Main entry point
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

        The description field is now LLM-generated (Fix 1).

        Returns None if:
        - heuristic_score is None
        - performance gap is below threshold (< 5% relative gap)
        - LLM call fails AND statistical fallback also fails
        """
        if heuristic_score is None:
            return None

        delta, optimal = self._compute_delta(heuristic_score, optimal_value)
        if delta <= 0:
            return None

        # Relative gap check — skip trivially small failures
        if optimal is not None and abs(optimal) > 1e-12:
            rel_gap = delta / abs(optimal)
            if rel_gap < 0.05:
                return None

        severity = self._estimate_severity(delta, optimal)

        # Compute instance statistics (used as LLM context)
        stats = self._extract_instance_stats(instance)
        stats_summary = self._format_stats(stats)

        # --- Fix 1: LLM-generated description (primary) ---
        gap_pct = (delta / abs(optimal) * 100) if optimal and abs(optimal) > 1e-12 else 0.0
        prompt = self._PROMPT_TEMPLATE.format(
            task=self._task or 'combinatorial optimization',
            stats_summary=stats_summary,
            generator_description=generator_description[:300] or 'not provided',
            heuristic_description=heuristic_description[:300] or 'not provided',
            heuristic_score=heuristic_score,
            optimal_value=optimal_value if optimal_value is not None else heuristic_score,
            gap_pct=gap_pct,
        )

        description = self._call_llm(prompt)

        # --- Fallback: statistical rule-based description ---
        if not description:
            description = self._fallback_description(stats, heuristic_score, optimal_value)

        if not description:
            return None

        return FailureMode(
            description=description.strip(),
            generator_id=generator_id,
            heuristic_id=heuristic_id,
            generator_description=generator_description,
            heuristic_description=heuristic_description,
            generation=generation,
            task=self._task,
            performance_delta=delta,
            optimal_value=optimal or 1.0,
            severity=severity,
            strength=0.5,
            coverage=0,
            instance_seed=instance_seed,
        )

    # ================================================================
    #  LLM call (Fix 1 core)
    # ================================================================

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM to generate a failure description.

        Uses the injected llm_call_fn if provided, otherwise calls the
        Anthropic API directly (same endpoint AdvEoH uses).
        """
        if self._llm_call_fn is not None:
            try:
                result = self._llm_call_fn(prompt, self._SYSTEM_PROMPT)
                return self._clean_description(result) if result else None
            except Exception:
                return None

        # Direct Anthropic API call
        try:
            import urllib.request
            import os

            api_key = os.environ.get('ANTHROPIC_API_KEY', '')
            if not api_key:
                return None

            payload = json.dumps({
                'model': self._model,
                'max_tokens': self._max_tokens,
                'system': self._SYSTEM_PROMPT,
                'messages': [{'role': 'user', 'content': prompt}],
            }).encode('utf-8')

            req = urllib.request.Request(
                'https://api.anthropic.com/v1/messages',
                data=payload,
                headers={
                    'Content-Type': 'application/json',
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01',
                },
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode('utf-8'))

            text = data.get('content', [{}])[0].get('text', '')
            return self._clean_description(text)

        except Exception:
            return None

    @staticmethod
    def _clean_description(text: str) -> Optional[str]:
        """Strip preamble, ensure single sentence, enforce max length."""
        if not text:
            return None
        # Take first sentence only
        for sep in ['. ', '.\n', '! ', '? ']:
            if sep in text:
                text = text.split(sep)[0] + '.'
                break
        text = text.strip().strip('"\'')
        # Enforce 300-char hard cap (safety)
        if len(text) > 300:
            text = text[:297] + '...'
        return text if len(text) > 10 else None

    # ================================================================
    #  Fallback: rule-based statistical description
    # ================================================================

    _FALLBACK_PATTERNS = {
        'clustering': (
            "Clustered structure causes nearest-neighbor heuristic to make locally "
            "optimal intra-cluster decisions while failing at inter-cluster bridging "
            "({stat_desc})."
        ),
        'symmetry': (
            "Near-symmetric instance creates tie-breaking ambiguity at critical "
            "decision points, causing the heuristic to oscillate without progress "
            "({stat_desc})."
        ),
        'sparsity': (
            "Hub-and-spoke layout with high pairwise distance variance exposes the "
            "heuristic's myopic nearest-neighbor selection at long inter-hub edges "
            "({stat_desc})."
        ),
        'scale': (
            "Extreme value range causes threshold-based heuristic to apply "
            "inappropriate normalisation, degrading decision quality ({stat_desc})."
        ),
        'deceptive': (
            "Deceptive local-optimum structure forces early commitment to decisions "
            "that are globally suboptimal and irrecoverable ({stat_desc})."
        ),
        'uniform': (
            "Near-uniform structure without exploitable patterns causes the heuristic "
            "to fail assumptions baked into its design ({stat_desc})."
        ),
    }

    def _fallback_description(
        self,
        stats: dict,
        heuristic_score: float,
        optimal_value: Optional[float],
    ) -> Optional[str]:
        category, stat_desc = self._categorize_from_stats(stats, heuristic_score, optimal_value)
        pattern = self._FALLBACK_PATTERNS.get(category, self._FALLBACK_PATTERNS['deceptive'])
        return pattern.format(stat_desc=stat_desc)

    @staticmethod
    def _categorize_from_stats(
        stats: dict,
        heuristic_score: float,
        optimal_value: Optional[float],
    ) -> tuple[str, str]:
        cv_pairwise = stats.get('cv_pairwise', 0.0)
        nn_skew = stats.get('nn_skew', 0.0)
        cv_val = stats.get('cv', 0.0)

        if cv_pairwise > 0.5 and nn_skew < -0.5:
            return 'clustering', f'cv_pairwise={cv_pairwise:.2f}, nn_skew={nn_skew:.2f}'
        if cv_pairwise < 0.2 and abs(nn_skew) < 0.3:
            return 'symmetry', f'cv_pairwise={cv_pairwise:.2f}, nn_skew={nn_skew:.2f}'
        if cv_pairwise > 0.5 and nn_skew > 0.5:
            return 'sparsity', f'cv_pairwise={cv_pairwise:.2f}, nn_skew={nn_skew:.2f}'
        if cv_val > 2.0:
            return 'scale', f'value_cv={cv_val:.2f}'

        delta = 0.0
        if optimal_value and abs(optimal_value) > 1e-12:
            delta = (optimal_value - heuristic_score) / abs(optimal_value)
        if delta > 0.5:
            return 'deceptive', f'gap={delta*100:.1f}%'
        return 'uniform', f'cv={cv_val:.2f}'

    # ================================================================
    #  Instance statistics extraction (unchanged from original)
    # ================================================================

    @staticmethod
    def _extract_instance_stats(instance: Any) -> dict:
        stats = {}
        try:
            import numpy as np

            def _to_float_array(data):
                if data is None:
                    return None
                if isinstance(data, np.ndarray):
                    return data
                if isinstance(data, (tuple, list)):
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
            finite_vals = flat[np.isfinite(flat)]

            if finite_vals.size > 0:
                stats['min'] = float(finite_vals.min())
                stats['max'] = float(finite_vals.max())
                stats['mean'] = float(finite_vals.mean())
                stats['std'] = float(finite_vals.std())
                stats['range'] = stats['max'] - stats['min']
                if abs(stats['mean']) > 1e-12:
                    stats['cv'] = stats['std'] / abs(stats['mean'])
                else:
                    stats['cv'] = stats['std']

                if arr.ndim == 2 and arr.shape[1] == 2 and finite_vals.size >= 4:
                    coords = arr[~np.isnan(arr).any(axis=1) & ~np.isinf(arr).any(axis=1)]
                    if coords.shape[0] >= 4:
                        try:
                            from scipy.spatial.distance import pdist
                            from scipy.spatial import KDTree
                            dists = pdist(coords)
                            stats['mean_pairwise_dist'] = float(dists.mean())
                            stats['std_pairwise_dist'] = float(dists.std())
                            stats['cv_pairwise'] = (
                                stats['std_pairwise_dist'] / stats['mean_pairwise_dist']
                                if stats['mean_pairwise_dist'] > 1e-12 else 0
                            )
                            tree = KDTree(coords)
                            nn_dists, _ = tree.query(coords, k=2)
                            nn_dists = nn_dists[:, 1]
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

    @staticmethod
    def _format_stats(stats: dict) -> str:
        """Format instance statistics as a compact human-readable summary."""
        parts = []
        if 'n_elements' in stats:
            parts.append(f'n={stats["n_elements"]}')
        if 'cv_pairwise' in stats:
            parts.append(f'cv_pairwise={stats["cv_pairwise"]:.2f}')
        if 'nn_skew' in stats:
            parts.append(f'nn_skew={stats["nn_skew"]:.2f}')
        if 'cv' in stats:
            parts.append(f'value_cv={stats["cv"]:.2f}')
        if 'range' in stats:
            parts.append(f'range={stats["range"]:.3f}')
        return ', '.join(parts) if parts else 'no structural stats available'

    # ================================================================
    #  Delta and severity (unchanged)
    # ================================================================

    @staticmethod
    def _compute_delta(
        heuristic_score: float,
        optimal_value: Optional[float],
    ) -> tuple[float, Optional[float]]:
        if optimal_value is not None and abs(optimal_value) > 1e-12:
            delta = max(0.0, optimal_value - heuristic_score)
            return delta, optimal_value
        if heuristic_score < 0:
            return abs(heuristic_score), 0.0
        return max(0.0, 1.0 - heuristic_score), 1.0

    @staticmethod
    def _estimate_severity(delta: float, optimal: Optional[float]) -> float:
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
    #  Batch convenience
    # ================================================================

    def analyze_batch(
        self,
        evaluation_results: List[dict],
        generation: int,
    ) -> List[FailureMode]:
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