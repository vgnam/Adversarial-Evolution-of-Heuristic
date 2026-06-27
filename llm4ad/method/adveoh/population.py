from __future__ import annotations

import hashlib
import math
import random
from collections import deque
from threading import Lock
from typing import List

from ..eoh.population import Population as _EoHPopulation
from ...base import Function


class Population(_EoHPopulation):
    """Population for AdvEoH. Inherits from EoH's Population but relaxes the
    duplicate check: only code (str) equality is considered, NOT score equality.
    In adversarial co-evolution, different heuristics legitimately produce the
    same bin count on the same instance set, so score-based dedup is too aggressive.
    """

    def __init__(self, pop_size, generation=0, pop: List[Function] | _EoHPopulation | None = None,
                 eta: float = 1.0):
        super().__init__(pop_size=pop_size, generation=generation, pop=pop)
        self.eta = eta
        self.log_w: list[float] = []
        self._id_map: dict[str, float] = {}

    @staticmethod
    def _individual_id(ind: Function) -> str:
        ind_id = getattr(ind, 'id', None)
        if ind_id is not None:
            return ind_id
        ind_id = hashlib.sha1(str(ind).encode('utf-8')).hexdigest()
        ind.id = ind_id
        return ind_id

    def has_duplicate_function(self, func: str | Function) -> bool:
        func_str = str(func) if isinstance(func, Function) else func
        for f in self._population:
            if str(f) == func_str:
                return True
        for f in self._next_gen_pop:
            if str(f) == func_str:
                return True
        return False

    @staticmethod
    def _is_valid_score(score) -> bool:
        try:
            return score is not None and math.isfinite(float(score))
        except (TypeError, ValueError):
            return False

    def register_function(self, func: Function):
        """Register offspring without triggering mid-batch survival.

        AdvEoH can run sampler threads in parallel. The base EoH population
        calls ``survival()`` as soon as ``next_gen_pop`` reaches pop_size, which
        mutates the selectable population while sibling sampler threads may
        still be choosing parents. AdvEoH performs survival explicitly after a
        sampling batch has completed.
        """
        if self._generation == 0 and func.score is None:
            return
        if func.score is None:
            func.score = float('-inf')
        with self._lock:
            if self.has_duplicate_function(func):
                func.score = float('-inf')
            self._next_gen_pop.append(func)

    def update_weights(self):
        """Update MWU log-weights once after population scores are comparable."""
        if not self._population:
            self.log_w = []
            self._id_map = {}
            return

        ids = [self._individual_id(ind) for ind in self._population]
        log_w = [self._id_map.get(ind_id, 0.0) for ind_id in ids]
        scores = [ind.score for ind in self._population]
        valid_indices = [
            i for i, score in enumerate(scores)
            if self._is_valid_score(score)
        ]

        if valid_indices:
            valid_scores = [float(scores[i]) for i in valid_indices]
            smin = min(valid_scores)
            smax = max(valid_scores)
            if smax - smin >= 1e-12:
                for i in valid_indices:
                    s_norm = (float(scores[i]) - smin) / (smax - smin)
                    log_w[i] += self.eta * s_norm

        self.log_w = log_w
        self._id_map = {ind_id: lw for ind_id, lw in zip(ids, log_w)}

    def _probs(self, funcs: list[Function]) -> list[float]:
        if not funcs:
            return []
        log_w = [self._id_map.get(self._individual_id(func), 0.0) for func in funcs]
        max_log_w = max(log_w)
        weights = [math.exp(lw - max_log_w) for lw in log_w]
        total = sum(weights)
        if total <= 0:
            return [1.0 / len(funcs)] * len(funcs)
        return [w / total for w in weights]

    def selection(self) -> Function:
        with self._lock:
            funcs = [
                f for f in self._population
                if self._is_valid_score(f.score)
            ]
            probs = self._probs(funcs)
        if not funcs:
            raise ValueError('Cannot select from a population without finite scores.')
        return random.choices(funcs, weights=probs, k=1)[0]


class HallOfFame:
    """Sliding-window Hall of Fame. Keeps the top-k elite from each of the last
    ``max_gen`` generations. Provides stratified sampling (2 elite + rest random)
    and description extraction for opponent-aware prompts.
    """

    def __init__(self, max_gen: int = 5, top_k: int = 5):
        self._max_gen = max_gen
        self._top_k = top_k
        self._archive: deque[list[Function]] = deque(maxlen=max_gen)
        self._lock = Lock()

    def __len__(self):
        return sum(len(gen_elites) for gen_elites in self._archive)

    def update(self, population: Population):
        with self._lock:
            pop = sorted(
                population.population,
                key=lambda f: f.score if f.score is not None else float('-inf'),
                reverse=True,
            )
            self._archive.append(list(pop[:self._top_k]))

    def _all_funcs(self) -> List[Function]:
        return [f for gen_elites in self._archive for f in gen_elites]

    def sample(self, n: int, stratified: bool = True) -> List[Function]:
        with self._lock:
            all_funcs = self._all_funcs()
        if not all_funcs:
            return []
        valid = [f for f in all_funcs if f.score is not None and f.score != float('-inf')]
        if not valid:
            return []
        if stratified:
            sorted_funcs = sorted(valid, key=lambda f: f.score, reverse=True)
            n_elite = min(2, len(sorted_funcs), n)
            elites = sorted_funcs[:n_elite]
            rest = sorted_funcs[n_elite:]
            n_random = max(0, n - len(elites))
            pool = rest if rest else sorted_funcs
            sampled = elites + random.sample(pool, min(n_random, len(pool)))
            return sampled[:n]
        else:
            return random.sample(valid, min(n, len(valid)))

    def get_top_k_descriptions(self, k: int) -> List[str]:
        with self._lock:
            all_funcs = self._all_funcs()
        valid = [f for f in all_funcs if f.score is not None and f.score != float('-inf')]
        if not valid:
            return []
        sorted_funcs = sorted(valid, key=lambda f: f.score, reverse=True)
        descs = []
        for f in sorted_funcs[:k]:
            desc = getattr(f, 'algorithm', None) or '(no description)'
            descs.append(desc)
        return descs
