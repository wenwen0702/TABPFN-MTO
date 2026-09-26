from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

def _normalize_minmax(y: Sequence[float]) -> np.ndarray:
    """简单 Min-Max 归一化；全常数时返回 0 向量。"""
    y = np.asarray(y, dtype=np.float64)
    ymin, ymax = (np.min(y), np.max(y))
    if ymax - ymin < 1e-12:
        return np.zeros_like(y)
    return (y - ymin) / (ymax - ymin)

def build_fitness_nn_pairs(X_src: np.ndarray, f_src: Sequence[float], X_tgt: np.ndarray, f_tgt: Sequence[float]):
    """Construct uncertainty-filtered mutual nearest-neighbor pairs in normalized fitness space."""
    X_src = np.asarray(X_src, dtype=np.float64)
    X_tgt = np.asarray(X_tgt, dtype=np.float64)
    f_src = np.asarray(f_src, dtype=np.float64).reshape(-1)
    f_tgt = np.asarray(f_tgt, dtype=np.float64).reshape(-1)
    if X_src.ndim == 1:
        X_src = X_src[None, :]
    if X_tgt.ndim == 1:
        X_tgt = X_tgt[None, :]
    if len(X_src) == 0 or len(X_tgt) == 0:
        return []
    if len(f_src) != len(X_src) or len(f_tgt) != len(X_tgt):
        raise ValueError(f'Shape mismatch: X_src={X_src.shape}, f_src={f_src.shape}, X_tgt={X_tgt.shape}, f_tgt={f_tgt.shape}')
    fn_src = _normalize_minmax(f_src)
    fn_tgt = _normalize_minmax(f_tgt)

    def nearest_index(value, candidates):
        candidates = np.asarray(candidates, dtype=np.float64)
        dist = np.abs(candidates - value)
        min_dist = np.min(dist)
        tied = np.flatnonzero(np.isclose(dist, min_dist, rtol=0.0, atol=1e-12))
        if len(tied) == 1:
            return int(tied[0])
        tied_values = candidates[tied]
        best_value = np.min(tied_values)
        tied2 = tied[np.isclose(tied_values, best_value, rtol=0.0, atol=1e-12)]
        return int(np.min(tied2))

    def directed_nn_uncertainty(value, candidates, matched_idx, eps=1e-12):
        """Return d1/d2 ambiguity for a deterministic nearest-neighbor match."""
        candidates = np.asarray(candidates, dtype=np.float64).reshape(-1)
        if candidates.size < 2:
            return 1.0
        dist = np.abs(candidates - float(value))
        matched_idx = int(matched_idx)
        d1 = float(dist[matched_idx])
        mask = np.ones(candidates.size, dtype=bool)
        mask[matched_idx] = False
        others = dist[mask]
        if others.size == 0:
            return 1.0
        d2 = float(np.min(others))
        if np.isclose(d1, d2, rtol=0.0, atol=1e-12):
            return 1.0
        if d2 <= eps:
            return 1.0
        return float(np.clip(d1 / (d2 + eps), 0.0, 1.0))

    def build_mnn_indices():
        nn_j = np.empty(len(X_src), dtype=np.int64)
        for i in range(len(X_src)):
            nn_j[i] = nearest_index(fn_src[i], fn_tgt)
        nn_i = np.empty(len(X_tgt), dtype=np.int64)
        for j in range(len(X_tgt)):
            nn_i[j] = nearest_index(fn_tgt[j], fn_src)
        out = []
        for i in range(len(X_src)):
            j = int(nn_j[i])
            if int(nn_i[j]) == i:
                out.append((int(i), int(j)))
        return out
    pairs = []
    for i, j in build_mnn_indices():
        source_uncertainty = directed_nn_uncertainty(fn_src[i], fn_tgt, j)
        target_uncertainty = directed_nn_uncertainty(fn_tgt[j], fn_src, i)
        if max(source_uncertainty, target_uncertainty) <= 0.8:
            pairs.append((X_src[i].copy(), X_tgt[j].copy()))
    return pairs

@dataclass
class TaskState:
    """保存单个任务在在线进化过程中的完整状态。"""
    task: Any
    D: int
    D2: int
    pop_A: np.ndarray
    pop_B: np.ndarray
    last_fA: np.ndarray
    last_fB: np.ndarray
    bestA_sofar: float
    bestB_sofar: float
    mig_every: int
    mig_ratio_A: float = 0.2
    mig_ratio_B: float = 0.2
