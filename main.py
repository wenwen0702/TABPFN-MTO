"""CEC2017 Full algorithm. Public API: run_online_single_problem(question, init_seed)."""
import os
__all__ = ['run_online_single_problem']
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
import numpy as np
from CEC2017MTSO import Tasks
from utils import TaskState, build_fitness_nn_pairs

def _build_mapping_regressor(*, categorical_features_indices):
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor(fit_mode='low_memory', device='auto', categorical_features_indices=list(categorical_features_indices), random_state=42)

def _feature_layout(D_src: int):
    D_src = int(D_src)
    cat_idx = [D_src + 1]
    return (D_src + 3, cat_idx)

def _make_features_for_mapping(X_src: np.ndarray, D_src: int, D_tgt: int, dir_align: float=0.0, elite_center: np.ndarray=None):
    """Build full source-coordinate, elite-distance, dimension and alignment features."""
    X_src = np.asarray(X_src, dtype=np.float64).copy()
    X_src = X_src[:, :D_src]
    n = X_src.shape[0]
    feat_dim, _ = _feature_layout(D_src)
    if n == 0 or D_tgt == 0:
        return np.zeros((0, feat_dim), dtype=np.float64)
    if elite_center is None:
        elite_center = np.zeros((D_src,), dtype=np.float64)
    else:
        elite_center = np.asarray(elite_center, dtype=np.float64).reshape(-1)[:D_src]
    elite_dist = np.linalg.norm(X_src - elite_center.reshape(1, -1), axis=1, keepdims=True)
    X_rep = np.repeat(X_src, repeats=D_tgt, axis=0)
    elite_dist_rep = np.repeat(elite_dist, repeats=D_tgt, axis=0)
    dim_id = np.tile(np.arange(D_tgt, dtype=np.float64), reps=n).reshape(-1, 1)
    dir_col = np.full((X_rep.shape[0], 1), float(dir_align), dtype=np.float64)
    X_feat = np.concatenate([X_rep, elite_dist_rep, dim_id, dir_col], axis=1)
    return X_feat

def _compute_direction_alignment(pop_src: np.ndarray, f_src: np.ndarray, pop_tgt: np.ndarray, f_tgt: np.ndarray, *, D_src: int, D_tgt: int, elite_ratio: float=0.2):
    pop_src = np.asarray(pop_src, dtype=np.float64)[:, :D_src]
    pop_tgt = np.asarray(pop_tgt, dtype=np.float64)[:, :D_tgt]
    f_src = np.asarray(f_src, dtype=np.float64)
    f_tgt = np.asarray(f_tgt, dtype=np.float64)
    if len(pop_src) == 0 or len(pop_tgt) == 0:
        return 0.0
    D_align = max(D_src, D_tgt)
    if D_align <= 0:
        return 0.0
    pop_src_c = np.zeros((pop_src.shape[0], D_align), dtype=np.float64)
    pop_tgt_c = np.zeros((pop_tgt.shape[0], D_align), dtype=np.float64)
    pop_src_c[:, :D_src] = pop_src[:, :D_src]
    pop_tgt_c[:, :D_tgt] = pop_tgt[:, :D_tgt]
    m_src = np.mean(pop_src_c, axis=0)
    m_tgt = np.mean(pop_tgt_c, axis=0)
    n_elite_src = max(1, int(np.ceil(elite_ratio * len(pop_src_c))))
    n_elite_tgt = max(1, int(np.ceil(elite_ratio * len(pop_tgt_c))))
    idx_src = np.argsort(f_src)[:n_elite_src]
    idx_tgt = np.argsort(f_tgt)[:n_elite_tgt]
    c_src = np.mean(pop_src_c[idx_src], axis=0)
    c_tgt = np.mean(pop_tgt_c[idx_tgt], axis=0)
    v_src = c_src - m_src
    v_tgt = c_tgt - m_tgt
    norm_src = np.linalg.norm(v_src)
    norm_tgt = np.linalg.norm(v_tgt)
    if norm_src < 1e-12 or norm_tgt < 1e-12:
        return 0.0
    dir_align = float(np.dot(v_src, v_tgt) / (norm_src * norm_tgt + 1e-12))
    return float(np.clip(dir_align, -1.0, 1.0))

def _compute_elite_center(pop_src: np.ndarray, f_src: np.ndarray, *, D_src: int, elite_ratio: float=0.2):
    pop_src = np.asarray(pop_src, dtype=np.float64)[:, :D_src]
    f_src = np.asarray(f_src, dtype=np.float64)
    if len(pop_src) == 0:
        raise ValueError('pop_src is empty')
    n_elite = max(1, int(np.ceil(elite_ratio * len(pop_src))))
    idx_elite = np.argsort(f_src)[:n_elite]
    elite_center = np.mean(pop_src[idx_elite], axis=0)
    return elite_center.astype(np.float64)

def _de_mutation(pop, F, rng, target_idx: int):
    NP, D = pop.shape
    candidates = np.arange(NP)
    candidates = candidates[candidates != target_idx]
    idx = rng.choice(candidates, size=3, replace=False)
    r1, r2, r3 = pop[idx]
    v = r1 + F * (r2 - r3)
    return np.clip(v, 0.0, 1.0)

def _de_crossover(target, donor, CR, rng):
    D = target.shape[0]
    out = target.copy()
    j_rand = rng.integers(0, D)
    for j in range(D):
        if rng.random() < CR or j == j_rand:
            out[j] = donor[j]
    return out

def _de_generate_offspring(pop, F, CR, rng, n_offspring):
    NP, D = pop.shape
    children = np.zeros((n_offspring, D))
    for i in range(n_offspring):
        idx = rng.integers(0, NP)
        target = pop[idx]
        donor = _de_mutation(pop, F, rng, target_idx=idx)
        trial = _de_crossover(target, donor, CR, rng)
        children[i] = trial
    return children

def _make_offspring_DE(pop_A, pop_B, D: int, D2: int, rng, *, n_off_A: int, n_off_B: int, F=0.5, CR=0.9):
    off_A = np.zeros((n_off_A, D), dtype=np.float64)
    off_B = np.zeros((n_off_B, D2), dtype=np.float64)
    if n_off_A > 0:
        off_A = _de_generate_offspring(pop_A[:, :D], F=F, CR=CR, rng=rng, n_offspring=n_off_A)
    if n_off_B > 0:
        off_B = _de_generate_offspring(pop_B[:, :D2], F=F, CR=CR, rng=rng, n_offspring=n_off_B)
    return (off_A, off_B)

def _normalize_minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return v
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmax - vmin < 1e-12:
        return np.zeros_like(v, dtype=np.float64)
    return (v - vmin) / (vmax - vmin + 1e-12)

def _remove_duplicate_rows(X: np.ndarray, f: np.ndarray, round_decimals: int=12):
    X = np.asarray(X, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    if X.size == 0:
        return (np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=np.float64), np.zeros((0,), dtype=np.float64))
    order = np.argsort(f)
    X = X[order]
    f = f[order]
    seen = set()
    keep_X = []
    keep_f = []
    for x, fx in zip(X, f):
        key = tuple(np.round(x, round_decimals))
        if key in seen:
            continue
        seen.add(key)
        keep_X.append(x)
        keep_f.append(fx)
    return (np.asarray(keep_X, dtype=np.float64), np.asarray(keep_f, dtype=np.float64))

def _min_distance_to_selected(X_pool: np.ndarray, X_sel: np.ndarray) -> np.ndarray:
    X_pool = np.asarray(X_pool, dtype=np.float64)
    if X_pool.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if X_sel is None or len(X_sel) == 0:
        return np.full((X_pool.shape[0],), np.inf, dtype=np.float64)
    X_sel = np.asarray(X_sel, dtype=np.float64)
    diff = X_pool[:, None, :] - X_sel[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return np.min(dist, axis=1)

def _greedy_farthest_indices(X_pool: np.ndarray, n_pick: int, X_seed: np.ndarray=None):
    X_pool = np.asarray(X_pool, dtype=np.float64)
    if X_pool.shape[0] == 0 or n_pick <= 0:
        return []
    chosen = []
    selected = None if X_seed is None else np.asarray(X_seed, dtype=np.float64)
    remaining = list(range(X_pool.shape[0]))
    for _ in range(min(n_pick, X_pool.shape[0])):
        cand = X_pool[remaining]
        d = _min_distance_to_selected(cand, selected)
        pick_local = int(np.argmax(d))
        pick_idx = remaining[pick_local]
        chosen.append(pick_idx)
        x_new = X_pool[pick_idx:pick_idx + 1]
        if selected is None or len(selected) == 0:
            selected = x_new.copy()
        else:
            selected = np.vstack([selected, x_new])
        remaining.pop(pick_local)
    return chosen

def _greedy_mid_indices(X_pool: np.ndarray, f_pool: np.ndarray, n_pick: int, X_seed: np.ndarray, fitness_weight: float=0.6, dist_weight: float=0.4):
    X_pool = np.asarray(X_pool, dtype=np.float64)
    f_pool = np.asarray(f_pool, dtype=np.float64)
    if X_pool.shape[0] == 0 or n_pick <= 0:
        return []
    chosen = []
    selected = None if X_seed is None else np.asarray(X_seed, dtype=np.float64)
    remaining = list(range(X_pool.shape[0]))
    for _ in range(min(n_pick, X_pool.shape[0])):
        cand_X = X_pool[remaining]
        cand_f = f_pool[remaining]
        fit_score = 1.0 - _normalize_minmax(cand_f)
        dist_score = _normalize_minmax(_min_distance_to_selected(cand_X, selected))
        score = fitness_weight * fit_score + dist_weight * dist_score
        pick_local = int(np.argmax(score))
        pick_idx = remaining[pick_local]
        chosen.append(pick_idx)
        x_new = X_pool[pick_idx:pick_idx + 1]
        if selected is None or len(selected) == 0:
            selected = x_new.copy()
        else:
            selected = np.vstack([selected, x_new])
        remaining.pop(pick_local)
    return chosen

def _sample_representatives_from_population(pop: np.ndarray, f: np.ndarray, sample_size: int, rng, elite_ratio: float=0.5, mid_ratio: float=0.3, diverse_ratio: float=0.2, elite_pool_ratio: float=0.2, mid_pool_end_ratio: float=0.6):
    pop = np.asarray(pop, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    if pop.ndim != 2:
        raise ValueError('pop must be 2D')
    n = len(f)
    if n <= 0 or sample_size <= 0:
        return (np.zeros((0, pop.shape[1]), dtype=np.float64), np.zeros((0,), dtype=np.float64))
    pop, f = _remove_duplicate_rows(pop, f, round_decimals=12)
    n = len(f)
    if n <= 0:
        return (np.zeros((0, pop.shape[1]), dtype=np.float64), np.zeros((0,), dtype=np.float64))
    sample_size = min(int(sample_size), n)
    order = np.argsort(f)
    n_elite_pool = max(1, int(np.ceil(elite_pool_ratio * n)))
    n_mid_end = max(n_elite_pool + 1, int(np.ceil(mid_pool_end_ratio * n)))
    n_mid_end = min(n_mid_end, n)
    elite_idx = order[:n_elite_pool]
    mid_idx = order[n_elite_pool:n_mid_end]
    diverse_idx = order[n_mid_end:]
    s = float(elite_ratio) + float(mid_ratio) + float(diverse_ratio)
    if s <= 0:
        elite_ratio, mid_ratio, diverse_ratio = (0.5, 0.3, 0.2)
        s = 1.0
    elite_ratio /= s
    mid_ratio /= s
    diverse_ratio /= s
    n_elite = int(round(sample_size * elite_ratio))
    n_mid = int(round(sample_size * mid_ratio))
    n_div = int(sample_size - n_elite - n_mid)
    if sample_size >= 3:
        if n_elite <= 0:
            n_elite = 1
        if mid_idx.size > 0 and n_mid <= 0:
            n_mid = 1
        if diverse_idx.size > 0 and n_div <= 0:
            n_div = 1
    while n_elite + n_mid + n_div > sample_size:
        if n_div > 0:
            n_div -= 1
        elif n_mid > 0:
            n_mid -= 1
        else:
            n_elite -= 1
    selected_global_idx = []
    selected_X = np.zeros((0, pop.shape[1]), dtype=np.float64)
    if n_elite > 0 and elite_idx.size > 0:
        best_idx = int(elite_idx[0])
        selected_global_idx.append(best_idx)
        selected_X = np.vstack([selected_X, pop[best_idx:best_idx + 1]])
        remain_elite = [idx for idx in elite_idx.tolist() if idx != best_idx]
        if n_elite > 1 and len(remain_elite) > 0:
            elite_local = _greedy_farthest_indices(pop[remain_elite], n_pick=n_elite - 1, X_seed=selected_X)
            extra_idx = [remain_elite[i] for i in elite_local]
            selected_global_idx.extend(extra_idx)
            selected_X = np.vstack([selected_X, pop[extra_idx]])
    mid_candidates = [idx for idx in mid_idx.tolist() if idx not in set(selected_global_idx)]
    if n_mid > 0 and len(mid_candidates) > 0:
        mid_local = _greedy_mid_indices(pop[mid_candidates], f[mid_candidates], n_pick=n_mid, X_seed=selected_X, fitness_weight=0.6, dist_weight=0.4)
        extra_idx = [mid_candidates[i] for i in mid_local]
        selected_global_idx.extend(extra_idx)
        selected_X = np.vstack([selected_X, pop[extra_idx]])
    rest_candidates = [idx for idx in diverse_idx.tolist() if idx not in set(selected_global_idx)]
    if n_div > 0 and len(rest_candidates) > 0:
        div_local = _greedy_farthest_indices(pop[rest_candidates], n_pick=n_div, X_seed=selected_X)
        extra_idx = [rest_candidates[i] for i in div_local]
        selected_global_idx.extend(extra_idx)
        selected_X = np.vstack([selected_X, pop[extra_idx]])
    if len(selected_global_idx) < sample_size:
        used = set(selected_global_idx)
        all_remaining = [idx for idx in order.tolist() if idx not in used]
        fill_local = _greedy_farthest_indices(pop[all_remaining], n_pick=sample_size - len(selected_global_idx), X_seed=selected_X)
        extra_idx = [all_remaining[i] for i in fill_local]
        selected_global_idx.extend(extra_idx)
    selected_global_idx = np.asarray(selected_global_idx[:sample_size], dtype=int)
    sample_X = pop[selected_global_idx]
    sample_f = f[selected_global_idx]
    order_final = np.argsort(sample_f)
    return (sample_X[order_final], sample_f[order_final])

def _stack_history_bucket(hist_pop_list, hist_f_list, start: int, end: int, width: int):
    xs, fs = ([], [])
    for p, fv in zip(hist_pop_list[start:end], hist_f_list[start:end]):
        if p is None or fv is None:
            continue
        if len(fv) <= 0:
            continue
        xs.append(np.asarray(p, dtype=np.float64))
        fs.append(np.asarray(fv, dtype=np.float64))
    if len(xs) == 0:
        return (np.zeros((0, width), dtype=np.float64), np.zeros((0,), dtype=np.float64))
    X = np.vstack(xs)
    y = np.concatenate(fs)
    return _remove_duplicate_rows(X, y, round_decimals=12)

def _context_temporal_quotas(k_total: int):
    cur_ratio, near_ratio, far_ratio = (0.7, 0.2, 0.1)
    k_total = max(0, int(k_total))
    s = float(cur_ratio) + float(near_ratio) + float(far_ratio)
    if s <= 0:
        cur_ratio, near_ratio, far_ratio = (0.7, 0.2, 0.1)
        s = 1.0
    cur_ratio = float(cur_ratio) / s
    near_ratio = float(near_ratio) / s
    far_ratio = float(far_ratio) / s
    k_cur = int(round(k_total * cur_ratio))
    k_near = int(round(k_total * near_ratio))
    k_far = int(k_total - k_cur - k_near)
    if k_far < 0:
        overflow = -k_far
        reduce_near = min(overflow, max(0, k_near))
        k_near -= reduce_near
        overflow -= reduce_near
        if overflow > 0:
            k_cur = max(0, k_cur - overflow)
        k_far = k_total - k_cur - k_near
    return (int(k_cur), int(k_near), int(k_far))

def _collect_training_representatives_from_history(hist_pop_list, hist_f_list, k_total: int, rng):
    """Sample the current, near and far history buckets with the original Full quotas."""
    if hist_pop_list is None or hist_f_list is None or len(hist_pop_list) == 0:
        return (np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.float64))
    W = hist_pop_list[-1].shape[1]
    k_total = max(0, int(k_total))
    if k_total <= 0:
        return (np.zeros((0, W), dtype=np.float64), np.zeros((0,), dtype=np.float64))
    k_cur, k_near, k_far = _context_temporal_quotas(k_total)
    cur_X, cur_f = _remove_duplicate_rows(np.asarray(hist_pop_list[-1], dtype=np.float64), np.asarray(hist_f_list[-1], dtype=np.float64), round_decimals=12)
    near_X, near_f = _stack_history_bucket(hist_pop_list, hist_f_list, max(0, len(hist_pop_list) - 4), max(0, len(hist_pop_list) - 1), W)
    far_X, far_f = _stack_history_bucket(hist_pop_list, hist_f_list, max(0, len(hist_pop_list) - 6), max(0, len(hist_pop_list) - 4), W)

    def _sample_hierarchical(X, f, k, elite_ratio, mid_ratio, diverse_ratio):
        if int(k) <= 0 or len(f) <= 0:
            return (np.zeros((0, W), dtype=np.float64), np.zeros((0,), dtype=np.float64))
        return _sample_representatives_from_population(X, f, k, rng, elite_ratio=elite_ratio, mid_ratio=mid_ratio, diverse_ratio=diverse_ratio)
    parts_X, parts_f = ([], [])
    Xc, fc = _sample_hierarchical(cur_X, cur_f, k_cur, elite_ratio=5.0, mid_ratio=3.0, diverse_ratio=1.0)
    Xn, fn = _sample_hierarchical(near_X, near_f, k_near, elite_ratio=3.0, mid_ratio=1.0, diverse_ratio=0.0)
    Xf, ff = _sample_hierarchical(far_X, far_f, k_far, elite_ratio=1.0, mid_ratio=0.0, diverse_ratio=1.0)
    for Xp, fp in [(Xc, fc), (Xn, fn), (Xf, ff)]:
        if len(fp) > 0:
            parts_X.append(Xp)
            parts_f.append(fp)
    if len(parts_X) == 0:
        X_all = np.zeros((0, W), dtype=np.float64)
        f_all = np.zeros((0,), dtype=np.float64)
    else:
        X_all = np.vstack(parts_X)
        f_all = np.concatenate(parts_f)
        X_all, f_all = _remove_duplicate_rows(X_all, f_all, round_decimals=12)
    if len(f_all) < k_total and len(cur_f) > 0:
        need = k_total - len(f_all)
        X_extra, f_extra = _sample_representatives_from_population(cur_X, cur_f, min(len(cur_f), k_total + need), rng, elite_ratio=5.0, mid_ratio=3.0, diverse_ratio=1.0)
        if len(f_extra) > 0:
            X_merge = np.vstack([X_all, X_extra])
            f_merge = np.concatenate([f_all, f_extra])
            X_merge, f_merge = _remove_duplicate_rows(X_merge, f_merge, round_decimals=12)
            if len(f_merge) > k_total:
                X_merge, f_merge = _sample_representatives_from_population(X_merge, f_merge, k_total, rng, elite_ratio=5.0, mid_ratio=3.0, diverse_ratio=1.0)
            X_all, f_all = (X_merge, f_merge)
    return (X_all, f_all)

def _build_online_ctx_pairs_dir(top_src_X, top_src_f, top_tgt_X, top_tgt_f, *, pop_src, f_src, pop_tgt, f_tgt, D_src, D_tgt):
    pairs = build_fitness_nn_pairs(top_src_X, top_src_f, top_tgt_X, top_tgt_f)
    n_pairs = len(pairs)
    if n_pairs == 0 or D_tgt == 0:
        feature_dim, _ = _feature_layout(D_src)
        return (np.zeros((0, feature_dim)), np.zeros(0), 0)
    alignment = _compute_direction_alignment(pop_src, f_src, pop_tgt, f_tgt, D_src=D_src, D_tgt=D_tgt, elite_ratio=0.2)
    X_src = np.stack([source[:D_src] for source, _ in pairs])
    X_tgt = np.stack([target[:D_tgt] for _, target in pairs])
    elite_center = _compute_elite_center(pop_src, f_src, D_src=D_src, elite_ratio=0.2)
    features = _make_features_for_mapping(X_src, D_src=D_src, D_tgt=D_tgt, dir_align=alignment, elite_center=elite_center)
    return (features, X_tgt.reshape(-1).astype(np.float64, copy=False), n_pairs)

def _try_fit_ic_context_dir(model, pop_src, pop_tgt, f_src, f_tgt, *, hist_pop_src=None, hist_pop_tgt=None, hist_f_src=None, hist_f_tgt=None, D_src, D_tgt, pop_size, last_ctx_hash=None, rng=None):
    """Return (did_fit, context_hash, current_context_valid), independently of logging."""
    k = min(20, pop_size)
    if k <= 1 or D_tgt <= 0:
        return (False, last_ctx_hash, False)
    if hist_pop_src is None or hist_f_src is None or len(hist_pop_src) == 0:
        hist_pop_src = [np.asarray(pop_src, dtype=np.float64)]
        hist_f_src = [np.asarray(f_src, dtype=np.float64)]
    if hist_pop_tgt is None or hist_f_tgt is None or len(hist_pop_tgt) == 0:
        hist_pop_tgt = [np.asarray(pop_tgt, dtype=np.float64)]
        hist_f_tgt = [np.asarray(f_tgt, dtype=np.float64)]
    top_src_X, top_src_f = _collect_training_representatives_from_history(hist_pop_src, hist_f_src, k_total=k, rng=rng)
    top_tgt_X, top_tgt_f = _collect_training_representatives_from_history(hist_pop_tgt, hist_f_tgt, k_total=k, rng=rng)
    if len(top_src_X) <= 1 or len(top_tgt_X) <= 1:
        return (False, last_ctx_hash, False)
    X_ctx, y_ctx, n_pairs = _build_online_ctx_pairs_dir(top_src_X, top_src_f, top_tgt_X, top_tgt_f, pop_src=pop_src, f_src=f_src, pop_tgt=pop_tgt, f_tgt=f_tgt, D_src=D_src, D_tgt=D_tgt)
    if n_pairs < 3 or X_ctx.shape[0] < 6:
        return (False, last_ctx_hash, False)
    X_ctx = np.round(X_ctx, 8)
    y_ctx = np.round(y_ctx, 8)
    if np.unique(X_ctx, axis=0).shape[0] < min(8, X_ctx.shape[0]):
        return (False, last_ctx_hash, False)
    if np.all(np.var(X_ctx, axis=0) < 1e-12):
        return (False, last_ctx_hash, False)
    ctx_hash = hash((X_ctx.tobytes(), y_ctx.tobytes()))
    if last_ctx_hash is not None and ctx_hash == last_ctx_hash:
        return (False, last_ctx_hash, True)
    try:
        model.fit(X_ctx, y_ctx)
    except Exception as error:
        print(f'[model-fit skipped] {type(error).__name__}: {error}')
        return (False, last_ctx_hash, False)
    return (True, ctx_hash, True)

def _map_solutions_dir(model_dir, X_src: np.ndarray, *, D_src: int, D_tgt: int, elite_center: np.ndarray, dir_align: float=0.0):
    X_src = np.asarray(X_src, dtype=np.float64)[:, :D_src]
    n = int(X_src.shape[0])
    if n == 0:
        return np.zeros((0, D_tgt), dtype=np.float64)
    X_feat = _make_features_for_mapping(X_src, D_src=D_src, D_tgt=D_tgt, dir_align=dir_align, elite_center=elite_center)
    _preds = None
    try:
        _preds = model_dir.predict(X_feat)
    except Exception as e:
        print(f'[model-predict skipped] {type(e).__name__}: {e}')
    if _preds is None:
        return np.zeros((0, D_tgt), dtype=np.float64)
    preds = np.asarray(_preds, dtype=np.float64).reshape(-1)
    mig = np.clip(preds.reshape(n, D_tgt), 0.0, 1.0)
    return mig

def _perform_migration(rec, model_A2B, model_B2A, *, n_mig_B2A: int, n_mig_A2B: int, D: int, D2: int, do_B2A: bool, do_A2B: bool):
    mig_B2A = np.zeros((0, D), dtype=np.float64)
    mig_A2B = np.zeros((0, D2), dtype=np.float64)
    rng_local = getattr(rec, 'rng', None)
    if do_B2A and n_mig_B2A > 0 and getattr(rec, 'model_ready_B2A', False):
        src_B, _ = _sample_representatives_from_population(rec.pop_B, rec.last_fB, sample_size=n_mig_B2A, rng=rng_local, elite_ratio=0.5, mid_ratio=0.3, diverse_ratio=0.2)
        elite_center_B = _compute_elite_center(pop_src=rec.pop_B, f_src=rec.last_fB, D_src=D2, elite_ratio=0.2)
        dir_align_B2A = _compute_direction_alignment(pop_src=rec.pop_B, f_src=rec.last_fB, pop_tgt=rec.pop_A, f_tgt=rec.last_fA, D_src=D2, D_tgt=D, elite_ratio=0.2)
        mig_B2A = _map_solutions_dir(model_B2A, src_B, D_src=D2, D_tgt=D, elite_center=elite_center_B, dir_align=dir_align_B2A)
    if do_A2B and n_mig_A2B > 0 and getattr(rec, 'model_ready_A2B', False):
        src_A, _ = _sample_representatives_from_population(rec.pop_A, rec.last_fA, sample_size=n_mig_A2B, rng=rng_local, elite_ratio=0.5, mid_ratio=0.3, diverse_ratio=0.2)
        elite_center_A = _compute_elite_center(pop_src=rec.pop_A, f_src=rec.last_fA, D_src=D, elite_ratio=0.2)
        dir_align_A2B = _compute_direction_alignment(pop_src=rec.pop_A, f_src=rec.last_fA, pop_tgt=rec.pop_B, f_tgt=rec.last_fB, D_src=D, D_tgt=D2, elite_ratio=0.2)
        mig_A2B = _map_solutions_dir(model_A2B, src_A, D_src=D, D_tgt=D2, elite_center=elite_center_A, dir_align=dir_align_A2B)
    return (mig_B2A, mig_A2B)

def _prepare_migration(rec, gen: int, pop_size: int, model_A2B, model_B2A, *, D: int, D2: int):
    context_sampling_rng = rec.rng
    context_valid_B2A = context_valid_A2B = False
    planned = (gen + 1) % int(rec.mig_every) == 0
    do_B2A = planned
    do_A2B = planned
    ratio_B2A = float(rec.mig_ratio_A)
    ratio_A2B = float(rec.mig_ratio_B)
    n_mig_B2A = int(round(ratio_B2A * pop_size)) if do_B2A else 0
    n_mig_A2B = int(round(ratio_A2B * pop_size)) if do_A2B else 0
    if do_B2A:
        n_mig_B2A = int(np.clip(n_mig_B2A, 5, int(0.3 * pop_size)))
    else:
        n_mig_B2A = 0
    if do_A2B:
        n_mig_A2B = int(np.clip(n_mig_A2B, 5, int(0.3 * pop_size)))
    else:
        n_mig_A2B = 0
    if not hasattr(rec, 'ctx_hash_B2A'):
        rec.ctx_hash_B2A = None
    if not hasattr(rec, 'ctx_hash_A2B'):
        rec.ctx_hash_A2B = None
    if not hasattr(rec, 'model_ready_B2A'):
        rec.model_ready_B2A = False
    if not hasattr(rec, 'model_ready_A2B'):
        rec.model_ready_A2B = False
    if do_B2A:
        did_fit_B2A, rec.ctx_hash_B2A, context_valid_B2A = _try_fit_ic_context_dir(model_B2A, pop_src=rec.pop_B, pop_tgt=rec.pop_A, f_src=rec.last_fB, f_tgt=rec.last_fA, hist_pop_src=rec.hist_pop_B, hist_pop_tgt=rec.hist_pop_A, hist_f_src=rec.hist_fB, hist_f_tgt=rec.hist_fA, D_src=D2, D_tgt=D, pop_size=pop_size, last_ctx_hash=rec.ctx_hash_B2A, rng=context_sampling_rng)
        rec.model_ready_B2A = rec.model_ready_B2A or did_fit_B2A
    if do_A2B:
        did_fit_A2B, rec.ctx_hash_A2B, context_valid_A2B = _try_fit_ic_context_dir(model_A2B, pop_src=rec.pop_A, pop_tgt=rec.pop_B, f_src=rec.last_fA, f_tgt=rec.last_fB, hist_pop_src=rec.hist_pop_A, hist_pop_tgt=rec.hist_pop_B, hist_f_src=rec.hist_fA, hist_f_tgt=rec.hist_fB, D_src=D, D_tgt=D2, pop_size=pop_size, last_ctx_hash=rec.ctx_hash_A2B, rng=context_sampling_rng)
        rec.model_ready_A2B = rec.model_ready_A2B or did_fit_A2B
    ready_B2A = bool(rec.model_ready_B2A) and context_valid_B2A
    ready_A2B = bool(rec.model_ready_A2B) and context_valid_A2B
    if do_B2A and (not ready_B2A):
        n_mig_B2A = 0
        do_B2A = False
    if do_A2B and (not ready_A2B):
        n_mig_A2B = 0
        do_A2B = False
    return (do_B2A, do_A2B, n_mig_B2A, n_mig_A2B)

def _update_migration_feedback(rec, nextA_labels, nextB_labels, pop_size, *, n_mig_B2A, n_mig_A2B, do_B2A, do_A2B):
    """Apply the paper's rank-weighted survival feedback, without diagnostics."""
    eps = 1e-12
    for side, labels, count, enabled in (('A', nextA_labels, n_mig_B2A, do_B2A), ('B', nextB_labels, n_mig_A2B, do_A2B)):
        if not enabled or count <= 0:
            continue
        labels = np.asarray(labels, dtype=int).reshape(-1)
        contribution = 1.0
        if labels.size:
            weights = np.arange(labels.size, 0, -1, dtype=np.float64)
            migrant_mask = (labels == 1).astype(np.float64)
            actual = float(np.sum(weights * migrant_mask)) / max(float(np.sum(weights)), eps)
            expected = float(count) / max(float(pop_size), eps)
            if expected > eps:
                if actual <= expected:
                    contribution = actual / max(expected, eps)
                else:
                    contribution = 1.0 + (actual - expected) / max(1.0 - expected, eps)
                contribution = float(np.clip(contribution, 0.0, 2.0))
        previous = float(getattr(rec, f'mig_ratio_{side}'))
        updated = float(np.clip(previous + 0.05 * (contribution - 1.0), 0.1, 0.3))
        setattr(rec, f'mig_ratio_{side}', updated)

def _true_fitness_on(tasks, X: np.ndarray, task_id: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    t = tasks[task_id]
    vals = []
    for x in X:
        vals.append(float(t.function(x)))
    return np.array(vals, dtype=np.float64)

def _diversity_aware_survival_selection(all_X: np.ndarray, all_f: np.ndarray, all_label: np.ndarray, pop_size: int, elite_keep_ratio: float=0.2, fitness_weight: float=0.5, dist_weight: float=0.5):
    all_X = np.asarray(all_X, dtype=np.float64)
    all_f = np.asarray(all_f, dtype=np.float64)
    all_label = np.asarray(all_label, dtype=int)
    n = len(all_f)
    if n <= pop_size:
        order = np.argsort(all_f)
        return (all_X[order], all_f[order], all_label[order])
    order = np.argsort(all_f)
    n_elite = max(1, int(round(pop_size * elite_keep_ratio)))
    n_elite = min(n_elite, pop_size)
    elite_idx = order[:n_elite]
    selected_idx = elite_idx.tolist()
    selected_set = set(selected_idx)
    remain_idx = [idx for idx in order if idx not in selected_set]
    while len(selected_idx) < pop_size and len(remain_idx) > 0:
        cand_X = all_X[remain_idx]
        cand_f = all_f[remain_idx]
        sel_X = all_X[selected_idx]
        fit_score = 1.0 - _normalize_minmax(cand_f)
        dist_score = _normalize_minmax(_min_distance_to_selected(cand_X, sel_X))
        score = fitness_weight * fit_score + dist_weight * dist_score
        best_local = int(np.argmax(score))
        best_idx = remain_idx[best_local]
        selected_idx.append(best_idx)
        selected_set.add(best_idx)
        remain_idx.pop(best_local)
    selected_idx = np.asarray(selected_idx, dtype=int)
    new_X = all_X[selected_idx]
    new_f = all_f[selected_idx]
    new_label = all_label[selected_idx]
    idx_final = np.argsort(new_f)
    new_X = new_X[idx_final]
    new_f = new_f[idx_final]
    new_label = new_label[idx_final]
    return (new_X, new_f, new_label)

def _select_next_generation(tasks, pop_A, pop_B, off_A, off_B, mig_B2A, mig_A2B, last_fA, last_fB):
    pop_size = pop_A.shape[0]
    fA_parents = np.asarray(last_fA, dtype=np.float64)
    fA_mut = _true_fitness_on(tasks, off_A, task_id=0) if off_A.size > 0 else np.array([], dtype=np.float64)
    fA_mig = _true_fitness_on(tasks, mig_B2A, task_id=0) if mig_B2A.size > 0 else np.array([], dtype=np.float64)
    fB_parents = np.asarray(last_fB, dtype=np.float64)
    fB_mut = _true_fitness_on(tasks, off_B, task_id=1) if off_B.size > 0 else np.array([], dtype=np.float64)
    fB_mig = _true_fitness_on(tasks, mig_A2B, task_id=1) if mig_A2B.size > 0 else np.array([], dtype=np.float64)
    n_eval_A = int(fA_mut.size + fA_mig.size)
    n_eval_B = int(fB_mut.size + fB_mig.size)

    def _select_one_task(pop_par, f_par, pop_off, f_off, pop_mig, f_mig):
        cand_X = []
        cand_f = []
        cand_label = []
        if len(f_par) > 0:
            cand_X.append(pop_par)
            cand_f.append(f_par)
            cand_label.append(np.full(len(f_par), 2, dtype=int))
        if len(f_off) > 0:
            cand_X.append(pop_off)
            cand_f.append(f_off)
            cand_label.append(np.full(len(f_off), 0, dtype=int))
        if len(f_mig) > 0:
            cand_X.append(pop_mig)
            cand_f.append(f_mig)
            cand_label.append(np.full(len(f_mig), 1, dtype=int))
        all_X = np.vstack(cand_X)
        all_f = np.concatenate(cand_f)
        all_label = np.concatenate(cand_label)
        new_pop, new_f, new_label = _diversity_aware_survival_selection(all_X, all_f, all_label, pop_size=pop_size, elite_keep_ratio=0.2, fitness_weight=0.7, dist_weight=0.3)
        return (new_pop, new_f, new_label)
    new_A, fA, nextA_labels = _select_one_task(pop_A, fA_parents, off_A, fA_mut, mig_B2A, fA_mig)
    new_B, fB, nextB_labels = _select_one_task(pop_B, fB_parents, off_B, fB_mut, mig_A2B, fB_mig)
    return (new_A, new_B, fA, fB, nextA_labels, nextB_labels, n_eval_A, n_eval_B)

def _update_records(rec, new_A, new_B, fA, fB, n_eval_A: int, n_eval_B: int, FE: int):
    FE += n_eval_A + n_eval_B
    rec.pop_A = new_A
    rec.last_fA = fA
    rec.pop_B = new_B
    rec.last_fB = fB
    cur_bestA = float(np.min(fA))
    cur_bestB = float(np.min(fB))
    rec.bestA_sofar = min(rec.bestA_sofar, cur_bestA)
    rec.bestB_sofar = min(rec.bestB_sofar, cur_bestB)
    return FE

def run_online_single_problem(question: str, init_seed: int=2025):
    """Run the fixed paper configuration for one benchmark and one seed."""
    n_generations = 1500
    pop_size = 50
    mig_every = 5
    max_fes = 100000
    if pop_size < 4 or max_fes < 2 * pop_size or max_fes % (2 * pop_size):
        raise ValueError('Require pop_size >= 4 and max_fes a positive multiple of 2*pop_size.')
    if n_generations < max_fes // (2 * pop_size) - 1:
        raise ValueError('n_generations is too small to reach the evaluation budget.')
    if init_seed < 0:
        raise ValueError('init_seed must be non-negative.')
    FE = 0
    task_name = question
    rng = np.random.default_rng(init_seed)
    taskA = Tasks(task_name, 1)
    taskB = Tasks(task_name, 2)
    tasks = [taskA, taskB]
    if task_name == 'PILS':
        D, D2 = (50, 25)
    else:
        D, D2 = (50, 50)
    pop_A = rng.random((pop_size, D))
    pop_B = rng.random((pop_size, D2))
    fA0 = _true_fitness_on(tasks, pop_A, task_id=0)
    fB0 = _true_fitness_on(tasks, pop_B, task_id=1)
    FE += fA0.size + fB0.size
    _, cat_idx_A2B = _feature_layout(D)
    _, cat_idx_B2A = _feature_layout(D2)
    model_A2B = _build_mapping_regressor(categorical_features_indices=cat_idx_A2B)
    model_B2A = _build_mapping_regressor(categorical_features_indices=cat_idx_B2A)
    rec = TaskState(task=tasks, D=D, D2=D2, pop_A=pop_A, pop_B=pop_B, last_fA=fA0, last_fB=fB0, bestA_sofar=float(np.min(fA0)), bestB_sofar=float(np.min(fB0)), mig_every=mig_every)
    rec.hist_pop_A = [rec.pop_A.copy()]
    rec.hist_fA = [rec.last_fA.copy()]
    rec.hist_pop_B = [rec.pop_B.copy()]
    rec.hist_fB = [rec.last_fB.copy()]
    rec.mig_ratio_A = 0.2
    rec.mig_ratio_B = 0.2
    rec.mig_every = int(mig_every)
    rec.rng = rng
    for gen in range(n_generations):
        if FE >= max_fes:
            break
        do_B2A, do_A2B, n_mig_B2A, n_mig_A2B = _prepare_migration(rec, gen, pop_size, model_A2B, model_B2A, D=D, D2=D2)
        mig_B2A, mig_A2B = _perform_migration(rec, model_A2B, model_B2A, n_mig_B2A=n_mig_B2A, n_mig_A2B=n_mig_A2B, D=D, D2=D2, do_B2A=do_B2A, do_A2B=do_A2B)
        n_mig_B2A = int(mig_B2A.shape[0])
        n_mig_A2B = int(mig_A2B.shape[0])
        do_B2A = bool(n_mig_B2A > 0)
        do_A2B = bool(n_mig_A2B > 0)
        n_off_A = pop_size - n_mig_B2A
        n_off_B = pop_size - n_mig_A2B
        off_A, off_B = _make_offspring_DE(rec.pop_A, rec.pop_B, D, D2, rng, n_off_A=n_off_A, n_off_B=n_off_B, F=0.5, CR=0.9)
        new_A, new_B, fA, fB, nextA_labels, nextB_labels, n_eval_A, n_eval_B = _select_next_generation(tasks, rec.pop_A, rec.pop_B, off_A, off_B, mig_B2A, mig_A2B, rec.last_fA, rec.last_fB)
        _update_migration_feedback(rec, nextA_labels, nextB_labels, pop_size, n_mig_B2A=n_mig_B2A, n_mig_A2B=n_mig_A2B, do_B2A=do_B2A, do_A2B=do_A2B)
        FE = _update_records(rec, new_A, new_B, fA, fB, n_eval_A, n_eval_B, FE)
        rec.hist_pop_A.append(rec.pop_A.copy())
        rec.hist_fA.append(rec.last_fA.copy())
        rec.hist_pop_B.append(rec.pop_B.copy())
        rec.hist_fB.append(rec.last_fB.copy())
        if len(rec.hist_pop_A) > 6:
            rec.hist_pop_A = rec.hist_pop_A[-6:]
            rec.hist_fA = rec.hist_fA[-6:]
        if len(rec.hist_pop_B) > 6:
            rec.hist_pop_B = rec.hist_pop_B[-6:]
            rec.hist_fB = rec.hist_fB[-6:]
    return {'bestA': float(rec.bestA_sofar), 'bestB': float(rec.bestB_sofar)}
