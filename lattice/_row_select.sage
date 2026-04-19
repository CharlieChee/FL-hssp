#!/usr/bin/env sage
"""Row / column selection strategies used to assemble an mHLCP instance from CNN tensors.

Three row-selection modes are exposed (set via ``MHLCP_ROW_MODE``):

  - ``random`` (default)  : ``select_rows_random_full_rank`` — random rows that keep ``rank(X)==n``.
  - ``relu_stratified``   : ``select_rows_relu_stratified``  — rows stratified by ReLU Hamming weight.
  - ``condnum``           : ``select_rows_condnum_greedy``   — greedy by (sub-)matrix condition number.

Plus column-selection (``select_cols_full_rank``), ReLU-mask loading
(``load_relu_mask``), random-X synthesis (``gen_random_sparse_matrix``), and
the post-construction sanity check ``ok_instance``.

Loaded by ``run_mhlcp_cnn_quant_x.sage`` after the cwd has been switched to the
repo root and ``multi_dim_hssp.sage`` has been ``load()``-ed.
"""
import numpy as np

def select_cols_full_rank(A_pool, n, l, seed=0, max_attempts=600):
    """
    Randomly choose l columns from A_pool (n x D), prioritizing column rank = min(n, l) (which is full rank n when l == n).
    """
    rng = np.random.RandomState(int(seed))
    rows, dim = A_pool.shape
    if rows != n:
        raise ValueError("A_pool has %s rows != n=%s (batch_size and n are expected to match)" % (rows, n))
    if dim < l:
        raise ValueError("A_pool has %s columns < l=%s; cannot sample columns of A." % (dim, l))
    target_rank = int(min(n, l))
    all_idx = np.arange(dim, dtype=np.int64)
    for _ in range(max_attempts):
        cols = rng.choice(all_idx, size=int(l), replace=False)
        A_sub = A_pool[:, cols]
        if int(np.linalg.matrix_rank(A_sub.astype(np.float64))) >= target_rank:
            return cols

    # Fallback: greedily build full rank, then top up the column count.
    order = rng.permutation(dim)
    picked = []
    cur = None
    for c in order:
        col = A_pool[:, int(c) : int(c) + 1].astype(np.float64)
        if cur is None:
            cur = col
            picked.append(int(c))
        else:
            cand = np.hstack([cur, col])
            if int(np.linalg.matrix_rank(cand)) > int(np.linalg.matrix_rank(cur)):
                cur = cand
                picked.append(int(c))
        if len(picked) >= target_rank:
            break
    if len(picked) < target_rank:
        raise RuntimeError("Cannot sample a column subset of rank %s from fc_input." % target_rank)
    rest = [i for i in range(dim) if i not in set(picked)]
    rng.shuffle(rest)
    picked.extend(rest[: int(l - len(picked))])
    return np.array(picked[:l], dtype=np.int64)


def load_relu_mask(batch_size):
    relu_path = resolve_relu_path(batch_size)
    relu_mask = np.load(relu_path)
    relu_int = relu_mask.astype(np.int64)
    if relu_int.ndim != 2:
        raise ValueError("relu_mask must be a 2D matrix, got shape=%s" % (relu_int.shape,))
    num_rows, n = relu_int.shape
    print("Loaded ReLU mask (used for stratified row selection): %s  shape=%s" % (relu_path, relu_int.shape))
    return relu_path, relu_int, num_rows, n


def select_rows_relu_stratified(relu_int, X_int, n, target_rows, seed=None):
    """Legacy mode: bucket rows by ReLU Hamming weight; rank decisions are made on the float-cast version of X_int (the integer pool)."""
    if seed is not None:
        np.random.seed(int(seed))
    pre_float = X_int.astype(np.float64)
    num_rows, dim = relu_int.shape
    if pre_float.shape != relu_int.shape:
        raise ValueError("Shape mismatch: %s vs %s" % (pre_float.shape, relu_int.shape))
    if dim != n:
        raise ValueError("Column count %s != n=%s" % (dim, n))
    target_rows = min(int(target_rows), int(num_rows))
    ones_counts = relu_int.sum(axis=1)
    mid = int(n) // 2
    lo, hi = max(0, mid - 2), min(int(n), mid + 2)
    lo2, hi2 = max(0, mid - 4), min(int(n), mid + 4)
    balanced_indices = [i for i, c in enumerate(ones_counts) if lo <= c <= hi]
    secondary_indices = [
        i for i, c in enumerate(ones_counts) if lo2 <= c <= hi2 and i not in balanced_indices
    ]
    other_indices = [i for i in range(num_rows) if i not in balanced_indices and i not in secondary_indices]
    np.random.shuffle(balanced_indices)
    np.random.shuffle(secondary_indices)
    np.random.shuffle(other_indices)
    selected = []
    current_matrix = None

    def fill_from(pool):
        nonlocal selected, current_matrix
        for idx in pool:
            if idx in selected:
                continue
            row = pre_float[idx : idx + 1, :]
            if len(selected) < n:
                if current_matrix is None:
                    current_matrix = row
                    selected.append(idx)
                else:
                    candidate = np.vstack([current_matrix, row])
                    if np.linalg.matrix_rank(candidate.astype(float)) > np.linalg.matrix_rank(
                        current_matrix.astype(float)
                    ):
                        current_matrix = candidate
                        selected.append(idx)
            else:
                selected.append(idx)
            if len(selected) == target_rows:
                break

    for pool in (balanced_indices, secondary_indices, other_indices):
        if len(selected) < target_rows:
            fill_from(pool)
    if len(selected) < n:
        raise RuntimeError("ReLU-stratified mode could not assemble full column rank n=%s" % n)
    return np.array(selected, dtype=np.int64)


def select_rows_random_full_rank(X_int, n, target_rows, seed=0, max_attempts=400):
    """
    Sort rows by a "balance" score over the count of zero entries, greedily build column rank n, then fill up to target_rows in sorted order.

    B_X > 1 (general integer matrix):
        priority: #zeros=n/2 -> n/2-1 -> ... -> 0 -> n/2+1 -> ... -> n
        Prefers dense rows (fewer zeros before more zeros).

    B_X <= 1 (0/1 matrix):
        priority: ascending |#zeros - n/2| (symmetric: 0 and 1 are treated equally).

    Within the same priority bucket, ordering is randomized using seed.
    """
    rng = np.random.RandomState(int(seed))
    num_rows, dim = X_int.shape
    if dim != n:
        raise ValueError("X_int has %s columns != n=%s" % (dim, n))
    target_rows = min(int(target_rows), int(num_rows))

    zero_counts = np.sum(X_int == 0, axis=1)
    half_n = n / 2.0
    b_max = int(np.max(np.abs(X_int)))

    if b_max <= 1:
        priority = np.abs(zero_counts - half_n)
    else:
        priority = np.where(
            zero_counts <= half_n,
            half_n - zero_counts,
            zero_counts.astype(np.float64),
        )

    tiebreaker = rng.random(num_rows) * 0.01
    order = np.argsort(priority + tiebreaker)

    selected = []
    skipped = []
    cur_matrix = None

    for idx in order:
        idx = int(idx)
        row = X_int[idx : idx + 1, :].astype(np.float64)
        if cur_matrix is None:
            cur_matrix = row
            selected.append(idx)
        else:
            cand = np.vstack([cur_matrix, row])
            if int(np.linalg.matrix_rank(cand)) > int(np.linalg.matrix_rank(cur_matrix)):
                cur_matrix = cand
                selected.append(idx)
            else:
                skipped.append(idx)
        if len(selected) >= n and int(np.linalg.matrix_rank(cur_matrix)) >= n:
            break

    if cur_matrix is None or int(np.linalg.matrix_rank(cur_matrix)) < n:
        raise RuntimeError(
            "Even after sorting by balance score, could not assemble full column rank n=%d (rows in pool=%d)." % (n, num_rows)
        )

    selected_set = set(selected)
    remaining = [int(i) for i in order if int(i) not in selected_set]
    for idx in remaining:
        if len(selected) >= target_rows:
            break
        selected.append(idx)

    if len(selected) < target_rows:
        raise RuntimeError(
            "Not enough rows: selected %d rows < target_rows=%d" % (len(selected), target_rows)
        )

    zc_sel = zero_counts[np.array(selected[:target_rows])]
    print(
        "Row selection (balance-score sort): B_max=%d  zero-count distribution over first n rows: min=%d median=%.0f max=%d"
        % (b_max, int(np.min(zc_sel[:n])), float(np.median(zc_sel[:n])), int(np.max(zc_sel[:n])))
    )

    return np.array(selected[:target_rows], dtype=np.int64)


def select_rows_condnum_greedy(X_int, n, target_rows, seed=0, max_attempts=400):
    """
    Condition-number-optimized row selection: greedily pick the row subset that minimizes the condition number of (X^T X).

    Procedure:
    1. Use select_rows_random_full_rank to obtain the initial n rows (guarantees rank).
    2. Append subsequent rows one at a time, choosing from the candidate pool the row that grows the condition number the least.
    3. The resulting X submatrix is more "uniform", which benefits Step2 recoverBox.
    """
    rng = np.random.RandomState(int(seed))
    num_rows, dim = X_int.shape
    if dim != n:
        raise ValueError("X_int has %s columns != n=%s" % (dim, n))
    target_rows = min(int(target_rows), int(num_rows))

    X_float = X_int.astype(np.float64)

    # Phase 1: greedily build full rank (identical to the original method).
    zero_counts = np.sum(X_int == 0, axis=1)
    half_n = n / 2.0
    b_max = int(np.max(np.abs(X_int)))
    if b_max <= 1:
        priority = np.abs(zero_counts - half_n)
    else:
        priority = np.where(zero_counts <= half_n, half_n - zero_counts,
                            zero_counts.astype(np.float64))
    tiebreaker = rng.random(num_rows) * 0.01
    order = np.argsort(priority + tiebreaker)

    selected = []
    cur_matrix = None
    for idx in order:
        idx = int(idx)
        row = X_float[idx:idx+1, :]
        if cur_matrix is None:
            cur_matrix = row
            selected.append(idx)
        else:
            cand = np.vstack([cur_matrix, row])
            if int(np.linalg.matrix_rank(cand)) > int(np.linalg.matrix_rank(cur_matrix)):
                cur_matrix = cand
                selected.append(idx)
        if len(selected) >= n and int(np.linalg.matrix_rank(cur_matrix)) >= n:
            break

    if cur_matrix is None or int(np.linalg.matrix_rank(cur_matrix)) < n:
        raise RuntimeError("Could not assemble full column rank n=%d" % n)

    # Phase 2: append additional rows greedily by condition number.
    selected_set = set(selected)
    pool = [int(i) for i in range(num_rows) if int(i) not in selected_set]
    rng.shuffle(pool)

    while len(selected) < target_rows and pool:
        best_idx = None
        best_cond = float('inf')
        # Randomly draw up to 50 candidates from the pool to evaluate (to avoid being too slow).
        candidates = pool[:min(50, len(pool))]
        for idx in candidates:
            trial = np.vstack([cur_matrix, X_float[idx:idx+1, :]])
            try:
                c = np.linalg.cond(trial)
            except Exception:
                c = float('inf')
            if c < best_cond:
                best_cond = c
                best_idx = idx
        if best_idx is not None:
            cur_matrix = np.vstack([cur_matrix, X_float[best_idx:best_idx+1, :]])
            selected.append(best_idx)
            pool.remove(best_idx)
        else:
            break

    # If still not enough, pad with the remaining rows.
    for idx in pool:
        if len(selected) >= target_rows:
            break
        selected.append(idx)

    cond_final = np.linalg.cond(X_float[np.array(selected[:target_rows]), :])
    print("Row selection (condition-number greedy): B_max=%d  cond(X)=%.1f  zero counts over first n rows: min=%d max=%d" %
          (b_max, cond_final,
           int(np.min(zero_counts[np.array(selected[:n])])),
           int(np.max(zero_counts[np.array(selected[:n])]))))

    return np.array(selected[:target_rows], dtype=np.int64)


def gen_random_sparse_matrix(rows, cols, b_max, non_zero_ratio, seed=0, require_rank=0, max_attempts=400):
    """
    Generate a rows x cols random sparse integer matrix with non-zero entries in [1, b_max];
    the non-zero proportion is controlled by non_zero_ratio. If require_rank>0, the result is guaranteed to have rank >= require_rank.
    """
    rng = np.random.RandomState(int(seed))
    for _ in range(max_attempts):
        mask = (rng.random((rows, cols)) < non_zero_ratio)
        values = rng.randint(1, int(b_max) + 1, size=(rows, cols))
        M = (mask * values).astype(np.int64)
        if require_rank <= 0 or int(np.linalg.matrix_rank(M.astype(np.float64))) >= require_rank:
            return M
    raise RuntimeError(
        "Random generation of a %dx%d matrix (non_zero=%.2f, B=%d) after %d attempts still could not reach rank=%d"
        % (rows, cols, non_zero_ratio, b_max, max_attempts, require_rank)
    )


def ok_instance(H):
    l = H.l
    try:
        r = rank(Matrix(Integers(H.x0), H.B[:l, :l]))
        return r >= l
    except Exception:
        return False


