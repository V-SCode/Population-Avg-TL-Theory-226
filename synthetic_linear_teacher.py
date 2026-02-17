def generate_linear_teacher_data(D: int, P: int, rho: float = 1.0, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.randn(P, D).astype(np.float32)

    beta = rng.randn(D).astype(np.float32)
    beta = beta / np.linalg.norm(beta)

    if abs(rho - 1.0) < 1e-8:
        w_tgt = beta.copy()
    else:
        # Sample random orthogonal component
        v = rng.randn(D).astype(np.float32)
        v -= (beta @ v) * beta  # make orthogonal to beta
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-8:
            # degenerate case, resample
            v = rng.randn(D).astype(np.float32)
            v -= (beta @ v) * beta
            v_norm = np.linalg.norm(v)
        v /= v_norm
        w_tgt = rho * beta + math.sqrt(1.0 - rho**2) * v

    y_src = (X @ beta) / math.sqrt(D)
    y_tgt = (X @ w_tgt) / math.sqrt(D)

    X_t = torch.from_numpy(X)
    y_src_t = torch.from_numpy(y_src)
    y_tgt_t = torch.from_numpy(y_tgt)

    return X_t, y_src_t, y_tgt_t, beta, w_tgt
