def make_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_source(
    model: MuPMLP,
    X: torch.Tensor,
    y: torch.Tensor,
    lambda_reg: float = 0.0,
    eta0: float = 1e-4,
    n_epochs: int = 200,
    batch_size: int = 32,
    verbose: bool = True,
):
    
    device = model.device
    X = X.to(device)
    y = y.to(device)

    lr = eta0 * (model.gamma0 ** 2) * model.N
    optimizer = optim.SGD(model.parameters(), lr=lr)

    P = X.shape[0]

    for epoch in range(n_epochs):
        loader = make_loader(X, y, batch_size, shuffle=True)
        epoch_loss = 0.0

        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            mse = 0.5 * torch.mean((preds - yb) ** 2)

            # weight decay on both W and w
            wd = 0.0
            if lambda_reg > 0.0:
                wd = 0.5 * lambda_reg * (model.W.pow(2).sum() + model.w.pow(2).sum())

            loss = mse + wd
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
            print(f"[Source] Epoch {epoch+1:4d}/{n_epochs}, loss = {epoch_loss / P:.6f}")

    return model


def train_target_transfer(
    model: MuPMLP,
    X: torch.Tensor,
    y: torch.Tensor,
    W_source: torch.Tensor,
    delta: float,
    lambda_reg: float = 0.0,
    eta0: float = 1e-4,
    n_epochs: int = 200,
    batch_size: int = 32,
    verbose: bool = True,
):
    
    device = model.device
    X = X.to(device)
    y = y.to(device)
    W_src = W_source.to(device)

    # Learning rate scaling
    lr = eta0 * (model.gamma0 ** 2) * model.N
    optimizer = optim.SGD(model.parameters(), lr=lr)

    P = X.shape[0]

    for epoch in range(n_epochs):
        loader = make_loader(X, y, batch_size, shuffle=True)
        epoch_loss = 0.0

        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            mse = 0.5 * torch.mean((preds - yb) ** 2)

            # Weight decay
            wd = 0.0
            if lambda_reg > 0.0:
                wd = 0.5 * lambda_reg * (model.W.pow(2).sum() + model.w.pow(2).sum())

            # Elastic penalty on first-layer weights
            elastic = 0.0
            if delta > 0.0:
                elastic = 0.5 * delta * (model.W - W_src).pow(2).sum()

            loss = mse + wd + elastic
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
            print(
                f"[Target δ={delta}] Epoch {epoch+1:4d}/{n_epochs}, "
                f"loss = {epoch_loss / P:.6f}"
            )

    return model


def evaluate_mse(model: MuPMLP, X: torch.Tensor, y: torch.Tensor) -> float:
    device = model.device
    model.eval()
    with torch.no_grad():
        preds = model(X.to(device))
        mse = torch.mean((preds - y.to(device)) ** 2).item()
    model.train()
    return mse


def demo_t1():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = 10
    N = 500
    P1 = 50
    P2 = 50
    gamma0 = 1.0
    lambda_reg = 1e-3
    eta0 = 1e-4
    n_epochs = 200

    print("=== Generating synthetic linear teacher data ===")
    # For T1 demo, use the SAME teacher for source and target (rho=1)
    X1, y1_src, y1_tgt, beta, w_tgt = generate_linear_teacher_data(D, P1, rho=1.0, seed=0)
    X2, y2_src, y2_tgt, _, _ = generate_linear_teacher_data(D, P2, rho=1.0, seed=1)

    # Test set
    X_test, y_test_src, y_test_tgt, _, _ = generate_linear_teacher_data(D, 1000, rho=1.0, seed=2)

    print("\n=== Training source model ===")
    source_model = MuPMLP(D=D, N=N, gamma0=gamma0, activation="linear", device=device)
    train_source(
        source_model,
        X1,
        y1_src,
        lambda_reg=lambda_reg,
        eta0=eta0,
        n_epochs=n_epochs,
        batch_size=16,
        verbose=True,
    )

    source_mse_train = evaluate_mse(source_model, X1, y1_src)
    source_mse_test = evaluate_mse(source_model, X_test, y_test_src)
    print(f"\n[Source] Train MSE: {source_mse_train:.6f}, Test MSE: {source_mse_test:.6f}")

    # Save source first-layer weights
    W_src = source_model.W.detach().clone()

    print("\n=== Training target model with δ = 0 (pure fine-tuning) ===")
    target_model_delta0 = MuPMLP(D=D, N=N, gamma0=gamma0, activation="linear", device=device)
    # initialize W from source; reinit w
    with torch.no_grad():
        target_model_delta0.W.copy_(W_src)

    train_target_transfer(
        target_model_delta0,
        X2,
        y2_tgt,
        W_source=W_src,
        delta=0.0,
        lambda_reg=lambda_reg,
        eta0=eta0,
        n_epochs=n_epochs,
        batch_size=16,
        verbose=True,
    )

    tgt0_mse_train = evaluate_mse(target_model_delta0, X2, y2_tgt)
    tgt0_mse_test = evaluate_mse(target_model_delta0, X_test, y_test_tgt)
    print(f"[Target δ=0] Train MSE: {tgt0_mse_train:.6f}, Test MSE: {tgt0_mse_test:.6f}")

    print("\n=== Training target model with δ = 1 (elastic coupling) ===")
    target_model_delta1 = MuPMLP(D=D, N=N, gamma0=gamma0, activation="linear", device=device)
    with torch.no_grad():
        target_model_delta1.W.copy_(W_src)

    train_target_transfer(
        target_model_delta1,
        X2,
        y2_tgt,
        W_source=W_src,
        delta=1.0,
        lambda_reg=lambda_reg,
        eta0=eta0,
        n_epochs=n_epochs,
        batch_size=16,
        verbose=True,
    )

    tgt1_mse_train = evaluate_mse(target_model_delta1, X2, y2_tgt)
    tgt1_mse_test = evaluate_mse(target_model_delta1, X_test, y_test_tgt)
    print(f"[Target δ=1] Train MSE: {tgt1_mse_train:.6f}, Test MSE: {tgt1_mse_test:.6f}")

    print("\n=== Summary (T1 demo) ===")
    print(f"Source test MSE      : {source_mse_test:.6f}")
    print(f"Target test MSE (δ=0): {tgt0_mse_test:.6f}")
    print(f"Target test MSE (δ=1): {tgt1_mse_test:.6f}")


if __name__ == "__main__":
    demo_t1()
