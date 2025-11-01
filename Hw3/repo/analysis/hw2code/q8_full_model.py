import sys, os, time, numpy as np, pandas as pd
from numpy.linalg import inv, solve
from scipy.stats import norm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.latex_table import export_df_to_latex

np.set_printoptions(precision=6, suppress=True)

def now():
    return time.strftime("%H:%M:%S")

def vi_grid(N=50):
    u = np.linspace(0.1, 0.9, N)
    return norm.ppf(u)

def shares_from_delta(delta_vec, x_tilde_vec, sigma, v_draws):
    util = delta_vec.reshape(-1,1) + sigma * x_tilde_vec.reshape(-1,1) * v_draws.reshape(1,-1)
    a = np.maximum(0.0, np.max(util, axis=0, keepdims=True))
    expu = np.exp(util - a)
    denom = np.exp(0.0 - a) + np.sum(expu, axis=0, keepdims=True)
    sij = expu / denom
    sj = np.mean(sij, axis=1)
    return sj

def shares_and_individual(delta_vec, x_tilde_vec, sigma, v_draws):
    util = delta_vec.reshape(-1,1) + sigma * x_tilde_vec.reshape(-1,1) * v_draws.reshape(1,-1)
    a = np.maximum(0.0, np.max(util, axis=0, keepdims=True))
    expu = np.exp(util - a)
    denom = np.exp(0.0 - a) + np.sum(expu, axis=0, keepdims=True)
    sij = expu / denom
    s_mean = np.mean(sij, axis=1)
    return s_mean, sij

def berry_contraction(s_obs, x_tilde, sigma, v, tol=1e-12, maxit=80000):
    s_sum = float(s_obs.sum())
    s_sum = min(s_sum, 0.999999)
    s0 = 1.0 - s_sum
    delta = np.log(np.clip(s_obs, 1e-300, 1.0)) - np.log(max(s0, 1e-300))
    for _ in range(maxit):
        s_pred = shares_from_delta(delta, x_tilde, sigma, v)
        new_delta = delta + (np.log(s_obs) - np.log(np.clip(s_pred, 1e-300, 1.0)))
        if np.max(np.abs(new_delta - delta)) < tol:
            return new_delta
        delta = new_delta
    return delta

def market_delta_and_ddsigma(s_obs, x_tilde, sigma, v):
    delta = berry_contraction(s_obs, x_tilde, sigma, v)
    s_bar, s_ij = shares_and_individual(delta, x_tilde, sigma, v)
    J = np.zeros((len(delta), len(delta)))
    g = np.zeros(len(delta))
    x = x_tilde
    for k in range(v.size):
        s = s_ij[:, k]
        J += np.diag(s) - np.outer(s, s)
        g += v[k] * s * (x - np.dot(s, x))
    J /= v.size
    g /= v.size
    reg = 1e-12
    ddelta = solve(J + reg * np.eye(J.shape[0]), -g)
    return delta, ddelta

def prepare_data_with_z(path="data_yoghurt.csv", balanced_only=False):
    t0 = time.time()
    df = pd.read_csv(path)
    df["market_id"] = list(zip(df["city"], df["period"]))
    df["sugar_per_g"] = df["sugar"] / df["weight"]
    mask_ref = (df["city"]==1) & (df["period"]==1)
    xbar = df.loc[mask_ref, "sugar_per_g"].mean()
    df["x_tilde"] = df["sugar_per_g"] - xbar 
    z_market = df.groupby("market_id")["sugar_per_g"].sum().rename("z_ct")
    df = df.merge(z_market, on="market_id", how="left")
    if balanced_only:
        ok = (df.groupby("market_id")["product"].nunique() == 5)
        ids = ok[ok].index.tolist()
        df = df[df["market_id"].isin(ids)].copy().reset_index(drop=True)
    else:
        df = df.copy().reset_index(drop=True)
    gb = df.groupby("market_id", sort=True)
    markets = {k: df.index[df["market_id"] == k].to_numpy(dtype=int) for k in gb.groups.keys()}
    print(f"[{now()}] Loaded data: rows={len(df)}, markets={len(markets)} in {time.time()-t0:.2f}s")
    return df, markets

def build_XZ(df):
    prod_d = pd.get_dummies(df["product"], prefix="prod", drop_first=False)
    city_d = pd.get_dummies(df["city"],    prefix="city", drop_first=True)
    time_d = pd.get_dummies(df["period"],  prefix="per",  drop_first=True)
    cost_shift = (df["distance"] * df["diesel"]).to_frame("cost_shift")
    X = pd.concat([df[["price"]], prod_d, city_d, time_d], axis=1).astype(float).values
    Z = pd.concat([cost_shift, df[["z_ct"]], prod_d, city_d, time_d], axis=1).astype(float).values
    W = inv(Z.T @ Z)
    Pz = Z @ W @ Z.T
    return X, Z, W, Pz

def invert_all_markets_with_deriv(df, markets, sigma, v, print_every=200):
    n = df.shape[0]
    delta_all = np.zeros(n)
    ddelta_all = np.zeros(n)
    total = len(markets)
    for i, (mkt, rows) in enumerate(markets.items(), start=1):
        if i % print_every == 0:
            print(f"[{now()}] σ={sigma:8.4f}  market {i}/{total}")
        rows_sorted = rows[np.argsort(df.loc[rows, "product"].values)]
        s_obs = df.loc[rows_sorted, "share"].values
        x_tilde = df.loc[rows_sorted, "x_tilde"].values
        delta, ddelta = market_delta_and_ddsigma(s_obs, x_tilde, sigma, v)
        delta_all[rows_sorted] = delta
        ddelta_all[rows_sorted] = ddelta
    return delta_all, ddelta_all

def G_and_grad(df, markets, sigma, v, X, Z, W, Pz, K=1_000_000.0, verbose=False):
    t0 = time.time()
    delta, ddelta = invert_all_markets_with_deriv(df, markets, sigma, v)
    if verbose:
        print(f"[{now()}] Inversion+derivative finished in {time.time()-t0:.2f}s")
    A = inv(X.T @ Z @ W @ Z.T @ X)
    beta = A @ (X.T @ Z @ W @ (Z.T @ delta))
    xi = delta - X @ beta
    n = X.shape[0]
    m = (Z.T @ xi) / n
    G = K * float(m.T @ W @ m)
    dbeta = A @ (X.T @ Z @ W @ (Z.T @ ddelta))
    dxi = ddelta - X @ dbeta
    dm = (Z.T @ dxi) / n
    dG = 2.0 * K * float(m.T @ W @ dm)
    return G, dG, beta, xi, m, dm

def newton_optimize_sigma(df, markets, v, X, Z, W, Pz, sigma0=0.0, K=1_000_000.0, maxit=20, tol=1e-6):
    sigma = float(sigma0)
    history = []
    for it in range(1, maxit+1):
        t0 = time.time()
        G, dG, beta, xi, m, dm = G_and_grad(df, markets, sigma, v, X, Z, W, Pz, K, verbose=True)
        GN_H = 2.0 * K * float(dm.T @ W @ dm) + 1e-12
        step = - dG / GN_H
        lam = 1.0
        improved = False
        for _ in range(20):
            sigma_new = sigma + lam * step
            G_new, _, _, _, _, _ = G_and_grad(df, markets, sigma_new, v, X, Z, W, Pz, K, verbose=False)
            if G_new < G:
                improved = True
                break
            lam *= 0.5
        if not improved:
            sigma_new = sigma
        print(f"[{now()}] it={it:02d}  σ={sigma: .6f}  G={G: .6f}  dG={dG: .6e}  H~={GN_H: .6e}  step={step: .6e}  λ={lam:.2f}  took {time.time()-t0:.2f}s")
        history.append((sigma, G, dG))
        if abs(dG) < tol and abs(sigma_new - sigma) < tol:
            sigma = sigma_new
            break
        sigma = sigma_new
    G, dG, beta, xi, m, dm = G_and_grad(df, markets, sigma, v, X, Z, W, Pz, K, verbose=True)
    return sigma, beta, xi, m, dm, history

def ehw_se(df, X, Z, W, beta, xi, m, dm):
    n = X.shape[0]
    A = inv(X.T @ Z @ W @ Z.T @ X)

    # --- 正确的 EHW "meat": Z' diag(xi^2) Z / n ---
    xi2 = (xi**2).reshape(-1, 1)
    S = (Z.T @ (xi2 * Z)) / n

    # 2SLS/IV 的稳健方差（给 beta）
    Vbeta = A @ (X.T @ Z @ W @ S @ W @ Z.T @ X) @ A

    alpha_hat = float(beta[0])
    se_alpha = float(np.sqrt(max(Vbeta[0,0], 0.0)))

    # GMM 对 sigma 的稳健方差：D=dm = ∂m/∂σ（维度= instruments × 1）
    D = dm.reshape(-1,1)
    middle = (D.T @ W @ S @ W @ D)
    bread  = (D.T @ W @ D)
    Vsig = inv(bread) @ middle @ inv(bread) / n
    se_sigma = float(np.sqrt(max(Vsig[0,0], 0.0)))
    return alpha_hat, se_alpha, se_sigma

def run_estimation(data_path="data_yoghurt.csv", use_first_k_markets=None, sigma0=80):
    print(f"[{now()}] Start")
    t0 = time.time()
    df, markets_all = prepare_data_with_z(data_path, balanced_only=False)
    if use_first_k_markets is not None and use_first_k_markets > 0:
        keys = list(markets_all.keys())[:use_first_k_markets]
        df = df[df["market_id"].isin(keys)].copy().reset_index(drop=True)
        markets = {k: df.index[df["market_id"]==k].to_numpy(dtype=int) for k in keys}
        print(f"[{now()}] Use first {len(keys)} markets, rows={len(df)}")
    else:
        markets = markets_all
    X, Z, W, Pz = build_XZ(df)
    print(f"[{now()}] X shape={X.shape}, Z shape={Z.shape}")
    v = vi_grid(50)
    sigma_hat, beta_hat, xi, m, dm, hist = newton_optimize_sigma(df, markets, v, X, Z, W, Pz, sigma0=sigma0)
    alpha_hat, se_alpha, se_sigma = ehw_se(df, X, Z, W, beta_hat, xi, m, dm)
    res = pd.DataFrame({"parameter": ["alpha", "sigma"], "estimate":  [alpha_hat,  sigma_hat], "se_ehw":    [se_alpha,   se_sigma]})
    os.makedirs("tables", exist_ok=True)
    export_df_to_latex(
        df=res,
        out_tex_path="tables/blp_q8_newton_results.tex",
        caption="Newton Estimation Results For BLP ($\\alpha$ And $\\sigma$) With EHW SE",
        label="tab:blp_q8_newton_results",
        index=False,
        use_booktabs=True,
    )
    print(f"[{now()}] Done in {(time.time()-t0)/60:.2f} min. LaTeX → tables/blp_q8_newton_results.tex")
    print(res.to_string(index=False))
    return res

if __name__ == "__main__":
    run_estimation(data_path="input/data_yoghurt.csv", use_first_k_markets=None, sigma0=140)
