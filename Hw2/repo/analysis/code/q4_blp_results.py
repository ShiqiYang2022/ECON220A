import numpy as np, pandas as pd, math
from numpy.linalg import inv, solve
from scipy.stats import norm

def vi_grid(N=50):
    u = np.linspace(0.1, 0.9, N)
    return norm.ppf(u)

def shares_from_delta(delta_vec, x_tilde_vec, sigma, v_draws):
    util = delta_vec.reshape(-1,1) + sigma * x_tilde_vec.reshape(-1,1) * v_draws.reshape(1,-1)
    a = np.max(util, axis=0, keepdims=True)
    expu = np.exp(util - a)
    denom = np.exp(-a) + np.sum(expu, axis=0, keepdims=True)
    sij = expu / denom
    sj = np.mean(sij, axis=1)
    return sj, sij

def berry_contraction_market(s_obs, x_tilde, sigma, v, tol=1e-12, maxit=50000):
    s0 = 1.0 - float(s_obs.sum())
    delta = np.log(np.clip(s_obs, 1e-300, 1.0)) - math.log(max(s0, 1e-300))
    for _ in range(maxit):
        s_pred, _ = shares_from_delta(delta, x_tilde, sigma, v)
        new_delta = delta + (np.log(s_obs) - np.log(np.clip(s_pred,1e-300,1.0)))
        if np.max(np.abs(new_delta - delta)) < tol:
            return new_delta
        delta = new_delta
    return None

def jacobian_and_gsig(delta_vec, x_tilde_vec, sigma, v):
    s_pred, sij = shares_from_delta(delta_vec, x_tilde_vec, sigma, v)
    J = len(delta_vec)
    Jmat = np.zeros((J,J))
    for n in range(sij.shape[1]):
        s_i = sij[:,n:n+1]
        Jmat += (np.diagflat(s_i) - s_i @ s_i.T)
    Jmat /= sij.shape[1]
    dsig = np.zeros(J)
    x = x_tilde_vec.reshape(-1,1)
    for n in range(sij.shape[1]):
        s_i = sij[:,n:n+1]
        v_i = v[n]
        xbar = float(s_i.T @ x)
        dsig += v_i * (s_i.flatten() * (x.flatten() - xbar))
    dsig /= sij.shape[1]
    return Jmat, dsig.reshape(-1,1), s_pred

def build_designs(df):
    prod_d = pd.get_dummies(df["product"], prefix="prod", drop_first=True)
    city_d = pd.get_dummies(df["city"],    prefix="city", drop_first=True)
    time_d = pd.get_dummies(df["period"],  prefix="per",  drop_first=True)
    X = pd.concat([df[["price"]], prod_d, city_d, time_d], axis=1).astype(float).values
    Z = pd.concat([df[["distance","diesel"]], prod_d, city_d, time_d], axis=1).astype(float).values
    W = inv(Z.T @ Z)
    return X, Z, W

def prepare_data():
    df = pd.read_csv("input/data_yoghurt.csv")
    df["market_id"] = list(zip(df["city"], df["period"]))
    df["sugar_per_g"] = df["sugar"]/df["weight"]
    mask_ref = (df["city"]==1) & (df["period"]==1)
    xbar = df.loc[mask_ref, "sugar_per_g"].mean()
    df["x_tilde"] = df["sugar_per_g"] - xbar
    balanced = (df.groupby("market_id")["product"].nunique()==5)
    ids = balanced[balanced].index.tolist()
    dfb = df[df["market_id"].isin(ids)].copy().reset_index(drop=True)
    gb = dfb.groupby("market_id", sort=True)
    market_rows = {k: np.array(v, dtype=int) for k,v in gb.indices.items()}
    return dfb, market_rows

def invert_all_markets(df, market_rows, sigma, v):
    delta = np.zeros(df.shape[0])
    for m, rows in market_rows.items():
        rows_sorted = rows[np.argsort(df.loc[rows,"product"].values)]
        s_obs = df.loc[rows_sorted,"share"].values
        x_tilde = df.loc[rows_sorted,"x_tilde"].values
        d = berry_contraction_market(s_obs, x_tilde, sigma, v)
        if d is None:
            raise RuntimeError(f"contraction failed for market {m}")
        delta[rows_sorted] = d
    return delta

def blp_objective_and_grad(sigma, df, market_rows, v):
    X, Z, W = build_designs(df)
    delta = invert_all_markets(df, market_rows, sigma, v)
    A = inv(X.T @ Z @ W @ Z.T @ X)
    beta = A @ (X.T @ Z @ W @ (Z.T @ delta))
    xi = delta - X @ beta
    n = X.shape[0]
    m = (Z.T @ xi)/n
    K = 1_000_000.0
    G = float(K * (m.T @ W @ m))
    ddelta = np.zeros_like(delta)
    for mkt, rows in market_rows.items():
        rows_sorted = rows[np.argsort(df.loc[rows,"product"].values)]
        Jmat, gsig, _ = jacobian_and_gsig(delta[rows_sorted], df.loc[rows_sorted,"x_tilde"].values, sigma, v)
        try:
            ddelta[rows_sorted] = solve(Jmat, -gsig).flatten()
        except np.linalg.LinAlgError:
            ddelta[rows_sorted] = solve(Jmat + 1e-12*np.eye(Jmat.shape[0]), -gsig).flatten()
    dbeta = A @ (X.T @ Z @ W @ (Z.T @ ddelta))
    dxi = ddelta - X @ dbeta
    dm = (Z.T @ dxi)/n
    dG = float(2.0 * K * (m.T @ W @ dm))
    return G, dG, {"beta":beta, "xi":xi, "delta":delta}

dfb, market_rows = prepare_data()
v = vi_grid(50)

rows_for_speed = list(market_rows.keys())[:120]
df_small = dfb[dfb["market_id"].isin(rows_for_speed)].copy().reset_index(drop=True)
gb_small = df_small.groupby("market_id", sort=True)
market_rows_small = {k: np.array(v, dtype=int) for k,v in gb_small.indices.items()}

sigmas = [0.0, 2.0]
results = []
for s in sigmas:
    G, dG, res = blp_objective_and_grad(s, df_small, market_rows_small, v)
    results.append({"sigma": s, "G": float(G), "dG": float(dG), "alpha": float(res["beta"][0])})

pd.DataFrame(results).to_csv("output/q4_blp_results_two_sigmas.csv", index=False)
