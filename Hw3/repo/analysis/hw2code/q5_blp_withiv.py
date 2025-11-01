import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import numpy as np
import pandas as pd
from numpy.linalg import inv, solve
from scipy.stats import norm
from utils.latex_table import export_df_to_latex

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

def berry_contraction(s_obs, x_tilde, sigma, v, tol=1e-12, maxit=50000):
    s0 = 1.0 - float(s_obs.sum())
    delta = np.log(np.clip(s_obs, 1e-300, 1.0)) - math.log(max(s0, 1e-300))
    for _ in range(maxit):
        s_pred, _ = shares_from_delta(delta, x_tilde, sigma, v)
        new_delta = delta + (np.log(s_obs) - np.log(np.clip(s_pred,1e-300,1.0)))
        if np.max(np.abs(new_delta - delta)) < tol:
            return new_delta
        delta = new_delta
    return delta

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
    return Jmat, dsig.reshape(-1,1)

def build_designs(df):
    prod_d = pd.get_dummies(df["product"], prefix="prod", drop_first=True)
    city_d = pd.get_dummies(df["city"],    prefix="city", drop_first=True)
    time_d = pd.get_dummies(df["period"],  prefix="per",  drop_first=True)
    cost_shift = (df["distance"] * df["diesel"]).to_frame("cost_shift")
    X = pd.concat([df[["price"]], prod_d, city_d, time_d], axis=1).astype(float).values
    Z = pd.concat([cost_shift, prod_d, city_d, time_d], axis=1).astype(float).values
    W = inv(Z.T @ Z)
    return X, Z, W

def prepare_data():
    df = pd.read_csv("input/data_yoghurt.csv")
    df["market_id"] = list(zip(df["city"], df["period"]))
    df["sugar_per_g"] = df["sugar"]/df["weight"]
    mask = (df["city"]==1) & (df["period"]==1)
    xbar = df.loc[mask,"sugar_per_g"].mean()
    df["x_tilde"] = df["sugar_per_g"] - xbar
    good = df.groupby("market_id")["product"].nunique()==5
    ids = good[good].index.tolist()
    dfb = df[df["market_id"].isin(ids)].copy().reset_index(drop=True)
    gb = dfb.groupby("market_id", sort=True)
    market_rows = {k: np.array(v, dtype=int) for k,v in gb.indices.items()}
    return dfb, market_rows

def invert_all(df, markets, sigma, v):
    delta = np.zeros(df.shape[0])
    for m, rows in markets.items():
        rows_sorted = rows[np.argsort(df.loc[rows,"product"].values)]
        s_obs = df.loc[rows_sorted,"share"].values
        x_tilde = df.loc[rows_sorted,"x_tilde"].values
        delta[rows_sorted] = berry_contraction(s_obs, x_tilde, sigma, v)
    return delta

def blp_G(df, markets, sigma, v):
    X, Z, W = build_designs(df)
    delta = invert_all(df, markets, sigma, v)
    A = inv(X.T @ Z @ W @ Z.T @ X)
    beta = A @ (X.T @ Z @ W @ (Z.T @ delta))
    xi = delta - X @ beta
    n = X.shape[0]
    m = (Z.T @ xi)/n
    G = 1_000_000 * float(m.T @ W @ m)
    return G, float(beta[0])

dfb, markets = prepare_data()
v = vi_grid(50)
results = []
for s in [0, 10]:
    G, alpha = blp_G(dfb, markets, s, v)
    results.append({"sigma": s, "G": G, "alpha": alpha})
out_df = pd.DataFrame(results)

os.makedirs("tables", exist_ok=True)
export_df_to_latex(
    df=out_df,
    out_tex_path="tables/q5_blp_costshifter_G.tex",
    caption="BLP Objective With Cost Shifter (Two Sigma Values)",
    label="tab:q5_blp_costshifter_G",
    index=False,
    use_booktabs=True,
)
