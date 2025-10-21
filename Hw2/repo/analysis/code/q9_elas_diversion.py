import os, numpy as np, pandas as pd, math
from numpy.linalg import inv
from scipy.stats import norm

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
    return sj, sij

def berry_contraction(s_obs, x_tilde, sigma, v, tol=1e-12, maxit=80000):
    s_sum = float(s_obs.sum())
    s_sum = min(s_sum, 0.999999)
    s0 = 1.0 - s_sum
    delta = np.log(np.clip(s_obs, 1e-300, 1.0)) - np.log(max(s0, 1e-300))
    for _ in range(maxit):
        s_pred, _ = shares_from_delta(delta, x_tilde, sigma, v)
        new_delta = delta + (np.log(s_obs) - np.log(np.clip(s_pred, 1e-300, 1.0)))
        if np.max(np.abs(new_delta - delta)) < tol:
            return new_delta
        delta = new_delta
    return delta

def prepare_data_with_z(path="input/data_yoghurt.csv", center_mode="market"):
    df = pd.read_csv(path)
    df["market_id"] = list(zip(df["city"], df["period"]))
    df["sugar_per_g"] = df["sugar"] / df["weight"]
    if center_mode == "market":
        df["x_tilde"] = df["sugar_per_g"] - df.groupby("market_id")["sugar_per_g"].transform("mean")
    elif center_mode == "global_ref_11":
        xbar = df.loc[(df["city"]==1)&(df["period"]==1), "sugar_per_g"].mean()
        df["x_tilde"] = df["sugar_per_g"] - xbar
    else:
        raise ValueError("center_mode must be 'market' or 'global_ref_11'")
    return df

def read_alpha_sigma_or_set_defaults(path="output/blp_q8_newton_results.csv",
                                     alpha_default=-3.044808357870437,
                                     sigma_default=36.69195461195635):
    if os.path.exists(path):
        tab = pd.read_csv(path)
        a = float(tab.loc[tab["parameter"]=="alpha", "estimate"].values[0])
        s = float(tab.loc[tab["parameter"]=="sigma", "estimate"].values[0])
        return a, s
    return alpha_default, sigma_default

def q9_elasticity_and_diversion(data_path="input/data_yoghurt.csv",
                                center_mode="market",
                                city=1, period=1,
                                outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    alpha_hat, sigma_hat = read_alpha_sigma_or_set_defaults()
    alpha_hat = alpha_hat * -1
    df = prepare_data_with_z(data_path, center_mode=center_mode)
    mask = (df["city"]==city) & (df["period"]==period)
    dfm = df.loc[mask, ["product","price","share","x_tilde"]].copy()
    dfm = dfm.sort_values("product").reset_index(drop=True)
    prod_ids = dfm["product"].tolist()
    p = dfm["price"].to_numpy(dtype=float)
    s_obs = dfm["share"].to_numpy(dtype=float)
    x_tilde = dfm["x_tilde"].to_numpy(dtype=float)
    v = vi_grid(50)
    delta = berry_contraction(s_obs, x_tilde, sigma_hat, v)
    s_bar, s_ij = shares_from_delta(delta, x_tilde, sigma_hat, v)
    N = s_ij.shape[1]
    M = (s_ij @ s_ij.T) / N
    own_part = s_bar - np.diag(M)
    dSdp = alpha_hat * M
    for j in range(len(own_part)):
        dSdp[j,j] = -alpha_hat * own_part[j]
    E = dSdp * (p.reshape(1,-1)) / s_bar.reshape(-1,1)
    denom = own_part
    D = np.zeros_like(M)
    for l in range(len(denom)):
        for k in range(len(denom)):
            if k==l: 
                D[l,k] = 0.0
            else:
                D[l,k] = M[k,l] / max(denom[l], 1e-14)
    el_df = pd.DataFrame(E, index=[f"j={j}" for j in prod_ids], columns=[f"p@{j}" for j in prod_ids])
    dv_df = pd.DataFrame(D, index=[f"from {j}" for j in prod_ids], columns=[f"to {j}" for j in prod_ids])
    el_path = os.path.join(outdir, f"q9_elasticities_city{city}_period{period}.csv")
    dv_path = os.path.join(outdir, f"q9_diversion_city{city}_period{period}.csv")
    el_df.to_csv(el_path)
    dv_df.to_csv(dv_path)
    print(f"alpha_hat={alpha_hat:.6f}, sigma_hat={sigma_hat:.6f}")
    print("Elasticities matrix (rows: demand for j; cols: price of l):")
    print(el_df.round(4).to_string())
    print("Diversion ratios (rows: from l; cols: to k):")
    print(dv_df.round(4).to_string())
    print(f"saved to {el_path} and {dv_path}")
    return el_df, dv_df, (alpha_hat, sigma_hat)

if __name__ == "__main__":
    q9_elasticity_and_diversion(
        data_path="input/data_yoghurt.csv",
        center_mode="market",
        city=1, period=1,
        outdir="output"
    )
