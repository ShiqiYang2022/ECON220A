import sys, os, numpy as np, pandas as pd
from numpy.linalg import inv, solve
from scipy.stats import norm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.latex_table import export_df_to_latex

def vi_grid(N=50):
    u = np.linspace(0.1, 0.9, N)
    return norm.ppf(u)

def shares_from_delta(delta_vec, x_tilde_vec, sigma, v_draws):
    util = delta_vec.reshape(-1,1) + sigma * x_tilde_vec.reshape(-1,1) * v_draws.reshape(1,-1)
    a = np.maximum(0.0, np.max(util, axis=0, keepdims=True))
    expu = np.exp(util - a)
    denom = np.exp(0.0 - a) + np.sum(expu, axis=0, keepdims=True)
    sij = expu / denom
    s_mean = np.mean(sij, axis=1)
    return s_mean, sij

def berry_contraction(s_obs, x_tilde, sigma, v, tol=1e-12, maxit=80000):
    s_sum = float(s_obs.sum()); s_sum = min(s_sum, 0.999999)
    s0 = 1.0 - s_sum
    delta = np.log(np.clip(s_obs, 1e-300, 1.0)) - np.log(max(s0, 1e-300))
    for _ in range(maxit):
        s_pred, _ = shares_from_delta(delta, x_tilde, sigma, v)
        new_delta = delta + (np.log(s_obs) - np.log(np.clip(s_pred,1e-300,1.0)))
        if np.max(np.abs(new_delta - delta)) < tol: return new_delta
        delta = new_delta
    return delta

def read_beta_sigma(csv_path="output/blp_q8_newton_results.csv", beta_default=-3.5, sigma_default=80.7):
    if os.path.exists(csv_path):
        tab = pd.read_csv(csv_path)
        beta_price = float(tab.loc[tab["parameter"]=="alpha","estimate"].values[0])
        sigma_hat  = float(tab.loc[tab["parameter"]=="sigma","estimate"].values[0])
    else:
        beta_price, sigma_hat = beta_default, sigma_default
    alpha_struct = -beta_price
    return alpha_struct, sigma_hat

def build_O(prod_ids, mode="single"):
    J = len(prod_ids)
    if mode == "single":
        return np.eye(J), ["prod"+str(p) for p in prod_ids]
    elif mode == "danone345":
        owners = []
        for p in prod_ids:
            if p in (3,4,5): owners.append("Danone")
            elif p==1: owners.append("Yoplait")
            elif p==2: owners.append("Chobani")
            else: owners.append(f"prod{p}")
        O = (np.equal.outer(owners, owners)).astype(float)
        return O, owners
    else:
        raise ValueError("unknown O mode")

def compute_market_costs(data_path="input/data_yoghurt.csv", city=1, period=1, center_mode="market", O_mode="single", alpha=None, sigma=None):
    df = pd.read_csv(data_path)
    df["market_id"] = list(zip(df["city"], df["period"]))
    df["sugar_per_g"] = df["sugar"]/df["weight"]
    if center_mode=="market":
        df["x_tilde"] = df["sugar_per_g"] - df.groupby("market_id")["sugar_per_g"].transform("mean")
    else:
        xbar = df.loc[(df["city"]==1)&(df["period"]==1),"sugar_per_g"].mean()
        df["x_tilde"] = df["sugar_per_g"] - xbar
    msk = (df["city"]==city) & (df["period"]==period)
    mkt = df.loc[msk, ["product","price","share","x_tilde"]].copy().sort_values("product").reset_index(drop=True)
    p = mkt["price"].to_numpy(float)
    s_obs = mkt["share"].to_numpy(float)
    x_tilde = mkt["x_tilde"].to_numpy(float)
    prod_ids = mkt["product"].astype(int).tolist()
    J = len(prod_ids)
    if (alpha is None) or (sigma is None):
        alpha_hat, sigma_hat = read_beta_sigma()
    else:
        alpha_hat, sigma_hat = alpha, sigma
    v = vi_grid(50)
    delta = berry_contraction(s_obs, x_tilde, sigma_hat, v)
    s_bar, s_ij = shares_from_delta(delta, x_tilde, sigma_hat, v)
    N = s_ij.shape[1]
    M = (s_ij @ s_ij.T) / N
    own = s_bar - np.diag(M)
    dSdp = alpha_hat * M
    for j in range(J):
        dSdp[j,j] = -alpha_hat * own[j]
    O, owners = build_O(prod_ids, mode=O_mode)
    Omega = O * dSdp
    markups = -solve(Omega.T, s_bar)
    mc = p - markups
    out = pd.DataFrame({"product":prod_ids, "owner":owners, "price":p, "share":s_bar, "markup":markups, "mc":mc})
    return out, dSdp, Omega

if __name__ == "__main__":
    os.makedirs("tables", exist_ok=True)
    out, dSdp, Omega = compute_market_costs(data_path="input/data_yoghurt.csv", city=1, period=1, center_mode="market", O_mode="single")
    stem = "city1_period1_single_market"
    export_df_to_latex(out.round(6), f"tables/q10_results_{stem}.tex", caption="Prices, Shares, Markups, And Marginal Costs (City 1, Period 1)", label=f"tab:q10_results_{stem}", index=False, use_booktabs=True)
    print(out.round(6).to_string(index=False))
