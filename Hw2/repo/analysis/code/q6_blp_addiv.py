import os, numpy as np, pandas as pd, math
from numpy.linalg import inv
from scipy.stats import norm
import matplotlib.pyplot as plt

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
    return sj

def berry_contraction(s_obs, x_tilde, sigma, v, tol=1e-12, maxit=50000):
    s0 = 1.0 - float(s_obs.sum())
    delta = np.log(np.clip(s_obs, 1e-300, 1.0)) - math.log(max(s0, 1e-300))
    for _ in range(maxit):
        s_pred = shares_from_delta(delta, x_tilde, sigma, v)
        new_delta = delta + (np.log(s_obs) - np.log(np.clip(s_pred,1e-300,1.0)))
        if np.max(np.abs(new_delta - delta)) < tol:
            return new_delta
        delta = new_delta
    return delta

def build_designs_q6(df):
    prod_d = pd.get_dummies(df["product"], prefix="prod", drop_first=True)
    city_d = pd.get_dummies(df["city"],    prefix="city", drop_first=True)
    time_d = pd.get_dummies(df["period"],  prefix="per",  drop_first=True)
    cost_shift = (df["distance"] * df["diesel"]).to_frame("cost_shift")
    X = pd.concat([df[["price"]], prod_d, city_d, time_d], axis=1).astype(float).values
    Z = pd.concat([cost_shift, df[["z_ct"]], prod_d, city_d, time_d], axis=1).astype(float).values
    W = inv(Z.T @ Z)
    return X, Z, W

def prepare_balanced_data_with_z(data_path="input/data_yoghurt.csv"):
    df = pd.read_csv(data_path)
    df["market_id"] = list(zip(df["city"], df["period"]))
    df["sugar_per_g"] = df["sugar"]/df["weight"]
    mask_ref = (df["city"]==1) & (df["period"]==1)
    xbar = df.loc[mask_ref, "sugar_per_g"].mean()
    df["x_tilde"] = df["sugar_per_g"] - xbar
    z_market = df.groupby("market_id")["sugar_per_g"].sum().rename("z_ct")
    df = df.merge(z_market, on="market_id", how="left")
    balanced = (df.groupby("market_id")["product"].nunique()==5)
    ids = balanced[balanced].index.tolist()
    dfb = df[df["market_id"].isin(ids)].copy().reset_index(drop=True)
    gb = dfb.groupby("market_id", sort=True)
    markets = {k: np.array(v, dtype=int) for k,v in gb.indices.items()}
    return dfb, markets

def invert_all_markets(df, markets, sigma, v):
    delta = np.zeros(df.shape[0])
    for mkt, rows in markets.items():
        rows_sorted = rows[np.argsort(df.loc[rows,"product"].values)]
        s_obs = df.loc[rows_sorted,"share"].values
        x_tilde = df.loc[rows_sorted,"x_tilde"].values
        delta[rows_sorted] = berry_contraction(s_obs, x_tilde, sigma, v)
    return delta

def blp_G_q6(df, markets, sigma, v):
    X, Z, W = build_designs_q6(df)
    delta = invert_all_markets(df, markets, sigma, v)
    A = inv(X.T @ Z @ W @ Z.T @ X)
    beta = A @ (X.T @ Z @ W @ (Z.T @ delta))
    xi = delta - X @ beta
    n = X.shape[0]
    m = (Z.T @ xi)/n
    G = 1_000_000.0 * float(m.T @ W @ m)
    return G, float(beta[0])

def main(data_path="input/data_yoghurt.csv", use_first_k_markets=100, outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    dfb, markets = prepare_balanced_data_with_z(data_path)
    keys = list(markets.keys())[:use_first_k_markets] if use_first_k_markets else list(markets.keys())
    df_small = dfb[dfb["market_id"].isin(keys)].copy().reset_index(drop=True)
    gb_small = df_small.groupby("market_id", sort=True)
    markets_small = {k: np.array(v, dtype=int) for k,v in gb_small.indices.items()}
    v = vi_grid(50)

    res_q6 = []
    for s in [0, 10]:
        G, alpha = blp_G_q6(df_small, markets_small, s, v)
        res_q6.append({"sigma": s, "G": G, "alpha": alpha})
    pd.DataFrame(res_q6).to_csv(f"{outdir}/q6_blp_costshifter_plus_z_G.csv", index=False)

    sigma_vals = np.arange(-200, 201, 10)
    G_vals = [blp_G_q6(df_small, markets_small, s, v)[0] for s in sigma_vals]
    dG_vals = np.gradient(G_vals, sigma_vals)
    pd.DataFrame({"sigma": sigma_vals, "G": G_vals, "dG": dG_vals}).to_csv(f"{outdir}/q7_Gsigma.csv", index=False)

    plt.figure(figsize=(9,5))
    plt.plot(sigma_vals, G_vals, lw=2)
    plt.axhline(0, lw=0.6)
    plt.xlabel("sigma")
    plt.ylabel("G(sigma)")
    plt.title("G(sigma) across sigma")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/q7_Gsigma.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9,5))
    plt.plot(sigma_vals, dG_vals, lw=2)
    plt.axhline(0, lw=0.6)
    plt.xlabel("sigma")
    plt.ylabel("dG/dsigma")
    plt.title("dG/dsigma across sigma")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/q7_dG.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
