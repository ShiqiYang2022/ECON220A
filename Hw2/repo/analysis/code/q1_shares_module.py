# Recompute and save Q1 outputs with the corrected function
import numpy as np, pandas as pd, math
from scipy.stats import norm

def individual_shares_fixed(delta_vec, x_tilde_vec, sigma, v_draws):
    util = delta_vec.reshape(-1,1) + sigma * x_tilde_vec.reshape(-1,1) * v_draws.reshape(1,-1)
    a = np.max(util, axis=0, keepdims=True)
    expu = np.exp(util - a)
    denom = np.exp(-a) + np.sum(expu, axis=0, keepdims=True)
    sij = expu / denom
    sj = np.mean(sij, axis=1)
    return sij, sj

def market_shares_fixed(delta_vec, x_tilde_vec, sigma, v_draws):
    _, sj = individual_shares_fixed(delta_vec, x_tilde_vec, sigma, v_draws)
    return sj

# Prepare inputs
df = pd.read_csv("./input/data_yoghurt.csv")
df["market_id"] = list(zip(df["city"], df["period"]))
df["sugar_per_g"] = df["sugar"]/df["weight"]
mask_ref = (df["city"]==1) & (df["period"]==1)
xbar_s = df.loc[mask_ref, "sugar_per_g"].mean()
df["x_tilde"] = df["sugar_per_g"] - xbar_s
rows = np.where(mask_ref)[0]
rows = rows[np.argsort(df.loc[rows, "product"].values)]
s_obs = df.loc[rows, "share"].values
s0 = 1.0 - s_obs.sum()
delta_mnl = np.log(s_obs) - np.log(s0)
x_tilde = df.loc[rows, "x_tilde"].values

u = np.linspace(0.1, 0.9, 50)
v = norm.ppf(u)

sj_sigma0 = market_shares_fixed(delta_mnl, x_tilde, 0.0, v)
sj_sigma2 = market_shares_fixed(delta_mnl, x_tilde, 2.0, v)

pd.DataFrame({"product": df.loc[rows, "product"].values.astype(int),
              "x_tilde": x_tilde,
              "share_obs": s_obs,
              "share_pred_sigma0": sj_sigma0}).to_csv("./output/q1_shares_sigma0.csv", index=False)

pd.DataFrame({"product": df.loc[rows, "product"].values.astype(int),
              "x_tilde": x_tilde,
              "share_pred_sigma2": sj_sigma2}).to_csv("."
              "./output/q1_shares_sigma2.csv", index=False)

