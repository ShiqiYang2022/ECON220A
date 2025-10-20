import numpy as np, pandas as pd, math
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
    return sj

def berry_contraction(s_obs, x_tilde, sigma, v_draws, tol=1e-14, maxit=20000):
    s0 = 1.0 - float(s_obs.sum())
    delta = np.log(np.clip(s_obs, 1e-300, 1.0)) - math.log(max(s0, 1e-300))
    for _ in range(maxit):
        s_pred = shares_from_delta(delta, x_tilde, sigma, v_draws)
        new_delta = delta + (np.log(s_obs) - np.log(np.clip(s_pred, 1e-300, 1.0)))
        if np.max(np.abs(new_delta - delta)) < tol:
            return new_delta
        delta = new_delta
    return delta

df = pd.read_csv("input/data_yoghurt.csv")
df["market_id"] = list(zip(df["city"], df["period"]))
df["sugar_per_g"] = df["sugar"]/df["weight"]
mask = (df["city"]==1) & (df["period"]==1)
xbar = df.loc[mask, "sugar_per_g"].mean()
df["x_tilde"] = df["sugar_per_g"] - xbar
rows = np.where(mask)[0]
rows = rows[np.argsort(df.loc[rows, "product"].values)]
s_obs = df.loc[rows, "share"].values
x_tilde = df.loc[rows, "x_tilde"].values
s0 = 1.0 - float(s_obs.sum())
v = vi_grid(50)
delta_hat = berry_contraction(s_obs, x_tilde, 0.0, v, tol=1e-14, maxit=50000)
delta_mnl = np.log(s_obs) - math.log(max(s0, 1e-300))
diff = delta_hat - delta_mnl

out = pd.DataFrame({
    "city": 1,
    "period": 1,
    "product": df.loc[rows, "product"].values.astype(int),
    "delta_hat_sigma0": delta_hat,
    "delta_log_s_minus_log_s0": delta_mnl,
    "difference": diff
})
out.to_csv("output/q2_delta_city1_period1_sigma0.csv", index=False)
