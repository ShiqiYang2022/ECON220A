import os, numpy as np, pandas as pd, statsmodels.api as sm

def _booktabs(df, caption, label):
    body = df.to_latex(index=False, escape=False, column_format="lccc", bold_rows=False)
    return "\\begin{table}[H]\n\\centering\n"+f"\\caption{{{caption}}}\n" + body + f"\\label{{{label}}}\n\\end{{table}}\n"

def _prepare(df, z_distance, z_diesel):
    for col in ["price","share",z_distance,z_diesel]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["market"] = list(zip(df["city"], df["period"]))
    inside = df.groupby("market")["share"].sum().rename("inside_share_sum")
    df = df.merge(inside, on="market", how="left")
    df["s0"] = 1.0 - df["inside_share_sum"]
    eps = 1e-12
    df["logit_dep"] = np.log(np.clip(df["share"], eps, None)) - np.log(np.clip(df["s0"], eps, None))
    df["iv_cost"] = df[z_distance]*df[z_diesel]
    return df

def _FE(df):
    d_prod = pd.get_dummies(df["product"], prefix="prod", drop_first=True)
    d_city = pd.get_dummies(df["city"], prefix="city", drop_first=True)
    d_per  = pd.get_dummies(df["period"], prefix="per", drop_first=True)
    FE = pd.concat([d_prod,d_city,d_per], axis=1)
    FE = sm.add_constant(FE, has_constant="add")
    return FE

def _iv2sls_hc1(y,X,Z):
    y = y.values.reshape(-1,1); X = X.values; Z = Z.values
    ZZ = Z.T @ Z; ZZinv = np.linalg.inv(ZZ); XZ = X.T @ Z; ZY = Z.T @ y
    beta = np.linalg.inv(XZ @ ZZinv @ XZ.T) @ (XZ @ ZZinv @ ZY)
    u = (y - X @ beta).reshape(-1)                      # n,
    S = (Z * (u**2)[:,None]).T @ Z                      # Z' diag(u^2) Z
    A = XZ @ ZZinv @ XZ.T
    B = XZ @ ZZinv @ S @ ZZinv @ XZ.T
    cov = np.linalg.inv(A) @ B @ np.linalg.inv(A)
    n,k = X.shape
    cov = cov * n/(n-k)
    return beta.flatten()

def estimate_alpha(df):
    FE = _FE(df)
    X = pd.concat([df[["price"]], FE], axis=1)
    Z = pd.concat([df[["iv_cost"]], FE], axis=1)
    X = X.replace([np.inf,-np.inf], np.nan).apply(pd.to_numeric, errors="coerce").astype(float)
    Z = Z.replace([np.inf,-np.inf], np.nan).apply(pd.to_numeric, errors="coerce").astype(float)
    y = pd.to_numeric(df["logit_dep"].replace([np.inf,-np.inf], np.nan), errors="coerce").astype(float)
    ok = y.notna() & X.notna().all(axis=1) & Z.notna().all(axis=1)
    y,X,Z = y.loc[ok], X.loc[ok], Z.loc[ok]
    b = _iv2sls_hc1(y,X,Z)
    return -float(b[list(X.columns).index("price")])

def recover_delta_obs(df, alpha):
    d = df.copy()
    d["delta_obs"] = d["logit_dep"] + alpha*d["price"]
    return d

def shares_from_delta(delta):
    mx = np.max(delta)
    eg = np.exp(delta-mx)
    s = eg/(np.exp(-mx) + eg.sum())
    return s

def fixed_point_prices(p0, c, alpha, delta_obs, tol=1e-12, itmax=10000):
    p = p0.copy()
    for _ in range(itmax):
        delta = delta_obs - alpha*p
        s = shares_from_delta(delta)
        K = len(s)
        M = np.diag(s) - np.outer(s,s)
        owners = np.eye(K)                         
        Delta = alpha * (owners * M)               
        mu = np.linalg.solve(Delta, s)
        p_new = c + mu
        if np.max(np.abs(p_new-p)) < tol:
            return p_new
        p = p_new
    return p

def main(in_csv="input/data_yoghurt_clean.csv",
         out_tex="output/q15_recovered_prices_city1p1.tex",
         z_distance="distance", z_diesel="diesel",
         city=1, period=1):
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    df = pd.read_csv(in_csv)
    df = _prepare(df, z_distance, z_diesel)
    alpha = estimate_alpha(df)
    d = recover_delta_obs(df, alpha)
    mkt = d[(d["city"]==city)&(d["period"]==period)].copy().sort_values("product")
    p_obs = mkt["price"].to_numpy(float)
    s_obs = mkt["share"].to_numpy(float)
    c = p_obs - 1/(alpha*(1.0-s_obs))
    delta_obs = (mkt["delta_obs"].to_numpy(float)).reshape(-1)
    p_rec = fixed_point_prices(p_obs.copy(), c, alpha, delta_obs)
    out = pd.DataFrame({"Product":mkt["product"].astype(int).astype(str),
                        "Observed price":p_obs,
                        "Recovered price":p_rec,
                        "Abs. diff":np.abs(p_rec-p_obs)})
    with open(out_tex,"w") as f:
        f.write(_booktabs(out, f"Observed vs. recovered equilibrium prices (City {city}, Period {period}).", f"tab:q15_recover_c{city}t{period}"))
    print(f"alpha_hat (IV-FE) = {alpha:.6f}")
    print("max|diff| =", float(np.max(np.abs(p_rec-p_obs))))
    print("Saved:", out_tex)

if __name__ == "__main__":
    main()
