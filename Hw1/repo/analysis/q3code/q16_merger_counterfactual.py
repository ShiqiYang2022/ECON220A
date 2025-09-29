# q16_merger_counterfactual.py
import os, numpy as np, pandas as pd, statsmodels.api as sm

NAMES = {1:"Yoplait",2:"Chobani",3:"Dannon",4:"Stonyfield Farm",5:"Activia"}

def _booktabs(df, caption, label):
    body = df.to_latex(index=False, escape=False, column_format="lcccc", bold_rows=False)
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
    u = (y - X @ beta).reshape(-1)
    S = (Z * (u**2)[:,None]).T @ Z
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
    return eg/(np.exp(-mx) + eg.sum())

def fixed_point_prices_owners(p0, c, alpha, delta_obs, owners, tol=1e-12, itmax=10000):
    p = p0.copy()
    for _ in range(itmax):
        delta = delta_obs - alpha*p
        s = shares_from_delta(delta)
        M = np.diag(s) - np.outer(s,s)
        Delta = alpha*(owners*M)
        mu = np.linalg.solve(Delta, s)
        p_new = c + mu
        if np.max(np.abs(p_new - p)) < tol:
            return p_new
        p = p_new
    return p

def owners_matrix(products, merged_groups=None):
    J = len(products)
    Omega = np.eye(J)
    if merged_groups:
        idx = {p:i for i,p in enumerate(products)}
        for grp in merged_groups:
            ids = [idx[p] for p in grp if p in idx]
            for i in ids:
                for j in ids:
                    Omega[i,j] = 1.0
    return Omega

def main(in_csv="input/data_yoghurt_clean.csv",
         out_tex="output/q16_merger_prices_city1p1.tex",
         z_distance="distance", z_diesel="diesel",
         city=1, period=1,
         merge_group=(2,3)):
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    df = pd.read_csv(in_csv)
    df = _prepare(df, z_distance, z_diesel)
    alpha = estimate_alpha(df)
    d = recover_delta_obs(df, alpha)
    mkt = d[(d["city"]==city)&(d["period"]==period)].copy().sort_values("product")
    p_obs = mkt["price"].to_numpy(float)
    s_obs = mkt["share"].to_numpy(float)
    delta_obs = mkt["delta_obs"].to_numpy(float)
    c = p_obs - 1.0/(alpha*(1.0 - s_obs))
    prods = mkt["product"].astype(int).tolist()
    names = [NAMES.get(p,str(p)) for p in prods]
    Omega_pre = owners_matrix(prods, merged_groups=None)
    Omega_post = owners_matrix(prods, merged_groups=[merge_group])
    p_pre = fixed_point_prices_owners(p_obs.copy(), c, alpha, delta_obs, Omega_pre)
    p_post = fixed_point_prices_owners(p_obs.copy(), c, alpha, delta_obs, Omega_post)
    out = pd.DataFrame({"Product":names,
                        "Old price":p_pre,
                        "New price (merge)":p_post,
                        "Diff":p_post-p_pre})
    with open(out_tex,"w") as f:
        f.write(_booktabs(out, f"Counterfactual merger (Chobani+Dannon), City {city}, Period {period}.", f"tab:q16_merge_c{city}t{period}"))
    print("Saved:", out_tex)
    print("alpha_hat (IV-FE) =", alpha)

if __name__ == "__main__":
    main()
