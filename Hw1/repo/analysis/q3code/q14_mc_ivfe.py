import os, numpy as np, pandas as pd, statsmodels.api as sm

NAMES = {1:"Yoplait",2:"Chobani",3:"Dannon",4:"Stonyfield Farm",5:"Activia"}

def _booktabs(df, caption, label):
    head = "\\begin{table}[H]\n\\centering\n"
    head += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    head += "\\begin{tabular}{lcccc}\n\\toprule\n"
    head += "Product & Price & Share & $1/\\{\\hat\\alpha(1-s)\\}$ & Marginal cost $c_{jct}$ \\\\\n\\midrule\n"
    body = "\n".join(f"{r.Product} & {r.Price:.4f} & {r.Share:.4f} & {r.Markup:.4f} & {r.MC:.4f} \\\\" for _, r in df.iterrows())
    tail = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return head+body+tail

def _prepare(df, z_distance, z_diesel):
    for col in ["price","share","weight","calories","sugar","protein",z_distance,z_diesel]:
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
    return beta.flatten(), np.sqrt(np.diag(cov)).flatten()

def estimate_alpha(df):
    FE = _FE(df)
    X = pd.concat([df[["price"]], FE], axis=1)
    Z = pd.concat([df[["iv_cost"]], FE], axis=1)
    X = X.replace([np.inf,-np.inf], np.nan).apply(pd.to_numeric, errors="coerce").astype(float)
    Z = Z.replace([np.inf,-np.inf], np.nan).apply(pd.to_numeric, errors="coerce").astype(float)
    y = pd.to_numeric(df["logit_dep"].replace([np.inf,-np.inf], np.nan), errors="coerce").astype(float)
    ok = y.notna() & X.notna().all(axis=1) & Z.notna().all(axis=1)
    y,X,Z = y.loc[ok], X.loc[ok], Z.loc[ok]
    b,_ = _iv2sls_hc1(y,X,Z)
    alpha = -float(b[list(X.columns).index("price")])
    return alpha

def compute_mc(df, alpha, city, period):
    mkt = df[(df["city"]==city)&(df["period"]==period)].copy().sort_values("product")
    mkt["Product"] = mkt["product"].map(NAMES).fillna(mkt["product"].astype(str))
    mkt["Price"]   = pd.to_numeric(mkt["price"], errors="coerce").astype(float)
    mkt["Share"]   = pd.to_numeric(mkt["share"], errors="coerce").astype(float)
    eps = 1e-12
    mkt["Markup"]  = 1.0/(alpha*np.maximum(eps,1.0-mkt["Share"]))
    mkt["MC"]      = mkt["Price"] - mkt["Markup"]
    return mkt[["Product","Price","Share","Markup","MC"]]

def main(in_csv="input/data_yoghurt_clean.csv",
         out_tex="output/q14_mc_city1p1_ivfe.tex",
         z_distance="distance", z_diesel="diesel",
         city=1, period=1):
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    df = pd.read_csv(in_csv)
    df = _prepare(df, z_distance, z_diesel)
    alpha = estimate_alpha(df)
    tbl = compute_mc(df, alpha, city, period)
    with open(out_tex,"w") as f:
        f.write(_booktabs(tbl, f"Recovered marginal costs (City {city}, Period {period}, IV-FE $\\hat\\alpha$).", f"tab:q14_mc_c{city}t{period}"))
    print(f"alpha_hat (IV-FE) = {alpha:.6f}")
    print("Saved:", out_tex)

if __name__ == "__main__":
    main()
