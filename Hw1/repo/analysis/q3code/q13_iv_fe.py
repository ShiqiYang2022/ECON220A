import os, numpy as np, pandas as pd, statsmodels.api as sm

NAMES = {1:"Yoplait",2:"Chobani",3:"Dannon",4:"Stonyfield Farm",5:"Activia"}

def _booktabs(df, caption, label, colfmt=None):
    if colfmt is None: colfmt = "l" + "c"*(df.shape[1]-1)
    body = df.to_latex(index=False, escape=False, column_format=colfmt, bold_rows=False)
    return "\\begin{table}[H]\n\\centering\n" +f"\\caption{{{caption}}}\n" + body + f"\\label{{{label}}}\n\\end{{table}}\n"

def _booktabs_matrix(M, row_names, col_names, caption, label):
    head = "\\begin{table}[H]\n\\centering\n"
    head += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    head += "\\begin{tabular}{l" + "r"*len(col_names) + "}\n\\toprule\n"
    head += " & " + " & ".join(col_names) + " \\\\\n\\midrule\n"
    lines = []
    for i, r in enumerate(row_names):
        lines.append(r + " & " + " & ".join(f"{v:.4f}" for v in M[i,:]) + " \\\\")
    tail = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return head + "\n".join(lines) + tail

def _ehw_2sls(y, X, Z):
    y = y.values.reshape(-1,1)
    X = X.values
    Z = Z.values
    ZZ = Z.T @ Z
    ZZinv = np.linalg.inv(ZZ)
    XZ = X.T @ Z
    ZY = Z.T @ y
    beta = np.linalg.inv(XZ @ ZZinv @ XZ.T) @ (XZ @ ZZinv @ ZY)
    u = (y - X @ beta).reshape(-1)                      # n,
    S = (Z * (u**2)[:,None]).T @ Z                      # Z' diag(u^2) Z
    A = XZ @ ZZinv @ XZ.T
    B = XZ @ ZZinv @ S @ ZZinv @ XZ.T
    cov = np.linalg.inv(A) @ B @ np.linalg.inv(A)
    n,k = X.shape
    cov = cov * n/(n-k)
    se = np.sqrt(np.diag(cov)).reshape(-1,1)
    t = (beta/se).flatten()
    return beta.flatten(), se.flatten(), t

def _prepare_fe_panels(df):
    d_prod = pd.get_dummies(df["product"], prefix="prod", drop_first=True)
    d_city = pd.get_dummies(df["city"], prefix="city", drop_first=True)
    d_per  = pd.get_dummies(df["period"], prefix="per", drop_first=True)
    FE = pd.concat([d_prod, d_city, d_per], axis=1)
    FE = sm.add_constant(FE, has_constant="add")
    return FE

def estimate_alpha_iv_fe(df, zname="iv_cost"):
    FE = _prepare_fe_panels(df)
    X = pd.concat([df[["price"]], FE], axis=1)
    Z = pd.concat([df[[zname]], FE], axis=1)
    X = X.replace([np.inf,-np.inf], np.nan).apply(pd.to_numeric, errors="coerce").astype(float)
    Z = Z.replace([np.inf,-np.inf], np.nan).apply(pd.to_numeric, errors="coerce").astype(float)
    y = pd.to_numeric(df["logit_dep"].replace([np.inf,-np.inf], np.nan), errors="coerce").astype(float)
    ok = y.notna() & X.notna().all(axis=1) & Z.notna().all(axis=1)
    y,X,Z = y.loc[ok], X.loc[ok], Z.loc[ok]
    b,se,t = _ehw_2sls(y, X, Z)
    names = X.columns.tolist()
    ix = names.index("price")
    return -float(b[ix]), float(se[ix]), float(t[ix])

def elasticities_and_diversion(df, alpha, city=1, period=1):
    mkt = df[(df["city"]==city)&(df["period"]==period)].copy().sort_values("product")
    mkt["product_name"] = mkt["product"].map(NAMES).fillna(mkt["product"].astype(str))
    P = mkt["price"].to_numpy(float); S = mkt["share"].to_numpy(float); names = mkt["product_name"].tolist()
    J = len(names)
    E = np.zeros((J+1,J))
    for m in range(J):
        E[0,m] = alpha*P[m]*S[m]
        for j in range(J):
            E[j+1,m] = -alpha*P[m]*(1.0-S[m]) if j==m else alpha*P[m]*S[m]
    D = np.zeros((J,J+1))
    s0 = float(1.0-S.sum())
    for j in range(J):
        denom = 1.0-S[j]
        for m in range(J):
            D[j,m] = 0.0 if j==m else S[m]/denom
        D[j,J] = s0/denom
    return E, (["Outside Option"]+names), names, D, names, (names+["Outside option"])

def main(in_csv="input/data_yoghurt_clean.csv",
         out_reg="output/q13_iv_fe_price.tex",
         out_elast="output/q13_elasticities_city1p1_ivfe.tex",
         out_div="output/q13_diversion_city1p1_ivfe.tex",
         z_distance="distance", z_diesel="diesel",
         city=1, period=1):
    os.makedirs(os.path.dirname(out_reg), exist_ok=True)
    os.makedirs(os.path.dirname(out_elast), exist_ok=True)
    os.makedirs(os.path.dirname(out_div), exist_ok=True)
    df = pd.read_csv(in_csv)
    for col in ["price","share","weight","calories","sugar","protein",z_distance,z_diesel]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["calories_per_g"] = df["calories"]/df["weight"]
    df["sugar_per_g"]    = df["sugar"]/df["weight"]
    df["protein_per_g"]  = df["protein"]/df["weight"]
    df["market"] = list(zip(df["city"], df["period"]))
    df = df.merge(df.groupby("market")["share"].sum().rename("inside_share_sum"), on="market", how="left")
    df["s0"] = 1.0 - df["inside_share_sum"]
    eps = 1e-12
    df["logit_dep"] = np.log(np.clip(df["share"], eps, None)) - np.log(np.clip(df["s0"], eps, None))
    df["iv_cost"] = df[z_distance]*df[z_diesel]
    a,se,t = estimate_alpha_iv_fe(df, "iv_cost")
    tbl = pd.DataFrame({"Variable":["Price"],"Coef.":[-a], "Std. Err. (HC1)":[se], "t":[(-a)/se], "p-value":[np.nan]})
    with open(out_reg,"w") as f: f.write(_booktabs(tbl, "IV(transport cost) with product/city/period FE (HC1).", "tab:q13_iv_fe_price"))
    E,rn,cn,D,rn2,cn2 = elasticities_and_diversion(df, a, city, period)
    with open(out_elast,"w") as f: f.write(_booktabs_matrix(E, rn, cn, f"Own- and cross-price elasticities (City {city}, Period {period}, IV-FE $\\hat\\alpha$).", f"tab:q13_elast_c{city}t{period}"))
    with open(out_div,"w") as f: f.write(_booktabs_matrix(D, rn2, cn2, f"Diversion ratios (City {city}, Period {period}).", f"tab:q13_div_c{city}t{period}"))
    print(f"alpha_hat (IV-FE) = {a:.6f}  SE(HC1) = {se:.6f}")

if __name__ == "__main__":
    main()
