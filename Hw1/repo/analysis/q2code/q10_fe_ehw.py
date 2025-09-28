import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

NAMES = {1:"Yoplait",2:"Chobani",3:"Dannon",4:"Stonyfield Farm",5:"Activia"}

def _stars(p):
    return "" if pd.isna(p) else ("***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "")))

def _fmt(s, d):
    return s.apply(lambda x: f"{x:.{d}f}")

def _booktabs(df, caption, label, colfmt=None):
    if colfmt is None:
        colfmt = "l" + "c"*(df.shape[1]-1)
    body = df.to_latex(index=False, escape=False, column_format=colfmt, bold_rows=False)
    return "\\begin{table}[H]\n\\centering\n" + body + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"

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

def estimate_alpha_FE(df):
    d_prod = pd.get_dummies(df["product"], prefix="prod", drop_first=True)
    d_city = pd.get_dummies(df["city"], prefix="city", drop_first=True)
    d_per  = pd.get_dummies(df["period"], prefix="per", drop_first=True)
    X = pd.concat([df[["price"]], d_prod, d_city, d_per], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = df["logit_dep"]
    X = X.replace([np.inf,-np.inf], np.nan).apply(pd.to_numeric, errors="coerce").astype(float)
    y = pd.to_numeric(y.replace([np.inf,-np.inf], np.nan), errors="coerce").astype(float)
    ok = y.notna() & X.notna().all(axis=1)
    y = y.loc[ok]
    X = X.loc[ok]
    res = sm.OLS(y, X).fit(cov_type="HC1")
    return res

def regression_table(res):
    out = pd.DataFrame({
        "Variable": ["Price"],
        "Coef.":    [res.params["price"]],
        "Std. Err. (HC1)": [res.bse["price"]],
        "t":        [res.tvalues["price"]],
        "p-value":  [res.pvalues["price"]],
    })
    out["Sig."] = out["p-value"].apply(_stars)
    out["Coef."] = _fmt(out["Coef."], 3) + out["Sig."]
    out["Std. Err. (HC1)"] = _fmt(out["Std. Err. (HC1)"], 3)
    out["t"] = _fmt(out["t"], 3)
    out["p-value"] = _fmt(out["p-value"], 3)
    out = out[["Variable","Coef.","Std. Err. (HC1)","t","p-value"]]
    return out

def elasticities_matrix(df, alpha, city=1, period=1):
    mkt = df[(df["city"]==city)&(df["period"]==period)].copy().sort_values("product")
    mkt["product_name"] = mkt["product"].map(NAMES).fillna(mkt["product"].astype(str))
    prices = mkt["price"].to_numpy(float)
    shares = mkt["share"].to_numpy(float)
    names  = mkt["product_name"].tolist()
    J = len(names)
    rows = ["Outside Option"] + names
    cols = names
    E = np.zeros((J+1, J))
    for m in range(J):
        E[0, m] = alpha * prices[m] * shares[m]
        for j in range(J):
            if j == m:
                E[j+1, m] = -alpha * prices[m] * (1.0 - shares[m])
            else:
                E[j+1, m] =  alpha * prices[m] * shares[m]
    return E, rows, cols

def diversion_matrix(df, city=1, period=1):
    mkt = df[(df["city"]==city)&(df["period"]==period)].copy().sort_values("product")
    mkt["product_name"] = mkt["product"].map(NAMES).fillna(mkt["product"].astype(str))
    shares = mkt["share"].to_numpy(float)
    names  = mkt["product_name"].tolist()
    s0 = float(1.0 - shares.sum())
    J = len(names)
    cols = names + ["Outside option"]
    rows = names
    D = np.zeros((J, J+1))
    for j in range(J):
        denom = 1.0 - shares[j]
        for m in range(J):
            D[j, m] = 0.0 if j==m else shares[m]/denom
        D[j, J] = s0/denom
    return D, rows, cols

def main(in_csv="input/data_yoghurt_clean.csv",
         out_reg="output/q10_fe_ols_price.tex",
         out_elast="output/q10_elasticities_city1p1_fe.tex",
         out_div="output/q10_diversion_city1p1_fe.tex",
         city=1, period=1):
    os.makedirs(os.path.dirname(out_reg), exist_ok=True)
    os.makedirs(os.path.dirname(out_elast), exist_ok=True)
    os.makedirs(os.path.dirname(out_div), exist_ok=True)
    df = pd.read_csv(in_csv)
    for col in ["price","share","weight","calories","sugar","protein"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["calories_per_g"] = df["calories"]/df["weight"]
    df["sugar_per_g"]    = df["sugar"]/df["weight"]
    df["protein_per_g"]  = df["protein"]/df["weight"]
    df["market"] = list(zip(df["city"], df["period"]))
    df = df.merge(df.groupby("market")["share"].sum().rename("inside_share_sum"), on="market", how="left")
    df["s0"] = 1.0 - df["inside_share_sum"]
    eps = 1e-12
    df["logit_dep"] = np.log(np.clip(df["share"], eps, None)) - np.log(np.clip(df["s0"], eps, None))
    res = estimate_alpha_FE(df)
    tbl = regression_table(res)
    with open(out_reg, "w") as f:
        f.write(_booktabs(tbl, "FE-OLS of the inverted logit demand (HC1; price coefficient only).", "tab:q10_fe_ols_price"))
    alpha_hat = -float(res.params["price"])
    E, rnames, cnames = elasticities_matrix(df, alpha_hat, city, period)
    with open(out_elast, "w") as f:
        f.write(_booktabs_matrix(E, rnames, cnames, f"Own- and cross-price elasticities (City {city}, Period {period}, FE-OLS $\\hat\\alpha$).", f"tab:q10_elast_c{city}t{period}_fe"))
    D, r2, c2 = diversion_matrix(df, city, period)
    with open(out_div, "w") as f:
        f.write(_booktabs_matrix(D, r2, c2, f"Diversion ratios (City {city}, Period {period}, logit).", f"tab:q10_div_c{city}t{period}_fe"))
    print(f"alpha_hat (FE) = {alpha_hat:.6f}")
    print(f"Saved: {out_reg}, {out_elast}, {out_div}")

if __name__ == "__main__":
    main()
