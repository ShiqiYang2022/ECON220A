# q11_marginal_costs_by_market_FE.py
import os, argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm

NAMES = {1:"Yoplait",2:"Chobani",3:"Dannon",4:"Stonyfield Farm",5:"Activia"}

def estimate_alpha_FE(df):
    g = df.copy()
    g["market"] = list(zip(g["city"], g["period"]))
    inside = g.groupby("market")["share"].sum().rename("inside_share_sum")
    g = g.merge(inside, on="market", how="left")
    g["s0"] = 1.0 - g["inside_share_sum"]
    eps = 1e-12
    g["logit_dep"] = np.log(np.clip(g["share"], eps, None)) - np.log(np.clip(g["s0"], eps, None))
    d_prod = pd.get_dummies(g["product"], prefix="prod", drop_first=True)
    d_city = pd.get_dummies(g["city"], prefix="city", drop_first=True)
    d_per  = pd.get_dummies(g["period"], prefix="per", drop_first=True)
    X = pd.concat([g[["price"]], d_prod, d_city, d_per], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = g["logit_dep"]
    X = X.replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce").astype(float)
    y = pd.to_numeric(y.replace([np.inf, -np.inf], np.nan), errors="coerce").astype(float)
    ok = y.notna() & X.notna().all(axis=1)
    y = y.loc[ok]
    X = X.loc[ok]
    res = sm.OLS(y, X).fit(cov_type="HC1")
    return -float(res.params["price"])

def to_booktabs(df, caption, label):
    head = "\\begin{table}[H]\n\\centering\n"
    head += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    head += "\\begin{tabular}{lcccc}\n\\toprule\n"
    head += "Product & Price & Share & $1/\\{\\hat\\alpha(1-s)\\}$ & Marginal cost $c_{jct}$ \\\\\n\\midrule\n"
    body = "\n".join(
        f"{r.Product} & {r.Price:.4f} & {r.Share:.4f} & {r.Markup:.4f} & {r.MC:.4f} \\\\"
        for _, r in df.iterrows()
    )
    tail = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return head + body + tail

def compute_mc(df, alpha, city, period):
    mkt = df[(df["city"]==city) & (df["period"]==period)].copy().sort_values("product")
    mkt["Product"] = mkt["product"].map(NAMES).fillna(mkt["product"].astype(str))
    mkt["Price"]   = pd.to_numeric(mkt["price"], errors="coerce").astype(float)
    mkt["Share"]   = pd.to_numeric(mkt["share"], errors="coerce").astype(float)
    eps = 1e-12
    mkt["Markup"]  = 1.0/(alpha*(np.maximum(eps, 1.0 - mkt["Share"])))
    mkt["MC"]      = mkt["Price"] - mkt["Markup"]
    return mkt[["city","period","product","Product","Price","Share","Markup","MC"]]

def main(in_path="input/data_yoghurt_clean.csv",
         out_tex="output/q11_mc_city1p1_fe.tex",
         city=1, period=1, alpha=None, export_all=False):
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    df = pd.read_csv(in_path)
    for col in ["price","share","weight","calories","sugar","protein"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["calories_per_g"] = df["calories"]/df["weight"]
    df["sugar_per_g"]    = df["sugar"]/df["weight"]
    df["protein_per_g"]  = df["protein"]/df["weight"]
    alpha_hat = estimate_alpha_FE(df) if alpha is None else float(alpha)
    print(f"alpha_hat used = {alpha_hat:.6f}")
    if export_all:
        panel = []
        for (c,t), _ in df.groupby(["city","period"]):
            panel.append(compute_mc(df, alpha_hat, c, t))
    mc_tbl = compute_mc(df, alpha_hat, city, period)
    with open(out_tex, "w") as f:
        f.write(to_booktabs(mc_tbl[["Product","Price","Share","Markup","MC"]],
                            f"Recovered marginal costs (City {city}, Period {period}, FE-OLS $\\hat\\alpha$).",
                            f"tab:q11_mc_c{city}t{period}_fe"))
    print("Saved LaTeX to    :", out_tex)

if __name__ == "__main__":
    main()