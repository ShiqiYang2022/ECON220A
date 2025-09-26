# q7_marginal_costs.py
import os
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm

PRODUCT_NAMES = {1:"Yoplait",2:"Chobani",3:"Dannon",4:"Stonyfield Farm",5:"Activia"}

def estimate_alpha(df: pd.DataFrame) -> float:
    df = df.copy()
    df["market"] = list(zip(df["city"], df["period"]))
    inside = df.groupby("market")["share"].sum().rename("inside_share_sum")
    df = df.merge(inside, on="market", how="left")
    df["s0"] = 1.0 - df["inside_share_sum"]
    eps = 1e-12
    df["logit_dep"] = np.log(np.clip(df["share"], eps, None)) - np.log(np.clip(df["s0"], eps, None))
    X = df[["price","weight","calories_per_g","sugar_per_g","protein_per_g"]]
    y = df["logit_dep"]
    res = sm.OLS(y, X).fit(cov_type="HC1")
    return -float(res.params["price"])

def to_booktabs(df: pd.DataFrame, caption: str, label: str) -> str:
    header = "\\begin{table}[H]\n\\centering\n"
    header += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    header += "\\begin{tabular}{lccc}\n\\toprule\n"
    header += "Product & Price & Share & Marginal cost $c_j$ \\\\\n\\midrule\n"
    body = "\n".join(
        f"{row.Product} & {row.Price:.3f} & {row.Share:.3f} & {row['Marginal cost']:.3f} \\\\"
        for _, row in df.iterrows()
    )
    footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + body + footer

def main(in_path="input/data_yoghurt_clean.csv",
         out_tex="output/q7_mc_city1p1.tex",
         alpha_override: float | None = None):
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)

    df = pd.read_csv(in_path)
    for col in ["price","share","weight","calories","sugar","protein"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["calories_per_g"] = df["calories"] / df["weight"]
    df["sugar_per_g"]     = df["sugar"]     / df["weight"]
    df["protein_per_g"]   = df["protein"]   / df["weight"]

    alpha = alpha_override if alpha_override is not None else estimate_alpha(df)

    mkt = df[(df["city"]==1) & (df["period"]==1)].copy().sort_values("product")
    mkt["Product"] = mkt["product"].map(PRODUCT_NAMES)
    mkt["Price"] = mkt["price"].astype(float)
    mkt["Share"] = mkt["share"].astype(float)

    mc = mkt[["Product","Price","Share"]].copy()
    mc["Marginal cost"] = mc["Price"] - 1.0/(alpha*(1.0 - mc["Share"]))

    with open(out_tex, "w") as f:
        f.write(to_booktabs(mc, "Recovered marginal costs (City 1, Period 1).", "tab:q7_mc"))

    print(f"alpha_hat used = {alpha:.6f}")
    print(f"Saved LaTeX to : {out_tex}")

if __name__ == "__main__":
    main()
