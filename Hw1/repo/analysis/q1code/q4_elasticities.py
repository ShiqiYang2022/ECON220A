# q4_elasticities.py
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

PRODUCT_NAMES = {1: "Yoplait", 2: "Chobani", 3: "Dannon", 4: "Stonyfield Farm", 5: "Activia"}

def _booktabs(df, caption, label):
    header = "\\begin{table}[H]\n\\centering\n"
    header += "\\caption{" + caption + "}\n\\label{" + label + "}\n"
    header += "\\begin{tabular}{l" + "r"*df.shape[1] + "}\n\\toprule\n"
    header += " & " + " & ".join(df.columns) + " \\\\\n\\midrule\n"
    body = "\n".join([idx + " & " + " & ".join([f"{v:.4f}" for v in row]) + " \\\\" for idx, row in zip(df.index, df.values)])
    footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + body + footer

def main(in_path="input/data_yoghurt_clean.csv",
         out_tex="output/q4_elasticities_city1p1.tex"):
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    df = pd.read_csv(in_path)
    df["market"] = list(zip(df["city"], df["period"]))
    inside = df.groupby("market")["share"].sum().rename("inside_share_sum")
    df = df.merge(inside, on="market", how="left")
    df["s0"] = 1.0 - df["inside_share_sum"]
    eps = 1e-12
    df["logit_dep"] = np.log(np.clip(df["share"], eps, None)) - np.log(np.clip(df["s0"], eps, None))
    X = df[["price", "weight", "calories_per_g", "sugar_per_g", "protein_per_g"]]
    y = df["logit_dep"]
    res = sm.OLS(y, X).fit(cov_type="HC1")
    alpha_hat = -res.params["price"]
    mkt = df[(df["city"] == 1) & (df["period"] == 1)].copy().sort_values("product")
    J = mkt.shape[0]
    prices = mkt["price"].to_numpy()
    shares = mkt["share"].to_numpy()
    row_names = ["Outside Option"] + [PRODUCT_NAMES.get(int(pid), f"Product {int(pid)}") for pid in mkt["product"]]
    col_names = [PRODUCT_NAMES.get(int(pid), f"Product {int(pid)}") for pid in mkt["product"]]
    E = np.zeros((J+1, J))
    for c in range(J):
        E[0, c] = alpha_hat * prices[c] * shares[c]
    for j in range(J):
        for m in range(J):
            if j == m:
                E[j+1, m] = -alpha_hat * prices[m] * (1 - shares[m])
            else:
                E[j+1, m] = alpha_hat * prices[m] * shares[m]
    elastic_df = pd.DataFrame(E, index=row_names, columns=col_names)
    latex = _booktabs(elastic_df,
                      "Own- and cross-price elasticities in City 1, Period 1 (logit).",
                      "tab:q4_elasticity_city1p1")
    with open(out_tex, "w") as f:
        f.write(latex)
    print(f"alpha_hat={alpha_hat:.6f}")
    print(f"Saved: {out_tex}")

if __name__ == "__main__":
    main()
