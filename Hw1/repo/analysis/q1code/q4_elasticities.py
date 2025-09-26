import pandas as pd, numpy as np, statsmodels.api as sm

def compute_elasticities_city1p1(in_csv="input/data_yoghurt_clean.csv",
                                 out_tex="output/q4_elasticities_city1p1.tex"):
    df = pd.read_csv(in_csv)

    for col in ["price", "share", "weight", "calories", "sugar", "protein"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["calories_per_g"] = df["calories"] / df["weight"]
    df["sugar_per_g"]     = df["sugar"]     / df["weight"]
    df["protein_per_g"]   = df["protein"]   / df["weight"]

    df["market"] = list(zip(df["city"], df["period"]))
    inside = df.groupby("market")["share"].sum().rename("inside_share_sum")
    df = df.merge(inside, on="market", how="left")
    df["s0"] = 1.0 - df["inside_share_sum"]
    eps = 1e-12
    df["logit_dep"] = np.log(np.clip(df["share"], eps, None)) - np.log(np.clip(df["s0"], eps, None))
    X = df[["price", "weight", "calories_per_g", "sugar_per_g", "protein_per_g"]]
    y = df["logit_dep"]
    res = sm.OLS(y, X).fit(cov_type="HC1")
    alpha_hat = -float(res.params["price"])

    prod_names = {1:"Yoplait",2:"Chobani",3:"Dannon",4:"Stonyfield Farm",5:"Activia"}
    mkt = df[(df["city"]==1)&(df["period"]==1)].copy().sort_values("product")
    mkt["product_name"] = mkt["product"].map(prod_names)
    prices = mkt["price"].to_numpy(float)
    shares = mkt["share"].to_numpy(float)
    prods  = mkt["product"].tolist()
    names  = mkt["product_name"].tolist()

    print("\n[DEBUG] alpha_hat =", alpha_hat)
    print("[DEBUG] price vector =", prices)
    print("[DEBUG] share vector =", shares)
    if len(prices)>0 and len(shares)>0:
        print("[DEBUG] alpha * p1 * s1 =", alpha_hat * prices[0] * shares[0])

    J = len(prods)
    rows = ["Outside Option"] + names
    cols = [f"{nm}" for nm in names]
    E = np.zeros((J+1, J), dtype=float)

    for m in range(J):
        E[0, m] = alpha_hat * prices[m] * shares[m]                 # outside row
        for j in range(J):
            if j == m:
                E[j+1, m] = -alpha_hat * prices[m] * (1 - shares[m])  # own
            else:
                E[j+1, m] =  alpha_hat * prices[m] * shares[m]         # cross

    elastic_df = pd.DataFrame(E, index=rows, columns=cols)

    def to_booktabs(df, caption, label):
        header = "\\begin{table}[H]\n\\centering\n"
        header += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        header += "\\begin{tabular}{l" + "r"*df.shape[1] + "}\n\\toprule\n"
        header += " & " + " & ".join(df.columns) + " \\\\\n\\midrule\n"
        body = "\n".join([f"{idx} & " + " & ".join([f'{v:.4f}' for v in row]) + " \\\\"
                          for idx, row in zip(df.index, df.to_numpy())])
        footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        return header + body + footer

    latex = to_booktabs(elastic_df,
                        "Own- and cross-price elasticities in City 1, Period 1 (logit).",
                        "tab:q4_elasticity_city1p1")
    with open(out_tex, "w") as f:
        f.write(latex)

    return elastic_df

if __name__ == "__main__":
    _ = compute_elasticities_city1p1()
