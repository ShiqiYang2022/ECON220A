import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

def _stars(p):
    return "" if pd.isna(p) else ("***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "")))

def _fmt(s, d):
    return s.apply(lambda x: f"{x:.{d}f}")

def main(in_path="input/data_yoghurt_clean.csv", out_tex="output/q3_ols_ehw.tex",
         caption="OLS estimation of the inverted logit demand (EHW robust SE).",
         label="tab:q3_ols_ehw"):
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    df = pd.read_csv(in_path)
    df["market"] = list(zip(df["city"], df["period"]))
    df = df.merge(df.groupby("market")["share"].sum().rename("inside_share_sum"), on="market", how="left")
    df["s0"] = 1.0 - df["inside_share_sum"]
    eps = 1e-12
    df["logit_dep"] = np.log(np.clip(df["share"], eps, None)) - np.log(np.clip(df["s0"], eps, None))

    Xcols = ["price", "weight", "calories_per_g", "sugar_per_g", "protein_per_g"]
    pretty = {
        "const": "Intercept",
        "price": "Price",
        "weight": "Package size",
        "calories_per_g": "Calories per g",
        "sugar_per_g": "Added sugar per g",
        "protein_per_g": "Protein per g"
    }
    X = sm.add_constant(df[Xcols], has_constant="add")
    y = df["logit_dep"]
    res = sm.OLS(y, X).fit(cov_type="HC1")

    # Use the exact order returned by statsmodels (includes 'const')
    order = list(res.params.index)

    out = pd.DataFrame({
        "Variable": [pretty.get(c, c) for c in order],
        "Coef.":    res.params[order].values,
        "Std. Err. (HC1)": res.bse[order].values,
        "t":        res.tvalues[order].values,
        "p-value":  res.pvalues[order].values,
    })

    out["Sig."] = out["p-value"].apply(_stars)
    out["Coef."] = _fmt(out["Coef."], 3) + out["Sig."]
    out["Std. Err. (HC1)"] = _fmt(out["Std. Err. (HC1)"], 3)
    out["t"] = _fmt(out["t"], 3)
    out["p-value"] = _fmt(out["p-value"], 3)
    out = out[["Variable", "Coef.", "Std. Err. (HC1)", "t", "p-value"]]

    body = out.to_latex(index=False, escape=False, column_format="lcccc", bold_rows=False)
    table_env = (
        "\\begin{table}[H]\n\\centering\n"
        + body
        + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )
    with open(out_tex, "w") as f:
        f.write(table_env)
    print(f"Saved LaTeX table to: {out_tex}")

if __name__ == "__main__":
    main()
