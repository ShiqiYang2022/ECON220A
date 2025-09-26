import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

def _stars(p):
    return "" if pd.isna(p) else ("***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "")))

def _format_series(s, digs=4):
    return s.apply(lambda x: f"{x:.{digs}f}")

def main(in_path="input/data_yoghurt_clean.csv", out_tex="output/q3_ols_ehw.tex"):
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    df = pd.read_csv(in_path)
    df["market"] = list(zip(df["city"], df["period"]))
    df = df.merge(df.groupby("market")["share"].sum().rename("inside_share_sum"), on="market", how="left")
    df["s0"] = 1.0 - df["inside_share_sum"]
    eps = 1e-12
    df["logit_dep"] = np.log(np.clip(df["share"], eps, None)) - np.log(np.clip(df["s0"], eps, None))
    Xcols = ["price", "weight", "calories_per_g", "sugar_per_g", "protein_per_g"]
    y = df["logit_dep"]
    X = df[Xcols]
    res = sm.OLS(y, X).fit(cov_type="HC1")
    out = pd.DataFrame({
        "Variable": Xcols,
        "Coef.": res.params.values,
        "Std. Err. (HC1)": res.bse.values,
        "t": res.tvalues.values,
        "p-value": res.pvalues.values
    })
    out["Sig."] = out["p-value"].apply(_stars)
    out["Coef."] = _format_series(out["Coef."], 3) + out["Sig."]
    out["Std. Err. (HC1)"] = _format_series(out["Std. Err. (HC1)"], 3)
    out["t"] = _format_series(out["t"], 3)
    out["p-value"] = _format_series(out["p-value"], 3)
    out = out[["Variable", "Coef.", "Std. Err. (HC1)", "t", "p-value"]]
    latex = out.to_latex(index=False, escape=False, column_format="lcccc",
                         caption="OLS estimation of the inverted logit demand with EHW robust standard errors.",
                         label="tab:q3_ols_ehw", bold_rows=False, longtable=False,
                         multicolumn=False, multicolumn_format="c")
    latex = latex.replace("\\begin{tabular}", "\\begin{tabular}").replace("\\toprule", "\\toprule").replace("\\midrule", "\\midrule").replace("\\bottomrule", "\\bottomrule")
    with open(out_tex, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX table to: {out_tex}")

if __name__ == "__main__":
    main()
