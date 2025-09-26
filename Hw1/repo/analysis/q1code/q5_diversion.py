import os
import numpy as np
import pandas as pd

def compute_diversion_ratios(
    in_csv="input/data_yoghurt_clean.csv",
    out_tex="output/q5_diversion_city1p1.tex",
):
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)

    # Load data
    df = pd.read_csv(in_csv)

    # Subset to city 1, period 1
    mkt = df[(df["city"] == 1) & (df["period"] == 1)].copy().sort_values("product")

    # Map product IDs to names
    prod_names = {
        1: "Yoplait",
        2: "Chobani",
        3: "Dannon",
        4: "Stonyfield Farm",
        5: "Activia",
    }
    mkt["product_name"] = mkt["product"].map(prod_names)

    shares = mkt["share"].to_numpy(float)
    names = mkt["product_name"].tolist()
    s0 = 1.0 - shares.sum()

    J = len(shares)
    cols = names + ["Outside option"]
    rows = names
    D = np.zeros((J, J + 1))

    for j in range(J):
        denom = 1 - shares[j]
        for m in range(J):
            D[j, m] = 0.0 if j == m else shares[m] / denom
        D[j, J] = s0 / denom

    div_df = pd.DataFrame(D, index=rows, columns=cols).round(4)


    # Export LaTeX (booktabs + [H])
    def to_booktabs(df, caption, label):
        header = "\\begin{table}[H]\\centering\n"
        header += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        header += "\\begin{tabular}{l" + "r" * df.shape[1] + "}\n\\toprule\n"
        header += " & " + " & ".join(df.columns) + " \\\\\n\\midrule\n"
        body = "\n".join(
            [
                f"{idx} & " + " & ".join([f"{v:.4f}" for v in row]) + " \\\\"
                for idx, row in zip(df.index, df.values)
            ]
        )
        footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        return header + body + footer

    latex = to_booktabs(
        div_df,
        "Diversion ratios in City 1, Period 1 (logit).",
        "tab:q5_diversion",
    )
    with open(out_tex, "w") as f:
        f.write(latex)

    print(f"Saved LaTeX table to: {out_tex}")
    return div_df


if __name__ == "__main__":
    df = compute_diversion_ratios()
    print(df)
