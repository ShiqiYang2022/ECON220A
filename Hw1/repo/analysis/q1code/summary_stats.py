import pandas as pd

def save_latex_table(df, out_path,
                     caption="Summary of Yogurt Products in City 1, Period 1",
                     label="tab:yoghurt_summary"):
    df_pretty = df.rename(columns={
        "product_id": "ID",
        "product_name": "Name",
        "price": "Price",
        "market_share": "Share",
        "package_size_g": "Size",
        "calories_per_g": "Kcal/g",
        "sugar_per_g": "Sugar/g",
        "protein_per_g": "Protein/g"
    })
    latex_body = df_pretty.to_latex(
        index=False,
        float_format="%.3f",
        column_format="lccccccc"
    )
    full_tex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"{latex_body}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    with open(out_path, "w") as f:
        f.write(full_tex)

def main():
    df = pd.read_csv("input/data_yoghurt_clean.csv")
    product_names = {
        1: "Yoplait",
        2: "Chobani",
        3: "Dannon",
        4: "Stonyfield Farm",
        5: "Activia"
    }
    sub = df[(df["city"] == 1) & (df["period"] == 1)].copy()
    table = (
        sub[["product", "price", "share", "weight",
             "calories_per_g", "sugar_per_g", "protein_per_g"]]
        .rename(columns={
            "product": "product_id",
            "share": "market_share",
            "weight": "package_size_g"
        })
        .sort_values("product_id")
        .reset_index(drop=True)
    )
    table.insert(1, "product_name", table["product_id"].map(product_names))
    top2_price = table.nlargest(2, "price")[["product_id", "product_name", "price"]]
    top_share = table.nlargest(2, "market_share")[["product_id", "product_name", "market_share"]]
    save_latex_table(table, "output/q1_city1_period1_table.tex")
    print("Top 2 by price:\n", top2_price)
    print("Top 2 by market share:\n", top_share)

if __name__ == "__main__":
    main()
