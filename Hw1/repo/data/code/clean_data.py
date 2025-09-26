# clean_data.py
import pandas as pd

def main():
    in_path = "input/data_yoghurt.csv"
    out_path = "output/data_yoghurt_clean.csv"

    df = pd.read_csv(in_path)
    df["calories_per_g"] = df["calories"] / df["weight"]
    df["sugar_per_g"]     = df["sugar"]     / df["weight"]
    df["protein_per_g"]   = df["protein"]   / df["weight"]

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

main()
