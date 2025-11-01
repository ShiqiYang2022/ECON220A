import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm
from utils.latex_table import export_df_to_latex

def vi_grid(n: int = 50) -> np.ndarray:
    u = np.linspace(0.1, 0.9, n)
    return norm.ppf(u)

def shares_from_delta(delta_vec: np.ndarray, x_tilde_vec: np.ndarray, sigma: float, v_draws: np.ndarray) -> np.ndarray:
    util = delta_vec.reshape(-1, 1) + sigma * x_tilde_vec.reshape(-1, 1) * v_draws.reshape(1, -1)
    a = np.max(util, axis=0, keepdims=True)
    expu = np.exp(util - a)
    denom = np.exp(-a) + np.sum(expu, axis=0, keepdims=True)
    sij = expu / denom
    return np.mean(sij, axis=1)

def berry_contraction(s_obs: np.ndarray, x_tilde: np.ndarray, sigma: float, v_draws: np.ndarray, tol: float = 1e-14, maxit: int = 20000) -> np.ndarray:
    s0 = 1.0 - float(s_obs.sum())
    delta = np.log(np.clip(s_obs, 1e-300, 1.0)) - math.log(max(s0, 1e-300))
    for _ in range(maxit):
        s_pred = shares_from_delta(delta, x_tilde, sigma, v_draws)
        new_delta = delta + (np.log(s_obs) - np.log(np.clip(s_pred, 1e-300, 1.0)))
        if np.max(np.abs(new_delta - delta)) < tol:
            return new_delta
        delta = new_delta
    return delta

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="input/data_yoghurt.csv")
    p.add_argument("--city", type=int, default=1)
    p.add_argument("--period", type=int, default=1)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--ndraws", type=int, default=50)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--caption", type=str, default=None)
    p.add_argument("--label", type=str, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df["market_id"] = list(zip(df["city"], df["period"]))
    df["sugar_per_g"] = df["sugar"] / df["weight"]
    mask = (df["city"] == args.city) & (df["period"] == args.period)
    xbar = df.loc[mask, "sugar_per_g"].mean()
    df["x_tilde"] = df["sugar_per_g"] - xbar
    rows = np.where(mask)[0]
    rows = rows[np.argsort(df.loc[rows, "product"].values)]
    s_obs = df.loc[rows, "share"].values
    x_tilde = df.loc[rows, "x_tilde"].values
    s0 = 1.0 - float(s_obs.sum())
    v = vi_grid(args.ndraws)
    delta_hat = berry_contraction(s_obs, x_tilde, args.sigma, v, tol=1e-14, maxit=50000)
    delta_mnl = np.log(s_obs) - math.log(max(s0, 1e-300))
    diff = delta_hat - delta_mnl

    out_df = pd.DataFrame(
        {
            "city": args.city,
            "period": args.period,
            "product": df.loc[rows, "product"].values.astype(int),
            "delta_hat": delta_hat,
            "delta_log_s_minus_log_s0": delta_mnl,
            "difference": diff,
        }
    )

    if args.out is None:
        stem = f"q2_delta_city{args.city}_period{args.period}_sigma{str(args.sigma).replace('.','p')}"
        args.out = os.path.join("tables", f"{stem}.tex")
    if args.caption is None:
        args.caption = f"Delta Comparison in City {args.city}, Period {args.period} (\\(\\sigma={args.sigma}\\))"
    if args.label is None:
        args.label = "tab:" + os.path.splitext(os.path.basename(args.out))[0]

    export_df_to_latex(
        df=out_df,
        out_tex_path=args.out,
        caption=args.caption,
        label=args.label,
        index=False,
        use_booktabs=True,
    )

if __name__ == "__main__":
    main()
