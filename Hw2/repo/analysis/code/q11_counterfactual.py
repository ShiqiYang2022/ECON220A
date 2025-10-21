import os, time, numpy as np, pandas as pd
from numpy.linalg import solve
from scipy.stats import norm

np.set_printoptions(precision=6, suppress=True)

def now(): return time.strftime("%H:%M:%S")

def vi_grid(N=50):
    u = np.linspace(0.1, 0.9, N)
    return norm.ppf(u)

def shares_from_delta(delta_vec, x_tilde_vec, sigma, v_draws):
    util = delta_vec.reshape(-1,1) + sigma * x_tilde_vec.reshape(-1,1) * v_draws.reshape(1,-1)
    a = np.maximum(0.0, np.max(util, axis=0, keepdims=True))
    expu = np.exp(util - a)
    denom = np.exp(0.0 - a) + np.sum(expu, axis=0, keepdims=True)
    sij = expu / denom
    s_mean = np.mean(sij, axis=1)
    return s_mean, sij

def berry_contraction(s_obs, x_tilde, sigma, v, tol=1e-12, maxit=80000):
    s_sum = float(s_obs.sum()); s_sum = min(s_sum, 0.999999)
    s0 = 1.0 - s_sum
    delta = np.log(np.clip(s_obs, 1e-300, 1.0)) - np.log(max(s0, 1e-300))
    for _ in range(maxit):
        s_pred, _ = shares_from_delta(delta, x_tilde, sigma, v)
        new_delta = delta + (np.log(s_obs) - np.log(np.clip(s_pred,1e-300,1.0)))
        if np.max(np.abs(new_delta - delta)) < tol: return new_delta
        delta = new_delta
    return delta

def read_beta_sigma(csv_path="output/blp_q8_newton_results.csv",
                    beta_default=-3.5, sigma_default=80.0):
    if os.path.exists(csv_path):
        tab = pd.read_csv(csv_path)
        beta_price = float(tab.loc[tab["parameter"]=="alpha","estimate"].values[0]) # 注意：这是 β_price
        sigma_hat  = float(tab.loc[tab["parameter"]=="sigma","estimate"].values[0])
    else:
        beta_price, sigma_hat = beta_default, sigma_default
    alpha_struct = -beta_price  # 结构 α>0
    return alpha_struct, beta_price, sigma_hat

def prepare_market(data_path="input/data_yoghurt.csv",
                   city=1, period=1, center_mode="market"):
    df = pd.read_csv(data_path)
    df["market_id"] = list(zip(df["city"], df["period"]))
    df["sugar_per_g"] = df["sugar"]/df["weight"]
    if center_mode=="market":
        df["x_tilde"] = df["sugar_per_g"] - df.groupby("market_id")["sugar_per_g"].transform("mean")
    else:
        xbar = df.loc[(df["city"]==1)&(df["period"]==1),"sugar_per_g"].mean()
        df["x_tilde"] = df["sugar_per_g"] - xbar
    m = (df["city"]==city) & (df["period"]==period)
    dfm = df.loc[m, ["product","price","share","x_tilde"]].copy().sort_values("product").reset_index(drop=True)
    return dfm

def demand_and_J(p_vec, delta_base, p_base, x_tilde, sigma, alpha_struct, v):
    # 在给定价格 p 下的 mean-utility:
    # δ(p) = δ_base + β_price * (p - p_base)，其中 β_price = -α
    beta_price = -alpha_struct
    delta_p = delta_base + beta_price * (p_vec - p_base)
    s_bar, s_ij = shares_from_delta(delta_p, x_tilde, sigma, v)
    N = s_ij.shape[1]
    M = (s_ij @ s_ij.T) / N          # E[s_ij s_il]
    own = s_bar - np.diag(M)         # E[s_ij(1 - s_ij)]
    dSdp = alpha_struct * M
    for j in range(len(own)):
        dSdp[j,j] = -alpha_struct * own[j]
    return s_bar, dSdp, delta_p

def compute_mc_identity(p_obs, s_obs, dSdp, O=None):
    # 预合并：单产品所有权默认 O=I；一般情形 μ = -[(O∘D)^T]^{-1} s
    J = dSdp.shape[0]
    if O is None: O = np.eye(J)
    Omega = O * dSdp
    markups = -solve(Omega.T, s_obs)
    return p_obs - markups

def cs_dollars(p_vec, delta_base, p_base, x_tilde, sigma, alpha_struct, v):
    beta_price = -alpha_struct
    delta_p = delta_base + beta_price * (p_vec - p_base)
    util = delta_p.reshape(-1,1) + sigma * x_tilde.reshape(-1,1) * v.reshape(1,-1)
    a = np.maximum(0.0, np.max(util, axis=0, keepdims=True))
    logsum = a + np.log(np.exp(0.0 - a) + np.sum(np.exp(util - a), axis=0, keepdims=True))
    cs = (1.0/alpha_struct) * float(np.mean(logsum))
    return cs

def q11_q12_counterfactual(data_path="input/data_yoghurt.csv",
                            city=1, period=1,
                            center_mode="market",
                            tol=1e-10, maxit=10_000,
                            outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    print(f"[{now()}] Start Q11–Q12")

    # 参数与市场数据
    alpha_struct, beta_price, sigma = read_beta_sigma()
    dfm = prepare_market(data_path, city, period, center_mode=center_mode)
    p_obs = dfm["price"].to_numpy(float)
    s_obs = dfm["share"].to_numpy(float)
    x_t   = dfm["x_tilde"].to_numpy(float)
    v = vi_grid(50)

    # 基准 δ（在观测价格与 σ 下，用 Berry 收缩与 s_obs 对齐）
    t0 = time.time()
    delta_base = berry_contraction(s_obs, x_t, sigma, v)
    print(f"[{now()}] delta_base via Berry done in {time.time()-t0:.2f}s")

    # 构造基准需求与雅可比；用它从观测价恢复边际成本（单品公司）
    s0, dSdp0, _ = demand_and_J(p_obs, delta_base, p_obs, x_t, sigma, alpha_struct, v)
    mc = compute_mc_identity(p_obs, s0, dSdp0, O=None)

    # === Q11: 合并情形（2 与 3 同一家公司），边际成本保持 mc 不变 ===
    J = len(p_obs)
    O_merge = np.eye(J)
    # 产品编号假定为 1..5，合并 {2,3}
    O_merge[1,2] = O_merge[2,1] = 1.0

    # 价格固定点迭代：p = c - [(O∘D(p))^T]^{-1} s(p)
    p = p_obs.copy()
    for it in range(1, maxit+1):
        s, dSdp, _ = demand_and_J(p, delta_base, p_obs, x_t, sigma, alpha_struct, v)
        Omega = O_merge * dSdp
        p_new = mc - solve(Omega.T, s)
        diff = np.max(np.abs(p_new - p))
        if it % 50 == 0 or diff < tol:
            print(f"[{now()}] iter={it:04d}  max|Δp|={diff:.3e}")
        p = p_new
        if diff < tol:
            break
    p_cf = p.copy()

    # 保存 Q11 结果
    tab11 = pd.DataFrame({
        "product": dfm["product"].astype(int).tolist(),
        "price_old": p_obs,
        "price_new": p_cf
    })
    out11 = os.path.join(outdir, f"q11_prices_counterfactual_city{city}_period{period}.csv")
    tab11.to_csv(out11, index=False)

    # === Q12: 消费者福利差（美元） ===
    cs_old = cs_dollars(p_obs, delta_base, p_obs, x_t, sigma, alpha_struct, v)
    cs_new = cs_dollars(p_cf,  delta_base, p_obs, x_t, sigma, alpha_struct, v)
    dCS = cs_new - cs_old

    tab12 = pd.DataFrame({
        "city":[city], "period":[period],
        "CS_no_merger":[cs_old],
        "CS_merger":[cs_new],
        "Delta_CS":[dCS]
    })
    out12 = os.path.join(outdir, f"q12_welfare_city{city}_period{period}.csv")
    tab12.to_csv(out12, index=False)

    print("\n=== Q11: Prices (old vs new) ===")
    print(tab11.round(6).to_string(index=False))
    print("\n=== Q12: Welfare (USD per consumer) ===")
    print(tab12.round(6).to_string(index=False))
    print(f"\nSaved → {out11}\nSaved → {out12}")
    return tab11, tab12

if __name__ == "__main__":
    q11_q12_counterfactual(
        data_path="input/data_yoghurt.csv",
        city=1, period=1,
        center_mode="market",   # 若需完全对齐官方 MATLAB，设为 "global_ref_11"
        tol=1e-10, maxit=10000,
        outdir="output"
    )
