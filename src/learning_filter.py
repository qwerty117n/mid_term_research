"""
手順4〜7：企業ごとの推定・フィルター・期待収益性の計算（論文 3章・4.3〜4.4節、Appendix A 手順4〜7）

1企業について、次を行う。
  1. 四半期系列（X：ROE、s：業種シグナル）を暦四半期の格子に並べる（欠けた四半期は NaN）
  2. 拡大していくウィンドウで、8四半期ごとにパラメータを SMM 推定する（smm.py）
  3. 各推定時点のパラメータで、系列の最初からフィルターを回し直し、
     次の推定時点までの四半期について、予想 μ̂、事後分散 ν、サプライズ dW̃ を求める
     （論文 4.4節：「その時点で最新のパラメータで、利用可能な全履歴にわたって再帰計算する」）
  4. 命題3.2 により 1年先の期待 ROE（ExpProfit）と改訂幅（ΔExpProfit）を計算し、
     短期成分（予想 μ̂ による部分）と長期成分（長期水準 μ̄ による部分）に分解する

フィルターの式（論文 式(7)〜(10)、Appendix B と一致する形を採用）
  dW̃X = (ΔX − λ(μ̂ − X_{t−1})dt) / σ_X
  dW̃s = (Δs − μ̂ dt) / σ_s                      ※ INDUSTRY_INNOVATION = "level" なら (s − μ̂)/σ_s
  μ̂_t = μ̂ + κ(μ̄ − μ̂)dt + (λν/σ_X) dW̃X + (ν/σ_s) dW̃s
  ν_t = ν + (σ_μ² − 2κν − λ²ν²/σ_X² − ν²/σ_s²) dt   （負になる更新は棄却）
  ※ 論文 4.4節の実装式は業種項に λ が付き、ν の式の λ が2乗でないが、
     理論式（式(7)(8)・Appendix B）に合わせた。

独自に決めた点
  ・数値計算の方法（config.NUMERICAL_SCHEME）：
      論文の連続時間の式を dt = 0.25 でそのまま（陽的 Euler 法で）計算すると、
      業種シグナルの精度が高い（σ_s が小さい）企業では1期あたりの更新が大きくなりすぎて発散する。
      そこで既定では、μ̂ と ν を含む項を更新後の値で評価する半陰的な方法で計算する（"implicit"）。
      同じ連続時間モデルの離散化であり、dt を小さくした極限では論文の式と一致する。
      論文どおりの計算は "explicit" で選べる。
  ・事後分散の初期値：ν0 = σ_μ²/(2κ)（μ の定常分散）
  ・欠けた四半期：観測による更新を行わず、平均回帰（予測ステップ）だけ進める
"""

import zlib

import numpy as np
import pandas as pd

import config
import smm


def firm_seed(firm, j):
    """
    企業・推定時点ごとの乱数シード。
    NRI_CODE には英字を含むもの（例：A1766）があるため、文字列から計算する。
    hash() は実行ごとに値が変わるため使わず、crc32 で毎回同じ値になるようにする。
    """
    return config.RANDOM_SEED + zlib.crc32(str(firm).encode("utf-8")) * 1000 + j


def expected_profitability(x, mu_hat, mu_bar, lam, kap, tau=config.HORIZON):
    """命題3.2：E_t[X_{t+τ}] = aX + bμ̂ + cμ̄ と、その改訂幅・短期成分・長期成分"""
    a = np.exp(-lam * tau)
    if abs(lam - kap) < 1e-8:
        b = lam * tau * np.exp(-lam * tau)
    else:
        b = lam / (lam - kap) * (np.exp(-kap * tau) - np.exp(-lam * tau))
    c = 1 - a - b
    exp_profit = a * x + b * mu_hat + c * mu_bar
    # ΔExpProfit = E − X = b(μ̂ − X) + c(μ̄ − X)   （a + b + c = 1 より）
    return exp_profit, exp_profit - x, b * (mu_hat - x), c * (mu_bar - x)


def run_filter(x, s, p, mu_hat0, start, end):
    """
    t = 0..end の系列に対してフィルターを回し、start..end の結果を返す。
    p はパラメータ（dict）、mu_hat0 は予想の初期値。
    """
    lam, kap, sig_mu = p["lambda"], p["kappa"], p["sigma_mu"]
    mu_bar, sig_x, sig_s = p["mu_bar"], p["sigma_x"], p["sigma_s"]
    dt = config.DT

    valid = ~np.isnan(x) & ~np.isnan(s)
    t0 = int(np.argmax(valid))
    mu_hat = mu_hat0
    nu = sig_mu ** 2 / (2 * kap)
    x_prev, s_prev = x[t0], s[t0]
    out = {}

    level = config.INDUSTRY_INNOVATION == "level"
    implicit = config.NUMERICAL_SCHEME == "implicit"

    for t in range(t0 + 1, end + 1):
        nu_prev = nu
        if valid[t]:
            # サプライズ（論文 式(9)(10)）。μ̂ は1期前の予想
            dwx = (x[t] - x_prev - lam * (mu_hat - x_prev) * dt) / sig_x
            dws = (s[t] - mu_hat) / sig_s if level else (s[t] - s_prev - mu_hat * dt) / sig_s

            a_obs = lam ** 2 / sig_x ** 2 + 1 / sig_s ** 2  # 観測による精度
            if implicit:
                # μ̂ を含む項を新しい μ̂ で評価する半陰的な更新（dt を小さくした極限では式(7)と一致）
                k_x = lam * nu / sig_x ** 2
                k_s = nu / sig_s ** 2
                num = (mu_hat + kap * mu_bar * dt
                       + k_x * (x[t] - x_prev + lam * x_prev * dt)
                       + k_s * (s[t] if level else s[t] - s_prev))
                den = 1 + kap * dt + k_x * lam * dt + k_s * (1 if level else dt)
                mu_hat = num / den
                # ν も陰的に解く：ν' = ν + (σ_μ² − 2κν' − Aν'²)dt の正の解（常に正になる）
                qa, qb, qc = a_obs * dt, 1 + 2 * kap * dt, -(nu + sig_mu ** 2 * dt)
                nu_new = (-qb + np.sqrt(qb ** 2 - 4 * qa * qc)) / (2 * qa)
            else:
                # 論文どおりの陽的 Euler 法（σ_s が小さいと発散しやすい）
                mu_hat = (mu_hat + kap * (mu_bar - mu_hat) * dt
                          + lam * nu / sig_x * dwx + nu / sig_s * dws)
                nu_new = nu + (sig_mu ** 2 - 2 * kap * nu - a_obs * nu ** 2) * dt
            x_prev, s_prev = x[t], s[t]
        else:
            # 観測がない四半期：予測ステップのみ
            dwx = dws = np.nan
            if implicit:
                mu_hat = (mu_hat + kap * mu_bar * dt) / (1 + kap * dt)
                nu_new = (nu + sig_mu ** 2 * dt) / (1 + 2 * kap * dt)
            else:
                mu_hat = mu_hat + kap * (mu_bar - mu_hat) * dt
                nu_new = nu + (sig_mu ** 2 - 2 * kap * nu) * dt
        if nu_new >= 0:  # 論文：負になる更新は棄却
            nu = nu_new

        if t >= start and valid[t]:
            out[t] = (mu_hat, nu, dwx, dws, nu_prev * dwx, nu_prev * dws)
    return out


def process_firm(args):
    """1企業分の推定とフィルター。並列計算から呼ばれる。"""
    firm, g = args
    g = g.sort_values("quarter")
    quarters = pd.period_range(g["quarter"].min(), g["quarter"].max(), freq="Q")
    g = g.set_index("quarter").reindex(quarters)
    x = g["X"].to_numpy(dtype=float)
    s = g["s"].to_numpy(dtype=float)
    valid = ~np.isnan(x) & ~np.isnan(s)
    n_obs = np.cumsum(valid)

    # 推定時点：観測数が MIN_WINDOW_QUARTERS に達した四半期と、その後 8 四半期ごと
    if n_obs[-1] < config.MIN_WINDOW_QUARTERS:
        return [], []
    first = int(np.argmax(n_obs >= config.MIN_WINDOW_QUARTERS))
    est_points = list(range(first, len(x), config.REESTIMATE_EVERY))

    params, rows = [], []
    mu_hat0 = None
    for j, k in enumerate(est_points):
        p = smm.estimate(x[: k + 1], s[: k + 1], seed=firm_seed(firm, j))
        if p is None:
            continue
        if mu_hat0 is None:
            mu_hat0 = p["mu_bar"]  # 論文：最初のウィンドウの μ̄ で初期化
        end = est_points[j + 1] - 1 if j + 1 < len(est_points) else len(x) - 1
        params.append({"NRI_CODE": firm, "推定四半期": str(quarters[k]), "window_obs": int(n_obs[k]), **p})

        for t, (mu_hat, nu, dwx, dws, nu_dwx, nu_dws) in run_filter(x, s, p, mu_hat0, k, end).items():
            e, de, sr, lr = expected_profitability(x[t], mu_hat, p["mu_bar"], p["lambda"], p["kappa"])
            row = g.iloc[t]
            rows.append({
                "NRI_CODE": firm, "quarter": str(quarters[t]),
                "当期決算年月日": row["当期決算年月日"], "公表日": row["公表日"],
                "証券コード": row["証券コード"], "業種コード": row["業種コード"],
                "X": x[t], "s": s[t], "mu_hat": mu_hat, "nu": nu,
                "dW_x": dwx, "dW_s": dws, "nu_dW_x": nu_dwx, "nu_dW_s": nu_dws,
                "ExpProfit": e, "dExpProfit": de, "dExpProfit_short": sr, "dExpProfit_long": lr,
                "lambda": p["lambda"], "kappa": p["kappa"], "sigma_mu": p["sigma_mu"],
                "mu_bar": p["mu_bar"], "sigma_x": p["sigma_x"], "sigma_s": p["sigma_s"],
                "推定四半期": str(quarters[k]),
            })
    return params, rows
