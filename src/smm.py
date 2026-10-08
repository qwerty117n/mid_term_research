"""
手順4・5：学習パラメータ θ = (λ, κ, σ_μ) の SMM 推定（論文 4.3節、Appendix A 手順4・5）

モデル（四半期刻み dt = 0.25 で離散化）
  μ_{t+1} = μ_t + κ(μ̄ − μ_t)dt + σ_μ √dt ε_μ        潜在的な収益性の源泉（観測できない）
  X_{t+1} = X_t + λ(μ_t − X_t)dt + σ_X √dt ε_X        企業の ROE（観測できる）
  s_t     = μ_t + σ_s ε_s                              業種シグナル（式(6)）

推定の手順（論文どおり）
  ・μ̄, σ_X, σ_s はウィンドウ内の標本値で固定する（μ̄ = X の平均、σ_X = X の標準偏差、σ_s = s の標準偏差）
  ・3つのモーメントを一致させる：(i) X の1次自己相関、(ii) s の1次自己相関、(iii) X と s の共分散
  ・シミュレーションはウィンドウと同じ長さで行う（短い標本での自己相関のバイアスも再現される）
  ・初期値は X, s の AR(1) の OLS 推定から求める

論文に記載がなく独自に決めた点
  ・重み行列：diag(1, 1, 1/(σ_X σ_s)²)。共分散の単位をそろえ、3つのモーメントが同程度に効くようにする
  ・最適化：Nelder-Mead。λ, κ は (PARAM_LOWER, PARAM_UPPER) に収まるよう変数変換して探索する
  ・乱数は企業・推定時点ごとに固定する（目的関数を滑らかにするため。common random numbers）
"""

import numpy as np
from scipy.optimize import minimize

import config


# ===== モーメントの計算 =====
def _lag1_autocorr(a: np.ndarray) -> float:
    """欠損を含む系列の1次自己相関（t と t−1 の両方がそろう組だけを使う）"""
    x0, x1 = a[:-1], a[1:]
    ok = ~np.isnan(x0) & ~np.isnan(x1)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(x0[ok], x1[ok])[0, 1])


def data_moments(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(x) & ~np.isnan(s)
    cov = float(np.cov(x[ok], s[ok])[0, 1])
    return np.array([_lag1_autocorr(x), _lag1_autocorr(s), cov])


def _rowwise_autocorr(a: np.ndarray) -> np.ndarray:
    x0, x1 = a[:, :-1], a[:, 1:]
    x0 = x0 - x0.mean(axis=1, keepdims=True)
    x1 = x1 - x1.mean(axis=1, keepdims=True)
    den = np.sqrt((x0 ** 2).sum(axis=1) * (x1 ** 2).sum(axis=1))
    return (x0 * x1).sum(axis=1) / np.where(den > 0, den, np.nan)


def simulated_moments(theta, mu_bar, sig_x, sig_s, shocks) -> np.ndarray:
    lam, kap, sig_mu = theta
    e_mu, e_x, e_s, z0 = shocks
    n_paths, n_steps = e_mu.shape
    sq = np.sqrt(config.DT)

    # μ は定常分布から、X は μ から出発し、助走期間を捨てる
    mu = mu_bar + sig_mu / np.sqrt(2 * kap) * z0
    x = mu.copy()
    xs = np.empty((n_paths, n_steps))
    mus = np.empty((n_paths, n_steps))
    for t in range(n_steps):
        x = x + lam * (mu - x) * config.DT + sig_x * sq * e_x[:, t]
        mu = mu + kap * (mu_bar - mu) * config.DT + sig_mu * sq * e_mu[:, t]
        xs[:, t] = x
        mus[:, t] = mu

    b = config.SIM_BURN_IN
    xs, mus = xs[:, b:], mus[:, b:]
    ss = mus + sig_s * e_s[:, b:]

    ac_x = np.nanmean(_rowwise_autocorr(xs))
    ac_s = np.nanmean(_rowwise_autocorr(ss))
    xc = xs - xs.mean(axis=1, keepdims=True)
    sc = ss - ss.mean(axis=1, keepdims=True)
    cov = np.mean((xc * sc).sum(axis=1) / (xs.shape[1] - 1))
    return np.array([ac_x, ac_s, cov])


# ===== パラメータ変換（制約付き -> 制約なし） =====
def _to_unconstrained(theta):
    lam, kap, sig_mu = theta
    lo, hi = config.PARAM_LOWER, config.PARAM_UPPER
    logit = lambda v: np.log((v - lo) / (hi - v))
    return np.array([logit(lam), logit(kap), np.log(sig_mu)])


def _to_constrained(u):
    lo, hi = config.PARAM_LOWER, config.PARAM_UPPER
    sig = lambda v: lo + (hi - lo) / (1 + np.exp(-v))
    return np.array([sig(u[0]), sig(u[1]), np.exp(u[2])])


def _ar1_speed(a: np.ndarray) -> float:
    """AR(1) の OLS 係数 φ から、平均回帰の速さ (1 − φ)/dt を求める（初期値用）"""
    phi = _lag1_autocorr(a)
    if np.isnan(phi):
        phi = 0.5
    return float(np.clip((1 - phi) / config.DT, 0.05, config.PARAM_UPPER - 0.05))


# ===== 推定 =====
def estimate(x: np.ndarray, s: np.ndarray, seed: int) -> dict:
    """
    1つのウィンドウ（企業 i の t0〜t の四半期系列）からパラメータを推定する。
    x, s は欠損（NaN）を含んでよい。
    """
    ok = ~np.isnan(x) & ~np.isnan(s)
    mu_bar = float(np.mean(x[ok]))
    sig_x = float(np.std(x[ok], ddof=1))
    sig_s = float(np.std(s[ok], ddof=1))
    m_data = data_moments(x, s)
    if not (sig_x > 0 and sig_s > 0) or np.isnan(m_data).any():
        return None

    # 初期値（論文：AR(1) の OLS）
    lam0 = _ar1_speed(x)
    kap0 = _ar1_speed(s)
    sig_mu0 = max(sig_s * np.sqrt(2 * kap0) * 0.5, 1e-4)

    rng = np.random.default_rng(seed)
    n_steps = len(x) + config.SIM_BURN_IN
    shocks = (
        rng.standard_normal((config.N_SIM_PATHS, n_steps)),
        rng.standard_normal((config.N_SIM_PATHS, n_steps)),
        rng.standard_normal((config.N_SIM_PATHS, n_steps)),
        rng.standard_normal(config.N_SIM_PATHS),
    )
    w = np.array([1.0, 1.0, 1.0 / (sig_x * sig_s) ** 2])

    def objective(u):
        theta = _to_constrained(u)
        g = simulated_moments(theta, mu_bar, sig_x, sig_s, shocks) - m_data
        if np.isnan(g).any():
            return 1e10
        return float(g @ (w * g))

    u0 = _to_unconstrained((lam0, kap0, sig_mu0))
    res = minimize(objective, u0, method="Nelder-Mead",
                   options={"maxiter": config.SMM_MAXITER, "xatol": 1e-4, "fatol": 1e-10})
    lam, kap, sig_mu = _to_constrained(res.x)
    return {
        "lambda": lam, "kappa": kap, "sigma_mu": sig_mu,
        "mu_bar": mu_bar, "sigma_x": sig_x, "sigma_s": sig_s,
        "lambda0": lam0, "kappa0": kap0,
        "m_ac_x": m_data[0], "m_ac_s": m_data[1], "m_cov": m_data[2],
        "smm_obj": res.fun, "smm_converged": bool(res.success),
    }
