# -*- coding: utf-8 -*-
"""
2_run_learning_model.py
=========================================================
Hou & Sinagl (2025) "Learning about Expected Profitability" の
構造的学習モデルを、1_generate_dummy_data.py が出力したパネルデータに
適用して「期待収益性改定(ΔExpProf)」を推定し、リターン予測力
(デシル/H-Lスプレッド、Fama-MacBeth回帰)を検証するスクリプト。

論文の各ステップとコードの対応:
  - 式(4)(5), Prop.3.1  : 企業ROEと潜在的収益性ドライバーμの学習モデル
  - 4.3節 (SMM推定)      : estimate_theta_smm()
  - 4.4節 (再帰フィルタ)  : run_recursive_filter()
  - Prop.3.2             : compute_expected_profitability_change()
  - 5.3-5.4節 (ポートフォリオソート/H-L) : decile_sort_spread()
  - 5.5節 (Fama-MacBeth)  : fama_macbeth()

【重要な注意】
本スクリプトはあくまで「手法を理解する」ための簡易実装です。
- SMMは論文の最適な重み行列Wの代わりに単位行列(等ウェイト)を使用
- 再帰的信念更新はProposition 3.1 / Appendix B(B.7)の式に従う
  (論文4.4節本文の数式はλが2箇所に出て3.1節と矛盾するように見えるため、
   数学的に導出されているProp.3.1 / Appendix Bの形を採用している)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_lyapunov

# ---------------------------------------------------------------
# 0. 設定
# ---------------------------------------------------------------
DATA_DIR = Path("/mnt/user-data/outputs/dummy_data")
OUT_DIR = Path("/mnt/user-data/outputs/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DT_Q = 0.25            # 1四半期 = 0.25年
REESTIMATE_EVERY = 8   # 何四半期ごとにSMMでθを再推定するか(論文4.3節: 2年= 8四半期)
MIN_WINDOW_Q = 16      # 推定に最低限必要な四半期数(4年)
N_SIMS = 40            # SMMのシミュレーションパス数(速度優先で少なめ。精度重視なら増やす)
HORIZON_YEARS = 1.0    # ΔExpProfの予測ホライズン(論文5.2節は1年先)
N_GROUPS = 5           # ポートフォリオの分位数(企業数が少ないのでデシルではなく5分位)
RNG = np.random.default_rng(123)


# ---------------------------------------------------------------
# 1. データ読み込み
# ---------------------------------------------------------------
def load_data():
    qpanel = pd.read_csv(DATA_DIR / "quarterly_fundamentals.csv", parse_dates=["quarter_end"])
    monthly = pd.read_csv(DATA_DIR / "monthly_panel.csv", parse_dates=["month_end", "quarter_end_used"])
    qpanel = qpanel.rename(columns={"roe_quarterly": "roe", "industry_roe_quarterly": "industry_roe"})
    qpanel["code"] = qpanel["code"].astype(str)
    monthly["code"] = monthly["code"].astype(str)
    return qpanel, monthly


# ---------------------------------------------------------------
# 2. SMM推定: θ_it = (λ_i, κ_i, σ_μi) を四半期ROEと業種ROEの
#    自己相関・共分散モーメントにマッチさせて推定する(論文4.3節)
# ---------------------------------------------------------------
def ar1_init(series):
    """AR(1) OLS: X_t = a + b*X_{t-1} + e から平均回帰速度の初期値を得る(論文4.3節の初期化)。"""
    x = series[:-1]
    y = series[1:]
    if len(x) < 3 or np.std(x) < 1e-10:
        return 0.4
    b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    b = np.clip(b, 1e-4, 0.999)
    lam0 = (1 - b) / DT_Q
    return float(np.clip(lam0, 0.01, 4.9))


def data_moments(X, s):
    """データから3つのターゲットモーメントを計算: (i)ROEの自己相関, (ii)業種ROEの自己相関, (iii)共分散。"""
    if len(X) < 4:
        return None
    ac_x = np.corrcoef(X[:-1], X[1:])[0, 1]
    ac_s = np.corrcoef(s[:-1], s[1:])[0, 1]
    cov_xs = np.cov(X, s, ddof=1)[0, 1]
    return np.array([ac_x, ac_s, cov_xs])


def analytic_moments(theta, sigma_x, sigma_s):
    """候補パラメータθ=(λ,κ,σ_μ)のもとでの理論(定常状態)モーメントを、
    シミュレーションではなく離散リアプノフ方程式を解くことで解析的に求める。

    状態ベクトル z_t=(X_t, μ_t)' の線形状態空間表現:
        z_t = A z_{t-1} + b + Q^(1/2) e_t ,  e_t ~ N(0, I)
    のもとで、定常分散 Σ は  Σ = A Σ A' + Q  (離散リアプノフ方程式) を満たす。
    業種シグナル s_t = μ_t + σ_s * ノイズ (式6の近似)として、
      Var(X), Var(s), Cov(X,μ)  → ラグ1自己相関・共分散を計算する。

    モンテカルロ・シミュレーションを使わないため厳密には論文のSMMとは異なるが、
    同じモーメント条件を高速かつ数値的に安定な形で近似する簡略版として採用する。
    """
    lam, kap, sig_mu = theta
    lam = max(lam, 1e-4)
    kap = max(kap, 1e-4)
    sig_mu = max(sig_mu, 1e-5)
    dt = DT_Q

    A = np.array([[1 - lam * dt, lam * dt],
                  [0.0, 1 - kap * dt]])
    Q = np.array([[sigma_x ** 2 * dt, 0.0],
                  [0.0, sig_mu ** 2 * dt]])

    # 数値的な安定性のため固有値をチェックし、境界付近ならわずかに縮小する
    eigvals = np.linalg.eigvals(A)
    if np.max(np.abs(eigvals)) >= 0.999:
        A = A * (0.998 / np.max(np.abs(eigvals)))

    try:
        Sigma = solve_discrete_lyapunov(A, Q)
    except (np.linalg.LinAlgError, ValueError):
        return np.array([0.0, 0.0, 0.0])

    AS = A @ Sigma  # Cov(z_t, z_{t-1})

    var_x = max(Sigma[0, 0], 1e-10)
    var_mu = max(Sigma[1, 1], 1e-10)
    var_s = var_mu + sigma_s ** 2
    cov_x_mu = Sigma[0, 1]

    ac_x = AS[0, 0] / var_x
    ac_s = AS[1, 1] / var_s          # s=μ+独立ノイズ なので Cov(s_t,s_{t-1})=Cov(μ_t,μ_{t-1})
    cov_xs = cov_x_mu                # Cov(X,s)=Cov(X,μ) (ノイズは独立)

    return np.array([ac_x, ac_s, cov_xs])


def estimate_theta_smm(X, s):
    """窓内データ(X=企業ROE, s=業種ROE)からSMMでθ=(λ,κ,σ_μ)を推定する。"""
    n = len(X)
    mu_bar0 = float(np.mean(X))
    sigma_x0 = float(np.std(X, ddof=1)) if n > 1 else 0.02
    sigma_s0 = float(np.std(s, ddof=1)) if n > 1 else 0.02
    sigma_s0 = max(sigma_s0, 1e-4)

    lam0 = ar1_init(X)
    kap0 = ar1_init(s)
    sig_mu0 = max(sigma_x0 * 0.5, 1e-3)

    m_data = data_moments(X, s)
    if m_data is None:
        return dict(lambda_hat=lam0, kappa_hat=kap0, sigma_mu_hat=sig_mu0,
                     mu_bar_hat=mu_bar0, sigma_x_hat=sigma_x0, sigma_s_hat=sigma_s0)

    def objective(theta):
        m_sim = analytic_moments(theta, sigma_x=sigma_x0, sigma_s=sigma_s0)
        diff = m_sim - m_data
        return float(diff @ diff)  # 等ウェイト(W=単位行列)のSMM目的関数

    # SMMの探索範囲はAR(1)初期値の近傍に限定する。
    # 理由: 自己相関・共分散の3モーメントだけではλを大域的に強く識別できず、
    # 無制約だと数値探索が境界(4.9)に張り付きやすい(弱識別の問題)。
    # AR(1)回帰は各系列の自己回帰係数から直接λ,κを推定できているため、
    # SMMはその近傍での「補正」とみなし、局所的な範囲に制約する。
    lam_bounds = (max(0.02, 0.2 * lam0), min(4.9, 5 * lam0))
    kap_bounds = (max(0.02, 0.2 * kap0), min(4.9, 5 * kap0))
    lam0 = float(np.clip(lam0, *lam_bounds))
    kap0 = float(np.clip(kap0, *kap_bounds))

    res = minimize(
        objective, x0=[lam0, kap0, sig_mu0], method="Nelder-Mead",
        bounds=[lam_bounds, kap_bounds, (1e-4, 0.3)],
        options={"maxiter": 200, "maxfev": 400, "xatol": 1e-4, "fatol": 1e-8},
    )
    lam_hat, kap_hat, sig_mu_hat = res.x

    return dict(lambda_hat=float(lam_hat), kappa_hat=float(kap_hat),
                sigma_mu_hat=float(sig_mu_hat), mu_bar_hat=mu_bar0,
                sigma_x_hat=sigma_x0, sigma_s_hat=sigma_s0)


def build_theta_schedule(qpanel):
    """各企業について、拡大窓(expanding window)で REESTIMATE_EVERY 四半期ごとに
    θを再推定し、次の再推定日まで持ち越すスケジュールを作る(論文4.3節)。"""
    records = []
    for code, g in qpanel.groupby("code"):
        g = g.sort_values("quarter_end").reset_index(drop=True)
        X_full = g["roe"].values
        s_full = g["industry_roe"].values
        qends = g["quarter_end"].values

        for end_idx in range(MIN_WINDOW_Q - 1, len(g), REESTIMATE_EVERY):
            X = X_full[: end_idx + 1]
            s = s_full[: end_idx + 1]
            theta = estimate_theta_smm(X, s)
            theta["code"] = code
            theta["estimation_quarter_end"] = qends[end_idx]
            records.append(theta)

    theta_df = pd.DataFrame(records)
    return theta_df


def attach_theta_to_quarters(qpanel, theta_df):
    """推定日以降の各四半期に「直近の推定値」をcarry-forwardして割り当てる。"""
    out = []
    for code, g in qpanel.groupby("code"):
        g = g.sort_values("quarter_end").reset_index(drop=True)
        th = theta_df[theta_df["code"] == code].sort_values("estimation_quarter_end")
        if th.empty:
            continue
        merged = pd.merge_asof(
            g, th.drop(columns=["code"]),
            left_on="quarter_end", right_on="estimation_quarter_end",
            direction="backward",
        )
        merged = merged.dropna(subset=["lambda_hat"])
        out.append(merged)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------
# 3. 再帰フィルタ: 月次でμ̂とνを更新し、サプライズ(dW)を復元する
#    (論文4.4節 / Proposition 3.1 / Appendix B(B.7)(B.10))
# ---------------------------------------------------------------
def run_recursive_filter(qtheta):
    """四半期粒度で信念μ̂とその不確実性νを再帰的に更新する。

    【数値安定性についての重要な注意】
    Proposition 3.1 (式7)を素朴にオイラー法で離散化すると、
        μ̂_t = μ̂_{t-1} + κ(μ̄-μ̂_{t-1})dt + (λν/σx)dWx + (ν/σs)dWs
    において dWx 自体が μ̂_{t-1} を含む(式9)ため、整理すると
        μ̂_t = μ̂_{t-1}(1 - Λdt) + (forcing項)
    という形になり、実効的な減衰率 Λ = κ + λ²ν/σx² + ν/σs² が大きい
    (λが大きい/νが大きい/σxが小さい)場合、|1-Λdt| > 1 となって
    オイラー法が発散してしまう(いわゆる剛性(stiff)なSDEの数値不安定性)。

    そこで、この線形ODEを区間内で厳密に解く「指数積分法」を用いる:
        μ̂_t = μ̂_{t-1} exp(-Λdt) + (F/Λ)(1-exp(-Λdt))
    これはΛの大きさによらず常に安定(exp(-Λdt)∈(0,1])。
    """
    out_rows = []
    for code, g in qtheta.groupby("code"):
        g = g.sort_values("quarter_end").reset_index(drop=True)
        n = len(g)
        mu_hat = np.empty(n)
        nu = np.empty(n)

        # 初期化: 論文4.4節「初期化と信号ボラティリティ」に対応
        mu_hat[0] = g.loc[0, "mu_bar_hat"]
        nu[0] = g.loc[0, "sigma_x_hat"] ** 2  # 初期の不確実性は実現ROE分散で近似

        dWx = np.zeros(n)
        dWs = np.zeros(n)

        for t in range(1, n):
            lam = g.loc[t, "lambda_hat"]
            kap = g.loc[t, "kappa_hat"]
            sig_mu = g.loc[t, "sigma_mu_hat"]
            mu_bar = g.loc[t, "mu_bar_hat"]
            sig_x = max(g.loc[t, "sigma_x_hat"], 1e-4)
            sig_s = max(g.loc[t, "sigma_s_hat"], 1e-4)

            X_prev, X_t = g.loc[t - 1, "roe"], g.loc[t, "roe"]
            s_prev, s_t = g.loc[t - 1, "industry_roe"], g.loc[t, "industry_roe"]
            dX = X_t - X_prev
            ds = s_t - s_prev
            mu_prev, nu_prev = mu_hat[t - 1], nu[t - 1]

            # 式(9)(10): サプライズ(イノベーション)の復元(診断用。μ̂の更新自体には
            # 下の指数積分法を用いるが、報告用の dW は定義式どおりに計算する)
            wx = (dX - lam * (mu_prev - X_prev) * DT_Q) / sig_x
            ws = (ds - mu_prev * DT_Q) / sig_s
            dWx[t], dWs[t] = wx, ws

            # 実効減衰率Λと強制項F(上記コメント参照)
            Lambda = kap + (lam ** 2) * nu_prev / sig_x ** 2 + nu_prev / sig_s ** 2
            F = (
                kap * mu_bar
                + (lam * nu_prev / sig_x ** 2) * (dX / DT_Q)
                + (lam ** 2 * nu_prev / sig_x ** 2) * X_prev
                + (nu_prev / sig_s ** 2) * (ds / DT_Q)
            )
            if Lambda > 1e-8:
                decay = np.exp(-Lambda * DT_Q)
                mu_hat[t] = mu_prev * decay + (F / Lambda) * (1 - decay)
            else:
                mu_hat[t] = mu_prev + F * DT_Q

            # 式(8): 事後分散(リカッチ方程式の離散近似)。負値は棄却して前期の値を維持
            nu_new = nu_prev + (
                sig_mu ** 2 - 2 * kap * nu_prev
                - (lam ** 2 / sig_x ** 2) * nu_prev ** 2
                - (1 / sig_s ** 2) * nu_prev ** 2
            ) * DT_Q
            nu[t] = nu_new if nu_new > 0 else nu_prev

        g = g.copy()
        g["mu_hat"] = mu_hat
        g["nu"] = nu
        g["dW_x"] = dWx
        g["dW_s"] = dWs
        out_rows.append(g)

    return pd.concat(out_rows, ignore_index=True)


# ---------------------------------------------------------------
# 4. 期待収益性改定 ΔExpProf を計算(Proposition 3.2)
#    ΔExpProf = b*(μ̂-X) + c*(μ̄-X)  ※ (a-1)+b+c=0 を用いた同値変形
#    第1項 = 短期更新コンポーネント, 第2項 = 長期アンカーコンポーネント
# ---------------------------------------------------------------
def compute_expected_profitability_change(filtered, horizon_years=HORIZON_YEARS):
    lam = filtered["lambda_hat"].values
    kap = filtered["kappa_hat"].values.copy()
    same = np.abs(lam - kap) < 1e-6
    kap[same] += 1e-6  # 縮退回避

    a = np.exp(-lam * horizon_years)
    b = (lam / (lam - kap)) * (np.exp(-kap * horizon_years) - np.exp(-lam * horizon_years))
    c = 1 - a - b

    X = filtered["roe"].values
    mu_hat = filtered["mu_hat"].values
    mu_bar = filtered["mu_bar_hat"].values

    short_run = b * (mu_hat - X)
    long_run = c * (mu_bar - X)
    exp_change = short_run + long_run

    out = filtered.copy()
    out["exp_prof_change"] = exp_change
    out["short_run_component"] = short_run
    out["long_run_component"] = long_run
    out["nu_x_dWx"] = out["nu"] * out["dW_x"]  # 学習チャネル検証用(表VIII対応)
    out["nu_x_dWs"] = out["nu"] * out["dW_s"]
    return out


# ---------------------------------------------------------------
# 5. 月次パネルへ信号を接続し、翌月リターンとポートフォリオを作る
# ---------------------------------------------------------------
def merge_signal_to_monthly(monthly, signal_q):
    sig = signal_q[[
        "code", "quarter_end", "exp_prof_change", "short_run_component",
        "long_run_component", "nu_x_dWx", "nu_x_dWs",
    ]].rename(columns={"quarter_end": "quarter_end_used"})
    sig["code"] = sig["code"].astype(str)
    m = monthly.merge(sig, on=["code", "quarter_end_used"], how="inner")
    m = m.sort_values(["code", "month_end"]).reset_index(drop=True)
    # 翌月リターン(シグナルは月末t時点、リターンはt+1で実現)
    m["next_month_return"] = m.groupby("code")["monthly_return"].shift(-1)
    m = m.dropna(subset=["next_month_return", "exp_prof_change"])
    return m


def newey_west_tstat(x, lags=3):
    """簡易Newey-West HAC t検定(時系列平均がゼロかどうかの検定)。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    mean = x.mean()
    resid = x - mean
    gamma0 = np.sum(resid ** 2) / n
    var = gamma0
    for lag in range(1, lags + 1):
        cov = np.sum(resid[lag:] * resid[:-lag]) / n
        weight = 1 - lag / (lags + 1)
        var += 2 * weight * cov
    se = np.sqrt(max(var, 1e-12) / n)
    t = mean / se if se > 0 else np.nan
    return mean, t


def decile_sort_spread(m, signal_col, n_groups=N_GROUPS, value_weight=True, label=""):
    """月次クロスセクションでsignal_colに基づきn分位ソートし、
    等加重/価値加重の分位別リターンとH-Lスプレッドを計算する(論文5.4節)。"""
    monthly_group_ret = []
    for month, g in m.groupby("month_end"):
        g = g.copy()
        if g[signal_col].nunique() < n_groups:
            continue
        g["grp"] = pd.qcut(g[signal_col], n_groups, labels=False, duplicates="drop") + 1
        if value_weight:
            ret_by_grp = g.groupby("grp").apply(
                lambda x: np.average(x["next_month_return"], weights=x["market_cap"])
            )
        else:
            ret_by_grp = g.groupby("grp")["next_month_return"].mean()
        row = {"month_end": month}
        row.update(ret_by_grp.to_dict())
        monthly_group_ret.append(row)

    df = pd.DataFrame(monthly_group_ret).set_index("month_end").sort_index()
    df = df.rename(columns=lambda c: f"G{int(c)}" if isinstance(c, (int, float)) else c)
    top = df.columns[-1] if "G" not in "".join(map(str, df.columns[:1])) else f"G{n_groups}"
    top_col = [c for c in df.columns if c == f"G{n_groups}"][0]
    bot_col = [c for c in df.columns if c == "G1"][0]
    df["H_L"] = df[top_col] - df[bot_col]

    summary = {}
    for col in df.columns:
        mean_ret, t = newey_west_tstat(df[col].values)
        summary[col] = {"mean_monthly_return_%": mean_ret * 100, "t_stat_NW": t}
    summary_df = pd.DataFrame(summary).T
    print(f"\n--- {label}: {n_groups}分位ソート (価値加重={value_weight}) ---")
    print(summary_df.round(3))
    return df, summary_df


# ---------------------------------------------------------------
# 6. Fama-MacBeth横断面回帰(論文5.5節、表VI相当)
# ---------------------------------------------------------------
def fama_macbeth(m, y_col, x_cols):
    coefs = []
    for month, g in m.groupby("month_end"):
        g = g.dropna(subset=[y_col] + x_cols)
        if len(g) < len(x_cols) + 5:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[c].values for c in x_cols])
        y = g[y_col].values
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        coefs.append(beta)
    coefs = np.array(coefs)
    names = ["const"] + x_cols
    rows = {}
    for j, name in enumerate(names):
        mean_b, t = newey_west_tstat(coefs[:, j])
        rows[name] = {"mean_coef": mean_b, "t_stat_NW": t}
    result = pd.DataFrame(rows).T
    print(f"\n--- Fama-MacBeth回帰: {y_col} ~ {' + '.join(x_cols)} (月次クロスセクション数={len(coefs)}) ---")
    print(result.round(4))
    return result


# ---------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------
def main():
    print("1. データ読み込み...")
    qpanel, monthly = load_data()

    print("2. SMMによる学習パラメータ(λ, κ, σ_μ)推定中... (数分かかる場合があります)")
    theta_df = build_theta_schedule(qpanel)
    theta_df.to_csv(OUT_DIR / "estimated_theta.csv", index=False)
    print(f"   推定件数: {len(theta_df)} (企業×再推定時点)")

    qtheta = attach_theta_to_quarters(qpanel, theta_df)

    print("3. 再帰フィルタで信念μ̂と不確実性νを復元中...")
    filtered = run_recursive_filter(qtheta)

    print("4. Proposition 3.2に基づき期待収益性改定ΔExpProfを計算中...")
    signal_q = compute_expected_profitability_change(filtered, horizon_years=HORIZON_YEARS)
    signal_q.to_csv(OUT_DIR / "quarterly_signals.csv", index=False)

    print("5. 月次パネルにシグナルを接続し、翌月リターンと結合...")
    m = merge_signal_to_monthly(monthly, signal_q)
    m.to_csv(OUT_DIR / "monthly_panel_with_signal.csv", index=False)
    print(f"   有効サンプル数(企業×月): {len(m)}")

    print("\n===== 6. ポートフォリオソート(表III/IV/V 相当) =====")
    decile_sort_spread(m, "exp_prof_change", label="ΔExpProf (全体)")
    decile_sort_spread(m, "short_run_component", label="短期更新コンポーネント")
    decile_sort_spread(m, "long_run_component", label="長期アンカーコンポーネント")

    print("\n===== 7. Fama-MacBeth横断面回帰(表VI 相当) =====")
    m["log_me"] = np.log(m["market_cap"])
    fama_macbeth(m, "next_month_return", ["exp_prof_change"])
    fama_macbeth(m, "next_month_return", ["exp_prof_change", "roe", "log_me"])

    print("\n===== 8. 学習チャネルの直接検証(表VIII 相当: ν×dW の価格付け) =====")
    fama_macbeth(m, "next_month_return", ["nu_x_dWx", "roe"])

    print(f"\n結果CSVは {OUT_DIR} に保存されました。")


if __name__ == "__main__":
    main()
