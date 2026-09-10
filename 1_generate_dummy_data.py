# -*- coding: utf-8 -*-
"""
1_generate_dummy_data.py
=========================================================
Hou & Sinagl (2025) "Learning about Expected Profitability" の
構造的学習モデルを日本市場で検証するための【ダミーデータ生成】スクリプト。

このスクリプトは実データを一切使わず、論文のモデル(3節)に沿って
「真のパラメータ」から人工的にROEパネルとリターンを生成する。

なぜこの設計にしたか
---------------------
- 実データの代わりに「真のパラメータ(λ, κ, μ̄, σ_μ, σ_X)」から
  シミュレーションでデータを作ることで、2_run_learning_model.py が
  それを正しく推定できているか(=手法が機能しているか)を
  後から答え合わせできるようにしている。
- 出力するCSVのスキーマは、実際に日本のデータ(EDINET/日経NEEDS等)を
  使うときにも同じ形に整形すればそのまま2番目のスクリプトに
  投入できるように設計している。

出力ファイル (4つ、/mnt/user-data/outputs/dummy_data/ に保存):
  1. firm_master.csv              : 企業マスタ(証券コード・業種・上場日)
  2. quarterly_fundamentals.csv    : 四半期の会計データ(ROEの元になるもの)
  3. monthly_panel.csv             : 月次パネル(リターン・実証で実際に使うテーブル)
  4. true_parameters_answer_key.csv: シミュレーションに使った「真のパラメータ」
                                      (答え合わせ用。2番目のスクリプトはこれを使わない)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------
# 0. 基本設定
# ---------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

N_FIRMS = 50                      # 企業数(ダミーなので少なめでOK)
N_QUARTERS = 80                   # 四半期数(20年分。再推定間隔8Qを何回か試すため)
DT_Q = 0.25                       # 1四半期 = 0.25年(論文のスケーリングに合わせる)
REPORT_LAG_MONTHS = 2             # 決算発表から情報が市場に反映されるまでのラグ(月)

START_QUARTER = pd.Period("2005Q1", freq="Q-MAR")  # 日本の会計年度(4-3月)を意識

INDUSTRIES = [
    "輸送用機器", "電気機器", "情報通信", "銀行業", "卸売業",
    "小売業", "建設業", "化学", "食料品", "医薬品",
]

OUT_DIR = Path("/mnt/user-data/outputs/dummy_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# 1. 企業マスタ & 真のパラメータを生成
# ---------------------------------------------------------------
def make_firm_master_and_params():
    codes = rng.choice(np.arange(1300, 9999), size=N_FIRMS, replace=False)
    codes = np.sort(codes)

    industries = rng.choice(INDUSTRIES, size=N_FIRMS)

    # 上場からの年数をばらつかせる(論文でいう firm age / young firms を模す)
    listing_years_ago = rng.uniform(1, 40, size=N_FIRMS)
    listing_date = pd.Timestamp("2024-12-31") - pd.to_timedelta(
        listing_years_ago * 365.25, unit="D"
    )

    # --- 論文3.1節の学習モデルの「真のパラメータ」 ---
    # λ_i : 実現ROEの平均回帰速度(四半期ベース)。論文Fig.1の分布(中央値0.37付近)を意識
    lambda_true = rng.uniform(0.15, 1.2, size=N_FIRMS)
    # κ_i : 潜在的な収益性ドライバーμの平均回帰速度。論文Fig.1(中央値0.23付近)を意識
    kappa_true = rng.uniform(0.05, 0.6, size=N_FIRMS)
    # μ̄_i : 長期的な収益性の水準(アンカー)
    mu_bar_true = np.clip(rng.normal(0.08, 0.05, size=N_FIRMS), -0.05, 0.30)
    # σ_μi : 潜在ドライバーのボラティリティ(四半期)
    sigma_mu_true = rng.uniform(0.005, 0.03, size=N_FIRMS)
    # σ_Xi : 実現ROEのボラティリティ(四半期)
    sigma_x_true = rng.uniform(0.01, 0.05, size=N_FIRMS)
    # β_i : リターン生成用の市場ベータ(モデル本体とは無関係。ダミーリターン用)
    beta_true = rng.uniform(0.7, 1.3, size=N_FIRMS)
    # γ_i : 「期待収益性改定」がリターンに与える真の感応度(検証用に埋め込む)
    # NOTE: 実際の株式市場のS/N比よりかなり強めに設定している(50社という少数の
    #       サンプルでも手法の効果が可視化できるようにするため)。小さくするほど
    #       現実的だが、その分H-Lスプレッドの検出は難しくなる。
    gamma_true = rng.uniform(2.0, 4.0, size=N_FIRMS)
    # retention ratio : 純利益のうち内部留保される割合(自己資本の推移に使う簡便化)
    retention_true = rng.uniform(0.4, 0.8, size=N_FIRMS)
    # 期初自己資本(円)。企業規模のばらつきを作るため対数正規分布
    init_book_equity = rng.lognormal(mean=np.log(3e10), sigma=1.0, size=N_FIRMS)
    # 期初時価総額(円)
    init_market_cap = init_book_equity * rng.uniform(0.5, 3.0, size=N_FIRMS)

    master = pd.DataFrame({
        "code": codes.astype(str),
        "company_name": [f"サンプル企業{i+1:03d}" for i in range(N_FIRMS)],
        "industry": industries,
        "listing_date": listing_date.date,
    })

    params = pd.DataFrame({
        "code": codes.astype(str),
        "lambda_true": lambda_true,
        "kappa_true": kappa_true,
        "mu_bar_true": mu_bar_true,
        "sigma_mu_true": sigma_mu_true,
        "sigma_x_true": sigma_x_true,
        "beta_true": beta_true,
        "gamma_true": gamma_true,
        "retention_true": retention_true,
        "init_book_equity": init_book_equity,
        "init_market_cap": init_market_cap,
    })
    return master, params


# ---------------------------------------------------------------
# 2. 四半期パネル(潜在プロセス・実現ROE・会計変数)をシミュレーション
#    論文 式(4)(5): dX = λ(μ-X)dt + σ_X dW,  dμ = κ(μ̄-μ)dt + σ_μ dW
# ---------------------------------------------------------------
def simulate_quarterly(master, params):
    quarters = pd.period_range(START_QUARTER, periods=N_QUARTERS, freq="Q-MAR")
    quarter_end_dates = quarters.asfreq("Q-MAR").to_timestamp(how="end").normalize()

    rows = []
    for i, code in enumerate(master["code"]):
        p = params.iloc[i]
        lam, kap = p["lambda_true"], p["kappa_true"]
        mu_bar, sig_mu, sig_x = p["mu_bar_true"], p["sigma_mu_true"], p["sigma_x_true"]

        mu = np.empty(N_QUARTERS)
        X = np.empty(N_QUARTERS)
        mu[0] = mu_bar + rng.normal(0, sig_mu)
        X[0] = mu_bar + rng.normal(0, sig_x)

        for t in range(1, N_QUARTERS):
            eps_mu = rng.normal()
            eps_x = rng.normal()
            mu[t] = mu[t - 1] + kap * (mu_bar - mu[t - 1]) * DT_Q + sig_mu * np.sqrt(DT_Q) * eps_mu
            X[t] = X[t - 1] + lam * (mu[t - 1] - X[t - 1]) * DT_Q + sig_x * np.sqrt(DT_Q) * eps_x

        # 会計上の整合性を持たせる: 純利益 = ROE(=X) * 前期末自己資本
        book_equity = np.empty(N_QUARTERS)
        net_income = np.empty(N_QUARTERS)
        be_prev = p["init_book_equity"]
        for t in range(N_QUARTERS):
            ni = X[t] * be_prev
            net_income[t] = ni
            book_equity[t] = be_prev + ni * p["retention_true"]
            be_prev = book_equity[t]
        book_equity_lag = np.concatenate([[p["init_book_equity"]], book_equity[:-1]])

        rows.append(pd.DataFrame({
            "code": code,
            "quarter_end": quarter_end_dates,
            "mu_true": mu,           # 真の潜在ドライバー(答え合わせ用)
            "roe": X,                # 実現ROE = X_it (観測可能)
            "net_income": net_income,
            "book_equity_lag": book_equity_lag,
        }))

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.merge(master[["code", "industry"]], on="code", how="left")

    # 業種ROE信号 s_it = 四半期ごとの業種内ROE中央値(論文4.2節: リアルタイムに計算)
    ind_roe = (
        panel.groupby(["industry", "quarter_end"])["roe"]
        .median()
        .rename("industry_roe")
        .reset_index()
    )
    panel = panel.merge(ind_roe, on=["industry", "quarter_end"], how="left")
    return panel


# ---------------------------------------------------------------
# 3. 月次パネル(リターン込み)を作成
#    - 会計情報は「決算発表ラグ」を考慮してcarry-forward
#    - 月次リターンには真のΔExpProfを埋め込み、H-Lスプレッドが
#      検出できるようにする(=手法の検証がしやすいダミー)
# ---------------------------------------------------------------
def prop_3_2_weights(lam, kap, horizon_years):
    """論文 Proposition 3.2 の a_i, b_i, c_i を計算(真のパラメータ版)。"""
    a = np.exp(-lam * horizon_years)
    if abs(lam - kap) < 1e-8:
        kap = kap + 1e-6  # 縮退回避
    b = (lam / (lam - kap)) * (np.exp(-kap * horizon_years) - np.exp(-lam * horizon_years))
    c = 1.0 - a - b
    return a, b, c


def simulate_monthly(master, params, qpanel):
    months = pd.date_range(
        qpanel["quarter_end"].min(), qpanel["quarter_end"].max() + pd.offsets.MonthEnd(REPORT_LAG_MONTHS + 3),
        freq="ME",
    )

    # 市場ファクター(ダミーの月次市場リターン)
    mkt_ret = pd.Series(rng.normal(0.005, 0.045, size=len(months)), index=months, name="mkt_ret")

    all_rows = []
    for i, code in enumerate(master["code"]):
        p = params.iloc[i]
        fp = qpanel[qpanel["code"] == code].sort_values("quarter_end").reset_index(drop=True)
        # 情報が市場に「使えるようになる」日付(決算発表ラグを加算)
        fp["available_from"] = fp["quarter_end"] + pd.offsets.MonthEnd(REPORT_LAG_MONTHS)

        # 真のΔExpProf(1年先、Proposition 3.2)を四半期ごとに計算(検証用のリターン生成に使用)
        a, b, c = prop_3_2_weights(p["lambda_true"], p["kappa_true"], horizon_years=1.0)
        fp["exp_prof_change_true"] = b * (fp["mu_true"] - fp["roe"]) + c * (p["mu_bar_true"] - fp["roe"])

        # 時価総額(月次、幾何ブラウン運動)
        n_m = len(months)
        me = np.empty(n_m)
        me[0] = p["init_market_cap"]
        for t in range(1, n_m):
            me[t] = me[t - 1] * np.exp(rng.normal(0.0, 0.08))

        # 各月に「その時点で最新公表済みの」会計情報を割り当て(as-of結合)
        fp_avail = fp.set_index("available_from")
        merged = pd.DataFrame(index=months)
        merged = merged.join(
            fp_avail[["quarter_end", "roe", "industry_roe", "exp_prof_change_true"]],
        )
        # as-of (直近発表済みの値をcarry-forward)
        merged = merged.reindex(months)
        fp_sorted = fp.sort_values("available_from")
        idx = np.searchsorted(fp_sorted["available_from"].values, months.values, side="right") - 1
        idx = np.clip(idx, 0, len(fp_sorted) - 1)
        valid = idx >= 0
        roe_cf = fp_sorted["roe"].values[idx]
        ind_roe_cf = fp_sorted["industry_roe"].values[idx]
        qend_cf = fp_sorted["quarter_end"].values[idx]
        exp_true_cf = fp_sorted["exp_prof_change_true"].values[idx]
        # 最初の発表がまだ無い月はNaNにする
        first_avail = fp_sorted["available_from"].values[0]
        no_data_mask = months.values < first_avail

        beta = p["beta_true"]
        gamma = p["gamma_true"]
        idio_vol = 0.05  # 月次アイディオシンクラティック・ボラ
        idio = rng.normal(0.0, idio_vol, size=n_m)
        ret = 0.001 + beta * mkt_ret.values + gamma * exp_true_cf + idio
        ret[no_data_mask] = np.nan

        df = pd.DataFrame({
            "code": code,
            "month_end": months,
            "quarter_end_used": qend_cf,
            "roe": roe_cf,
            "industry_roe": ind_roe_cf,
            "market_cap": me,
            "monthly_return": ret,
        })
        df.loc[no_data_mask, ["roe", "industry_roe", "quarter_end_used"]] = np.nan
        all_rows.append(df)

    monthly = pd.concat(all_rows, ignore_index=True)
    monthly = monthly.merge(master[["code", "industry", "listing_date"]], on="code", how="left")
    monthly = monthly.dropna(subset=["roe"]).reset_index(drop=True)
    return monthly


# ---------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------
def main():
    master, params = make_firm_master_and_params()
    qpanel = simulate_quarterly(master, params)
    monthly = simulate_monthly(master, params, qpanel)

    # 出力するカラムを整理(必要最小限+規模・業種などの補助情報)
    master_out = master.copy()

    qpanel_out = qpanel[[
        "code", "quarter_end", "industry", "roe", "industry_roe",
        "net_income", "book_equity_lag",
    ]].rename(columns={"roe": "roe_quarterly", "industry_roe": "industry_roe_quarterly"})

    monthly_out = monthly[[
        "code", "industry", "month_end", "quarter_end_used",
        "roe", "industry_roe", "market_cap", "monthly_return", "listing_date",
    ]]

    master_out.to_csv(OUT_DIR / "firm_master.csv", index=False)
    qpanel_out.to_csv(OUT_DIR / "quarterly_fundamentals.csv", index=False)
    monthly_out.to_csv(OUT_DIR / "monthly_panel.csv", index=False)
    params.to_csv(OUT_DIR / "true_parameters_answer_key.csv", index=False)

    print("=== ダミーデータ生成完了 ===")
    print(f"企業数: {N_FIRMS}, 四半期数: {N_QUARTERS}, 月次パネル行数: {len(monthly_out)}")
    print(f"保存先: {OUT_DIR}")
    for f in ["firm_master.csv", "quarterly_fundamentals.csv", "monthly_panel.csv",
              "true_parameters_answer_key.csv"]:
        print(f" - {f}")


if __name__ == "__main__":
    main()
