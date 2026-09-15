"""
ファクターリターンのNewey-West (HAC) 検定
=========================================
CSVフォーマット想定:
    Date, Mkt-RF, SMB, HML, RMW, CMA
    Date は yyyymm 形式 (例: 202107)
    各ファクターリターンは % 表記 (例: 1.23 は 1.23%)

各ファクターについて、以下の回帰を行う:
    r_t = mu + epsilon_t
と定数項のみで回帰し、Newey-West HAC標準誤差を用いて
mu (=平均リターン) が統計的に有意にゼロと異なるかを検定する。
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

# ----------------------------------------
# 1. データ読み込み
# ----------------------------------------
# 実際のパスに置き換えてください
CSV_PATH = "factors.csv"

df = pd.read_csv(CSV_PATH)

# Date列 (yyyymm) をdatetime化しておくと、後々の作業がしやすい
df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m")
df = df.sort_values("Date").reset_index(drop=True)

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

# ----------------------------------------
# 2. Newey-West ラグ数の設定
# ----------------------------------------
# 月次データでは、経験則として 12 (1年分の自己相関) を使うことが多い。
# Newey-West (1994) の目安式を使いたい場合は以下のようにする:
#   L = int(4 * (T/100) ** (2/9))
T = len(df)
MAXLAGS = 12
# MAXLAGS = int(4 * (T / 100) ** (2 / 9))  # 代替案: データ駆動的に決める場合

# ----------------------------------------
# 3. 各ファクターについて HAC 検定を実行
# ----------------------------------------
results = []

for factor in FACTOR_COLS:
    r = df[factor].dropna().values
    n_obs = len(r)

    # 定数項のみの回帰: r_t = mu + epsilon_t
    X = np.ones((n_obs, 1))
    model = sm.OLS(r, X).fit(cov_type="HAC", cov_kwds={"maxlags": MAXLAGS})

    mu_hat = model.params[0]          # 平均リターン (=mu の推定値)
    se_hac = model.bse[0]             # HAC標準誤差
    t_stat = model.tvalues[0]         # HAC t統計量
    p_value = model.pvalues[0]        # 両側検定のp値

    # 参考: 通常のOLS標準誤差(比較用)
    model_ols = sm.OLS(r, X).fit()
    se_ols = model_ols.bse[0]
    t_ols = model_ols.tvalues[0]

    results.append({
        "Factor": factor,
        "N": n_obs,
        "Mean (%)": round(mu_hat, 4),
        "HAC SE": round(se_hac, 4),
        "HAC t-stat": round(t_stat, 3),
        "HAC p-value": round(p_value, 4),
        "OLS t-stat (参考)": round(t_ols, 3),
        "Significant (5%)": "Yes" if p_value < 0.05 else "No",
    })

results_df = pd.DataFrame(results)

# ----------------------------------------
# 4. 結果表示
# ----------------------------------------
pd.set_option("display.width", 120)
print(f"サンプル期間: {df['Date'].min().strftime('%Y-%m')} 〜 {df['Date'].max().strftime('%Y-%m')}")
print(f"観測数: {T} ヶ月, Newey-West ラグ数: {MAXLAGS}\n")
print(results_df.to_string(index=False))

# CSVとして保存したい場合
# results_df.to_csv("newey_west_results.csv", index=False)
