"""
手順8（前半）：月次パネルの作成（論文 4.4節、Appendix A 手順8）

各月末 m について、企業ごとに「m までに公表された最新の四半期」の結果を引き継ぐ。
  ・ROE（X）、期待収益性（ExpProfit）、改訂幅（ΔExpProfit）などは、新しい決算が公表されるまで引き継ぐ
  ・業種シグナルは、月末ごとに更新された値（s_month）も付ける
  ・決算期末から MAX_STALE_MONTHS か月を超えた古い決算は使わない【独自】

リターンと結合するときは、月末 m の値で m の翌月のリターンを予測する（先読みを避けるため）。
"""

import pandas as pd

import config


def build_monthly_panel(quarterly: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    q = quarterly.copy()
    q["公表月末"] = (q["公表日"] + pd.offsets.MonthEnd(0)).dt.normalize()
    # 同じ月に2つの四半期が公表された場合は、決算期末が新しい方を使う
    q = (q.sort_values(["NRI_CODE", "公表月末", "当期決算年月日"])
           .drop_duplicates(["NRI_CODE", "公表月末"], keep="last"))

    months = pd.date_range(q["公表月末"].min(), q["公表月末"].max(), freq="ME")
    panel = []
    for firm, g in q.groupby("NRI_CODE"):
        g = g.set_index("公表月末").sort_index()
        span = months[(months >= g.index.min())]
        f = g.reindex(span, method="ffill")
        f["NRI_CODE"] = firm
        panel.append(f)
    panel = pd.concat(panel).rename_axis("month_end").reset_index()

    # 古すぎる決算を除外
    age = (panel["month_end"].dt.to_period("M") - panel["当期決算年月日"].dt.to_period("M")).apply(lambda d: d.n)
    panel = panel[age <= config.MAX_STALE_MONTHS].copy()
    panel["決算からの経過月数"] = age[panel.index]

    # 月末時点の業種シグナル（月次で更新された値）
    panel = panel.merge(
        signal[["業種コード", "month_end", "s"]].rename(columns={"s": "s_month"}),
        on=["業種コード", "month_end"], how="left",
    )
    cols = ["month_end", "NRI_CODE", "証券コード", "業種コード", "quarter", "当期決算年月日", "公表日",
            "決算からの経過月数", "X", "s", "s_month", "mu_hat", "nu", "dW_x", "dW_s", "nu_dW_x", "nu_dW_s",
            "ExpProfit", "dExpProfit", "dExpProfit_short", "dExpProfit_long", "lambda", "kappa"]
    return panel[cols].sort_values(["month_end", "NRI_CODE"]).reset_index(drop=True)
