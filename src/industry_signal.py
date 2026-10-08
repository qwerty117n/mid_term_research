"""
手順3：業種シグナル s の計算（論文 4.2節、Appendix A 手順3）

各月末 m について、業種ごとに
  ・直近に終わった暦四半期（m の属する四半期の1つ前）を対象とし、
  ・その四半期の ROE のうち、m までに公表済み（公表日 <= m）で、かつプラスのものの中央値
を業種シグナルとする。公表済み企業が MIN_INDUSTRY_FIRMS 社未満の月は、前月の値を引き継ぐ【独自】。

例：4月末・5月末・6月末は、3月末の四半期を対象にする。
    3月決算企業の本決算は5月に出そろうため、4月末時点では12月決算・2月決算などの企業だけで計算され、
    社数が足りなければ前月（12月末の四半期を対象とした値）が引き継がれる。
"""

import numpy as np
import pandas as pd

import config


def compute_monthly_signal(df: pd.DataFrame) -> pd.DataFrame:
    """業種 × 月末 の業種シグナル表を返す（列：業種コード, month_end, s, n_firms, anchor_quarter）"""
    records = []
    for (ind, q), g in df.groupby(["業種コード", "quarter"]):
        pos = g[g["X"] > 0]
        pub = pos["公表日"].to_numpy()
        x = pos["X"].to_numpy()
        # この四半期を対象とする月末：次の四半期の3つの月末
        for k in (1, 2, 3):
            m = (q.end_time + pd.offsets.MonthEnd(k)).normalize()
            mask = pub <= np.datetime64(m)
            n = int(mask.sum())
            s = float(np.median(x[mask])) if n >= config.MIN_INDUSTRY_FIRMS else np.nan
            records.append((ind, m, s, n, q))

    sig = pd.DataFrame(records, columns=["業種コード", "month_end", "s", "n_firms", "anchor_quarter"])

    # 全月末の格子を作り、社数不足の月は前月の値を引き継ぐ
    months = pd.date_range(sig["month_end"].min(), sig["month_end"].max(), freq="ME")
    grid = pd.MultiIndex.from_product([sig["業種コード"].unique(), months], names=["業種コード", "month_end"])
    sig = sig.set_index(["業種コード", "month_end"]).reindex(grid).sort_index()
    sig["s"] = sig.groupby(level="業種コード")["s"].ffill()
    return sig.reset_index()


def attach_quarterly_signal(df: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """
    各企業・四半期に、その ROE が公表された月の月末時点の業種シグナル s を付ける。
    （ROE が観測可能になった時点で、投資家が同時に観測できる業種シグナル）
    """
    df = df.copy()
    df["obs_month_end"] = (df["公表日"] + pd.offsets.MonthEnd(0)).dt.normalize()
    df = df.merge(
        sig[["業種コード", "month_end", "s"]],
        left_on=["業種コード", "obs_month_end"], right_on=["業種コード", "month_end"], how="left",
    ).drop(columns="month_end")
    n_missing = df["s"].isna().sum()
    print(f"[industry_signal] 業種シグナルが付かなかった行: {n_missing:,}（分析から除外）")
    return df.dropna(subset=["s"]).reset_index(drop=True)
