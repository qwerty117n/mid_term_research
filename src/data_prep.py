"""
手順1・2：ROE データの読み込みと前処理（論文 Appendix A 手順1・2）

・ROE_Industry_query.sql の出力（またはサンプルデータ）を読み込む
・分析に使えない行を除外（ROE・公表日・業種が欠損、分母がゼロ以下）
・決算期末日を暦四半期に対応させ、四半期ごとに ROE を 1%・99% で winsorize する
"""

import pandas as pd

import config

USE_COLUMNS = [
    "NRI_CODE", "証券コード", "業種コード", "業種名", "当期決算年月日", "公表日",
    "当期会計基準", "profit_t", "BEQ_t_minus_1", "ROE",
]


def load_quarterly(path=config.INPUT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=USE_COLUMNS)
    df["当期決算年月日"] = pd.to_datetime(df["当期決算年月日"])
    df["公表日"] = pd.to_datetime(df["公表日"])
    n0 = len(df)

    # 分析に使えない行を除外
    df = df.dropna(subset=["ROE", "公表日", "業種コード"])
    df = df[df["BEQ_t_minus_1"] > 0]                   # 分母がゼロ以下（論文：nonpositive denominators を除外）
    df = df[df["公表日"] >= df["当期決算年月日"]]
    df["業種コード"] = df["業種コード"].astype(int)

    # 暦四半期（決算期末日を含む四半期）。2月決算などの企業も暦四半期にそろえる
    df["quarter"] = df["当期決算年月日"].dt.to_period("Q")

    # 決算期変更などで同じ暦四半期に2行ある場合は、決算日が新しい方を残す
    df = (df.sort_values(["NRI_CODE", "quarter", "当期決算年月日"])
            .drop_duplicates(["NRI_CODE", "quarter"], keep="last"))

    # 四半期ごとに 1%・99% で winsorize（論文 Appendix A 手順2）
    lo = df.groupby("quarter")["ROE"].transform(lambda s: s.quantile(config.WINSOR_LOWER))
    hi = df.groupby("quarter")["ROE"].transform(lambda s: s.quantile(config.WINSOR_UPPER))
    df["X"] = df["ROE"].clip(lo, hi)

    print(f"[data_prep] 読み込み {n0:,} 行 -> 分析対象 {len(df):,} 行, {df['NRI_CODE'].nunique():,} 社")
    return df.reset_index(drop=True)
