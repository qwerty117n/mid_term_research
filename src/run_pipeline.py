"""
Hou and Sinagl (2025) 日本版：期待収益性の改訂幅（ΔExpProfit）を作るまでの一連の処理を実行する。

使い方（src/ で実行）
  python run_pipeline.py                          # config.INPUT_CSV（サンプルデータ）を使う
  python run_pipeline.py --input ../data/roe.csv  # 実データを使う
  python run_pipeline.py --max-firms 50           # 動作確認用に企業数を絞る

出力（output/hou_sinagl/）
  industry_signal_monthly.csv : 業種 × 月末 の業種シグナル
  parameters.csv              : 企業 × 推定時点 のパラメータ推定結果（λ, κ, σ_μ など）
  quarterly_results.csv       : 企業 × 四半期 の予想 μ̂、事後分散 ν、サプライズ、ΔExpProfit
  monthly_panel.csv           : 企業 × 月末 のパネル（リターンと結合して検証に使う）
  summary_statistics.csv      : 主要変数の要約統計量（論文 Table I に対応）
  parameter_distribution.csv  : λ, κ の分布（論文 Figure 1 に対応）
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data_prep import load_quarterly
from industry_signal import attach_quarterly_signal, compute_monthly_signal
from learning_filter import process_firm
from monthly_panel import build_monthly_panel

SUMMARY_VARS = {
    "lambda": "λ：ROE の平均回帰の速さ",
    "kappa": "κ：潜在的な源泉 μ の平均回帰の速さ",
    "sigma_mu": "σ_μ：μ のボラティリティ",
    "mu_bar": "μ̄：長期水準",
    "X": "ROE（四半期、winsorize 後）",
    "s": "業種シグナル",
    "mu_hat": "μ̂：予想",
    "nu": "ν：事後分散",
    "dW_x": "dW̃X：企業サプライズ",
    "dW_s": "dW̃s：業種サプライズ",
    "ExpProfit": "1年先の期待 ROE",
    "dExpProfit": "ΔExpProfit：期待収益性の改訂幅",
    "dExpProfit_short": "ΔExpProfit の短期成分",
    "dExpProfit_long": "ΔExpProfit の長期成分",
}


def summarize(quarterly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v, label in SUMMARY_VARS.items():
        a = quarterly[v].dropna()
        rows.append({
            "変数": v, "説明": label, "N": len(a), "平均": a.mean(), "標準偏差": a.std(),
            "1%": a.quantile(0.01), "25%": a.quantile(0.25), "中央値": a.median(),
            "75%": a.quantile(0.75), "99%": a.quantile(0.99),
        })
    return pd.DataFrame(rows)


def parameter_distribution(params: pd.DataFrame) -> pd.DataFrame:
    """企業ごとの最新推定値の λ, κ のヒストグラム（論文 Figure 1）"""
    latest = params.sort_values("推定四半期").groupby("NRI_CODE").tail(1)
    bins = np.linspace(0, config.PARAM_UPPER, 26)
    out = pd.DataFrame({"階級下限": bins[:-1], "階級上限": bins[1:]})
    for v in ("lambda", "kappa"):
        out[f"{v}_企業数"] = np.histogram(latest[v], bins=bins)[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=config.INPUT_CSV)
    ap.add_argument("--output", type=Path, default=config.OUTPUT_DIR)
    ap.add_argument("--max-firms", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=config.N_JOBS)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 手順1・2：ROE の前処理
    df = load_quarterly(args.input)

    # 手順3：業種シグナル（業種シグナルは全企業で計算してから、対象企業を絞る）
    signal = compute_monthly_signal(df)
    signal.to_csv(args.output / "industry_signal_monthly.csv", index=False, encoding="utf-8-sig")
    df = attach_quarterly_signal(df, signal)
    if args.max_firms:
        keep = df["NRI_CODE"].drop_duplicates().iloc[: args.max_firms]
        df = df[df["NRI_CODE"].isin(keep)]

    # 手順4〜7：企業ごとの推定とフィルター
    tasks = list(df.groupby("NRI_CODE"))
    print(f"[run] {len(tasks):,} 社の推定を開始")
    params, rows = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, (p, r) in enumerate(ex.map(process_firm, tasks, chunksize=4), 1):
            params += p
            rows += r
            if i % 50 == 0:
                print(f"[run] {i:,}/{len(tasks):,} 社 完了（{time.time() - t_start:.0f} 秒）")

    if not rows:
        print(f"[run] 推定できた企業がありません。各社の観測数（{config.MIN_WINDOW_QUARTERS} 四半期以上）"
              f"と、業種シグナル（1業種 {config.MIN_INDUSTRY_FIRMS} 社以上）を確認してください。")
        return
    params = pd.DataFrame(params)
    quarterly = pd.DataFrame(rows)
    params.to_csv(args.output / "parameters.csv", index=False, encoding="utf-8-sig")
    quarterly.to_csv(args.output / "quarterly_results.csv", index=False, encoding="utf-8-sig")

    # 手順8（前半）：月次パネル
    panel = build_monthly_panel(quarterly, signal)
    panel.to_csv(args.output / "monthly_panel.csv", index=False, encoding="utf-8-sig")

    summarize(quarterly).to_csv(args.output / "summary_statistics.csv", index=False, encoding="utf-8-sig")
    parameter_distribution(params).to_csv(args.output / "parameter_distribution.csv", index=False, encoding="utf-8-sig")

    print(f"[run] 推定 {len(params):,} 件, 四半期結果 {len(quarterly):,} 行, 月次パネル {len(panel):,} 行")
    print(f"[run] 出力先: {args.output}（{time.time() - t_start:.0f} 秒）")


if __name__ == "__main__":
    main()
