"""
サンプルデータでの実装の確認（実データには使わない）。

1. パラメータの答え合わせ：推定した λ, κ を、サンプルデータ生成時の真の値（sample_true_params.csv）と比べる
2. 予測力の確認：1年先の期待 ROE（ExpProfit）が、実際の4四半期後の ROE をどれだけ当てるかを、
   素朴な予測（現在の ROE がそのまま続く／長期平均 μ̄ に戻る）と比べる

使い方（src/ で実行、run_pipeline.py の後）
  python validate_sample.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config

TRUE_PARAMS = config.ROOT / "sample_data" / "sample_true_params.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=config.OUTPUT_DIR)
    args = ap.parse_args()

    params = pd.read_csv(args.output / "parameters.csv", encoding="utf-8-sig")
    quarterly = pd.read_csv(args.output / "quarterly_results.csv", encoding="utf-8-sig")
    true = pd.read_csv(TRUE_PARAMS, encoding="utf-8-sig")
    lines = []

    # ===== 1. パラメータの答え合わせ（企業ごとの最新の推定値） =====
    latest = params.sort_values("推定四半期").groupby("NRI_CODE").tail(1)
    m = latest.merge(true, on="NRI_CODE", suffixes=("_est", "_true"))
    lines.append(f"■ パラメータの答え合わせ（{len(m)} 社、各社の最新の推定値）")
    lines.append(f"{'':10s}{'真の平均':>10s}{'推定の平均':>10s}{'順位相関':>10s}{'誤差の中央値':>12s}")
    for est, tru in [("lambda", "lambda"), ("kappa", "kappa"), ("sigma_mu", "sigma_mu")]:
        e = m[f"{est}_est"] if f"{est}_est" in m else m[est]
        t = m[f"{tru}_true"] if f"{tru}_true" in m else m[tru]
        rho = e.rank().corr(t.rank())
        lines.append(f"{est:10s}{t.mean():10.3f}{e.mean():10.3f}{rho:10.3f}{(e - t).abs().median():12.3f}")
    lines.append(f"境界（{config.PARAM_UPPER}）付近に張り付いた推定の割合："
                 f"λ {np.mean(latest['lambda'] > config.PARAM_UPPER * 0.95):.1%}、"
                 f"κ {np.mean(latest['kappa'] > config.PARAM_UPPER * 0.95):.1%}")

    # ===== 2. 予測力の確認 =====
    q = quarterly.copy()
    q["qp"] = pd.PeriodIndex(q["quarter"], freq="Q")
    fut = q[["NRI_CODE", "qp", "X"]].copy()
    fut["qp"] = fut["qp"] - 4
    q = q.merge(fut.rename(columns={"X": "X_4q後"}), on=["NRI_CODE", "qp"], how="inner")
    q = q.replace([np.inf, -np.inf], np.nan).dropna(subset=["ExpProfit", "X_4q後"])

    def rmse(pred):
        return np.sqrt(np.mean((q["X_4q後"] - pred) ** 2))

    lines.append("")
    lines.append(f"■ 4四半期後の ROE の予測（{len(q):,} 企業・四半期）")
    lines.append(f"  RMSE：ExpProfit {rmse(q['ExpProfit']):.5f} ／ 現在の ROE {rmse(q['X']):.5f} ／ 長期平均 μ̄ {rmse(q['mu_bar']):.5f}")
    chg = q["X_4q後"] - q["X"]
    lines.append(f"  実際の変化（X_4q後 − X）と ΔExpProfit の相関：{chg.corr(q['dExpProfit']):.3f}")
    lines.append(f"    短期成分との相関 {chg.corr(q['dExpProfit_short']):.3f}、長期成分との相関 {chg.corr(q['dExpProfit_long']):.3f}")

    text = "\n".join(lines)
    print(text)
    (args.output / "validation_sample.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
