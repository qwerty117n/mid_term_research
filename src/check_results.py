"""
推定結果の点検（run_pipeline.py の出力 CSV だけを読み込む。再推定はしない）

点検する内容
  1. カバレッジ      ：何社・何期分が推定できたか、月ごとの企業数
  2. 業種シグナル    ：水準、社数不足で前月の値を引き継いだ期間、季節性
  3. パラメータ      ：λ, κ, σ_μ の分布（論文 Table I と比較）、探索範囲の端への張り付き、収束、安定性
  4. 主要変数の分布  ：ΔExpProf, μ̂, ν, サプライズ（論文 Table I と比較）、極端な値
  5. 予測力          ：ΔExpProf が4四半期後の ROE の変化を当てているか（リターンを使わない検証）
  6. 先読みの有無    ：公表日より前の情報を使っていないか

使い方（src/ で実行）
  python check_results.py                          # config.OUTPUT_DIR の結果を点検
  python check_results.py --output <結果のフォルダ>

出力（<結果のフォルダ>/check/）
  check_report.txt            : 点検結果のまとめ（画面にも表示）
  coverage_by_month.csv       : 月ごとの企業数
  industry_signal_check.csv   : 業種ごとの業種シグナルの点検結果
  parameters_by_year.csv      : 推定年ごとのパラメータの平均・張り付き割合
  prediction_by_quintile.csv  : ΔExpProf の5分位ごとの、4四半期後の ROE の変化
  extreme_firms.csv           : ΔExpProf が極端な値をとる企業・四半期
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config

# 論文 Table I の値（比較用）
PAPER = {
    "lambda": (0.46, 0.39),
    "kappa": (0.23, 0.22),
    "sigma_mu": (0.06, None),
    "dW_x": (-0.152, 2.5),
    "dW_s": (-0.44, 0.84),
}

# 「要確認」とする目安【独自】
TH_BOUNDARY_SHARE = 0.05      # 探索範囲の端に張り付いた推定の割合
TH_CONVERGED_SHARE = 0.90     # 最適化が収束した割合
TH_CARRIED_SHARE = 0.20       # 業種シグナルを前月から引き継いだ月の割合（業種ごと）
TH_MIN_FIRMS_PER_MONTH = 100  # 月次パネルの企業数がこれ以上になった月を「分析可能な開始月」とする
TH_NONFINITE_SHARE = 0.001    # 無限大・欠損の割合

CODES = {"NRI_CODE": str, "証券コード": str}


class Report:
    """点検結果を画面とファイルに書き出す"""

    def __init__(self):
        self.lines = []
        self.n_warn = 0

    def section(self, title):
        self.add("")
        self.add("=" * 70)
        self.add(title)
        self.add("=" * 70)

    def add(self, text=""):
        print(text)
        self.lines.append(text)

    def check(self, ok, text):
        if not ok:
            self.n_warn += 1
        self.add(f"  [{'OK' if ok else '要確認'}] {text}")

    def table(self, df, float_fmt="{:.4f}"):
        for line in df.to_string(float_format=lambda v: float_fmt.format(v)).splitlines():
            self.add("    " + line)


def load(out: Path):
    params = pd.read_csv(out / "parameters.csv", encoding="utf-8-sig", dtype=CODES)
    quarterly = pd.read_csv(out / "quarterly_results.csv", encoding="utf-8-sig", dtype=CODES,
                            parse_dates=["当期決算年月日", "公表日"])
    panel = pd.read_csv(out / "monthly_panel.csv", encoding="utf-8-sig", dtype=CODES,
                        parse_dates=["month_end", "当期決算年月日", "公表日"])
    signal = pd.read_csv(out / "industry_signal_monthly.csv", encoding="utf-8-sig",
                         parse_dates=["month_end"])
    return params, quarterly, panel, signal


# ===== 1. カバレッジ =====
def check_coverage(r, params, quarterly, panel, out):
    r.section("1. カバレッジ（何社・何期分が推定できたか）")
    n_param_firms = params["NRI_CODE"].nunique()
    n_q_firms = quarterly["NRI_CODE"].nunique()
    r.add(f"  推定できた企業：{n_param_firms:,} 社（推定 {len(params):,} 件）")
    r.add(f"  四半期結果：{len(quarterly):,} 行、{n_q_firms:,} 社、"
          f"{quarterly['quarter'].min()} 〜 {quarterly['quarter'].max()}")
    r.add(f"  月次パネル：{len(panel):,} 行、{panel['NRI_CODE'].nunique():,} 社、"
          f"{panel['month_end'].min():%Y-%m} 〜 {panel['month_end'].max():%Y-%m}")

    by_month = panel.groupby("month_end").agg(
        企業数=("NRI_CODE", "nunique"),
        ΔExpProf欠損=("dExpProfit", lambda s: s.isna().sum()),
    )
    by_month.to_csv(out / "coverage_by_month.csv", encoding="utf-8-sig")

    enough = by_month[by_month["企業数"] >= TH_MIN_FIRMS_PER_MONTH]
    if len(enough):
        r.add(f"  企業数が {TH_MIN_FIRMS_PER_MONTH} 社以上になる最初の月：{enough.index.min():%Y-%m}"
              f"（以降 {len(enough)} か月）")
    r.check(len(enough) >= 60, f"企業数 {TH_MIN_FIRMS_PER_MONTH} 社以上の月が60か月（5年）以上ある")

    yearly = by_month["企業数"].groupby(by_month.index.year).agg(["min", "mean", "max"]).round(0)
    yearly.index.name = "年"
    r.add("  年ごとの月次企業数（最小・平均・最大）")
    r.table(yearly, "{:.0f}")

    # 推定できたのに四半期結果がない企業（フィルターで結果が出なかった）
    missing = set(params["NRI_CODE"]) - set(quarterly["NRI_CODE"])
    r.check(len(missing) == 0, f"推定できたが四半期結果がない企業：{len(missing)} 社")


# ===== 2. 業種シグナル =====
def check_industry_signal(r, signal, out):
    r.section("2. 業種シグナル（業種内のプラスの ROE の中央値）")
    s = signal["s"].dropna()
    r.add(f"  水準：中央値 {s.median():.4f}、1% {s.quantile(0.01):.4f}、99% {s.quantile(0.99):.4f}")
    r.check(0 < s.median() < 0.05, "業種シグナルの中央値が四半期 ROE としてもっともらしい（0〜5%）")

    # 社数不足で前月の値を引き継いだ月（n_firms が最低社数未満、または欠損）
    signal = signal.copy()
    signal["引き継ぎ"] = signal["n_firms"].fillna(0) < config.MIN_INDUSTRY_FIRMS
    by_ind = signal.groupby("業種コード").agg(
        月数=("s", "size"),
        業種シグナル欠損=("s", lambda v: v.isna().sum()),
        引き継ぎ割合=("引き継ぎ", "mean"),
        平均社数=("n_firms", "mean"),
        s中央値=("s", "median"),
    )
    by_ind.to_csv(out / "industry_signal_check.csv", encoding="utf-8-sig")
    bad = by_ind[by_ind["引き継ぎ割合"] > TH_CARRIED_SHARE]
    r.check(len(bad) == 0,
            f"前月の値を引き継いだ月の割合が {TH_CARRIED_SHARE:.0%} を超える業種：{len(bad)} 業種")
    if len(bad):
        r.table(bad[["引き継ぎ割合", "平均社数"]].sort_values("引き継ぎ割合", ascending=False).head(10))

    # 季節性：対象四半期（決算期末の月）ごとの業種シグナルの平均
    sig = signal.dropna(subset=["s", "anchor_quarter"]).copy()
    sig["対象四半期の月"] = pd.PeriodIndex(sig["anchor_quarter"], freq="Q").month
    season = sig.groupby("対象四半期の月")["s"].mean()
    r.add("  対象四半期（決算期末の月）ごとの業種シグナルの平均（季節性の確認）")
    r.table(season.to_frame("s平均"))
    spread = (season.max() - season.min()) / s.median()
    r.check(spread < 0.3, f"季節による差が中央値の30%未満（実際：{spread:.0%}）。"
                          "大きい場合は、本決算の特別損益や DTL の計上時期の影響を疑う")


# ===== 3. パラメータ =====
def check_parameters(r, params, out):
    r.section("3. パラメータ（λ, κ, σ_μ）")
    lo, hi = config.PARAM_LOWER, config.PARAM_UPPER
    p = params.copy()
    p["λ張り付き"] = (p["lambda"] <= lo * 1.5) | (p["lambda"] >= hi * 0.98)
    p["κ張り付き"] = (p["kappa"] <= lo * 1.5) | (p["kappa"] >= hi * 0.98)
    p["σμ張り付き"] = (p["sigma_mu"] >= config.SIGMA_MU_UPPER * 0.98) | (p["sigma_mu"] <= config.SIGMA_MU_LOWER * 1.5)

    latest = p.sort_values("推定四半期").groupby("NRI_CODE").tail(1)
    rows = []
    for v in ("lambda", "kappa", "sigma_mu"):
        rows.append({"変数": v, "平均（全推定）": p[v].mean(), "標準偏差": p[v].std(),
                     "中央値": p[v].median(), "平均（各社最新）": latest[v].mean(),
                     "論文の平均": PAPER[v][0], "論文の標準偏差": PAPER[v][1]})
    r.add("  分布（論文 Table I と比較）")
    r.table(pd.DataFrame(rows).set_index("変数"), "{:.3f}")

    for col, label in (("λ張り付き", "λ"), ("κ張り付き", "κ"), ("σμ張り付き", "σ_μ")):
        share = p[col].mean()
        r.check(share < TH_BOUNDARY_SHARE, f"{label} が探索範囲の端に張り付いた推定の割合：{share:.1%}")
    conv = p["smm_converged"].mean()
    r.check(conv >= TH_CONVERGED_SHARE, f"最適化が収束した割合：{conv:.1%}")

    obj = p["smm_obj"]
    r.add(f"  SMM の目的関数（モーメントの当てはまり。小さいほど良い）：中央値 {obj.median():.4f}、"
          f"90% {obj.quantile(0.9):.4f}、99% {obj.quantile(0.99):.4f}")

    # 安定性：同じ企業の、連続する推定時点間での変化
    p = p.sort_values(["NRI_CODE", "推定四半期"])
    for v, label in (("lambda", "λ"), ("kappa", "κ")):
        chg = p.groupby("NRI_CODE")[v].diff().abs().dropna()
        if len(chg):
            r.add(f"  {label} の推定時点間の変化（絶対値）：中央値 {chg.median():.3f}、90% {chg.quantile(0.9):.3f}")
    chg_l = p.groupby("NRI_CODE")["lambda"].diff().abs().dropna()
    if len(chg_l):
        r.check(chg_l.median() < 0.5, "λ の推定時点間の変化が小さい（中央値 0.5 未満）")

    # 推定年ごと（ウィンドウが長くなるにつれて安定するか）
    p["推定年"] = p["推定四半期"].str[:4].astype(int)
    by_year = p.groupby("推定年").agg(
        推定件数=("lambda", "size"), 平均観測数=("window_obs", "mean"),
        λ平均=("lambda", "mean"), κ平均=("kappa", "mean"), σμ中央値=("sigma_mu", "median"),
        λ張り付き=("λ張り付き", "mean"), κ張り付き=("κ張り付き", "mean"), σμ張り付き=("σμ張り付き", "mean"),
        収束割合=("smm_converged", "mean"),
    )
    by_year.to_csv(out / "parameters_by_year.csv", encoding="utf-8-sig")
    r.add("  推定年ごとの推移")
    r.table(by_year, "{:.3f}")


# ===== 4. 主要変数の分布 =====
def check_distributions(r, quarterly, out):
    r.section("4. 主要変数の分布")
    cols = ["X", "s", "mu_hat", "nu", "dW_x", "dW_s", "ExpProfit", "dExpProfit",
            "dExpProfit_short", "dExpProfit_long"]
    q = quarterly[cols]
    nonfinite = (~np.isfinite(q)).mean()
    for c in cols:
        if nonfinite[c] > 0:
            r.check(nonfinite[c] < TH_NONFINITE_SHARE, f"{c} の無限大・欠損の割合：{nonfinite[c]:.2%}")
    r.check((nonfinite < TH_NONFINITE_SHARE).all(), "主要変数に無限大・欠損がほとんどない")

    desc = q.replace([np.inf, -np.inf], np.nan).describe(percentiles=[0.01, 0.5, 0.99]).T
    desc = desc[["count", "mean", "std", "1%", "50%", "99%"]]
    r.table(desc, "{:.4f}")

    for v in ("dW_x", "dW_s"):
        m, sd = quarterly[v].mean(), quarterly[v].std()
        r.add(f"  {v}：平均 {m:.3f}、標準偏差 {sd:.3f}（論文：平均 {PAPER[v][0]}、標準偏差 {PAPER[v][1]}）")

    # μ̂ が長期水準 μ̄ から大きく離れていないか（論文の定式化では μ̂ が0側に寄りやすい）
    gap = (quarterly["mu_hat"] - quarterly["mu_bar"]).median()
    r.add(f"  μ̂ − μ̄ の中央値：{gap:.4f}（論文でも μ̂ は μ̄ より低めに出る）")

    # ΔExpProf の短期成分・長期成分の合計の整合性
    resid = (quarterly["dExpProfit"] - quarterly["dExpProfit_short"] - quarterly["dExpProfit_long"]).abs().max()
    r.check(resid < 1e-8, f"ΔExpProf ＝ 短期成分 ＋ 長期成分 が成り立つ（最大誤差 {resid:.1e}）")

    # 極端な値
    d = quarterly["dExpProfit"]
    z = (d - d.median()) / d.std()
    extreme = quarterly.loc[z.abs() > 5, ["NRI_CODE", "quarter", "業種コード", "X", "s", "mu_hat",
                                          "dExpProfit", "lambda", "kappa", "sigma_mu"]]
    extreme.sort_values("dExpProfit").to_csv(out / "extreme_firms.csv", index=False, encoding="utf-8-sig")
    r.check(len(extreme) / len(quarterly) < 0.005,
            f"ΔExpProf が平均から標準偏差の5倍以上離れた行：{len(extreme):,} 行"
            f"（{extreme['NRI_CODE'].nunique()} 社、extreme_firms.csv）")


# ===== 5. 予測力 =====
def check_prediction(r, quarterly, out):
    r.section("5. 予測力（ΔExpProf は4四半期後の ROE の変化を当てているか）")
    q = quarterly.copy()
    q["qp"] = pd.PeriodIndex(q["quarter"], freq="Q")
    fut = q[["NRI_CODE", "qp", "X"]].rename(columns={"X": "X_4q後"})
    fut["qp"] = fut["qp"] - 4
    q = q.merge(fut, on=["NRI_CODE", "qp"], how="inner")
    q = q.replace([np.inf, -np.inf], np.nan).dropna(subset=["ExpProfit", "dExpProfit", "X_4q後"])
    q["実際の変化"] = q["X_4q後"] - q["X"]
    r.add(f"  対象：{len(q):,} 企業・四半期")

    def rmse(pred):
        return float(np.sqrt(np.mean((q["X_4q後"] - pred) ** 2)))

    r_exp, r_rw, r_mu = rmse(q["ExpProfit"]), rmse(q["X"]), rmse(q["mu_bar"])
    r.add(f"  4四半期後の ROE の予測誤差（RMSE）：ExpProf {r_exp:.5f} ／ 現在の ROE {r_rw:.5f} ／ 長期平均 μ̄ {r_mu:.5f}")
    r.check(r_exp < r_rw, "期待 ROE（ExpProf）が、現在の ROE がそのまま続くという予測より正確")

    corr = q["実際の変化"].corr(q["dExpProfit"])
    r.check(corr > 0, f"ΔExpProf と実際の ROE の変化の相関がプラス（{corr:.3f}）")

    # 四半期ごとの順位相関の平均（Fama-MacBeth 型。時点をまたいだ水準差の影響を除く）
    ic = q.groupby("qp").apply(
        lambda g: g["dExpProfit"].rank().corr(g["実際の変化"].rank()) if len(g) >= 20 else np.nan
    ).dropna()
    t = ic.mean() / (ic.std() / np.sqrt(len(ic))) if len(ic) > 1 else np.nan
    r.check(ic.mean() > 0 and t > 2,
            f"四半期ごとの順位相関の平均：{ic.mean():.3f}（t値 {t:.1f}、{len(ic)} 四半期）")
    for comp, label in (("dExpProfit_short", "短期成分"), ("dExpProfit_long", "長期成分")):
        r.add(f"    {label}と実際の変化の相関：{q['実際の変化'].corr(q[comp]):.3f}")

    # 5分位ごとの実際の変化（単調に増えるか）
    q["5分位"] = q.groupby("qp")["dExpProfit"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1 if len(s) >= 5 else np.nan)
    by_q = q.groupby("5分位").agg(件数=("実際の変化", "size"), ΔExpProf平均=("dExpProfit", "mean"),
                                 実際の変化平均=("実際の変化", "mean"))
    by_q.to_csv(out / "prediction_by_quintile.csv", encoding="utf-8-sig")
    r.add("  ΔExpProf の5分位（各四半期内で分類）ごとの、4四半期後の ROE の変化")
    r.table(by_q, "{:.5f}")
    r.check(by_q["実際の変化平均"].is_monotonic_increasing, "5分位が上がるほど、実際の ROE の変化も大きい（単調）")


# ===== 6. 先読みの有無 =====
def check_lookahead(r, quarterly, panel):
    r.section("6. 先読みの有無（公表前の情報を使っていないか）")
    bad_pub = (quarterly["公表日"] < quarterly["当期決算年月日"]).sum()
    r.check(bad_pub == 0, f"公表日が決算期末日より前の行：{bad_pub}")
    early = (panel["month_end"] < panel["公表日"]).sum()
    r.check(early == 0, f"月次パネルで、公表日より前の月末に値が使われている行：{early}")
    stale = (panel["決算からの経過月数"] > config.MAX_STALE_MONTHS).sum()
    r.check(stale == 0, f"決算期末から {config.MAX_STALE_MONTHS} か月を超えた古い値：{stale}")
    est_q = pd.PeriodIndex(quarterly["推定四半期"], freq="Q")
    cur_q = pd.PeriodIndex(quarterly["quarter"], freq="Q")
    future_param = (est_q > cur_q).sum()
    r.check(future_param == 0, f"その四半期より後に推定したパラメータを使っている行：{future_param}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=config.OUTPUT_DIR, help="run_pipeline.py の出力フォルダ")
    args = ap.parse_args()
    out = args.output / "check"
    out.mkdir(parents=True, exist_ok=True)

    params, quarterly, panel, signal = load(args.output)
    r = Report()
    r.add(f"点検対象：{args.output}")

    check_coverage(r, params, quarterly, panel, out)
    check_industry_signal(r, signal, out)
    check_parameters(r, params, out)
    check_distributions(r, quarterly, out)
    check_prediction(r, quarterly, out)
    check_lookahead(r, quarterly, panel)

    r.section(f"まとめ：要確認 {r.n_warn} 件")
    r.add("  「要確認」の項目は、上の各節の説明と出力 CSV（check/ フォルダ）で詳細を確認してください。")
    (out / "check_report.txt").write_text("\n".join(r.lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
