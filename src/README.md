# Hou and Sinagl (2025) 日本版の実装

"Learning about Expected Profitability" の期待収益性の改訂幅（ΔExpProfit）を、日本の四半期 ROE から作る。

## 実行方法

```
pip install -r requirements.txt
cd src
python run_pipeline.py                         # サンプルデータ（sample_data/sample_data.csv）
python run_pipeline.py --input <実データのCSV>  # ROE_Industry_query.sql の出力
python validate_sample.py                      # サンプルデータのみ：真のパラメータとの答え合わせ
```

出力は `output/hou_sinagl/` に保存される。サンプルデータ（約300社）で約3分、実データ（約3,800社）では1時間弱かかる見込み。

## ファイルと論文の対応

| ファイル | 処理 | 論文 |
|---|---|---|
| `config.py` | 設定値（論文にない値には【独自】と記載） | ― |
| `data_prep.py` | ROE の読み込み・除外・四半期ごとの winsorize | Appendix A 手順1・2 |
| `industry_signal.py` | 業種シグナル（公表済み・プラスの ROE の業種中央値、月次更新） | 4.2節、手順3 |
| `smm.py` | λ, κ, σ_μ の SMM 推定（拡大ウィンドウ、8四半期ごと） | 4.3節、手順4・5 |
| `learning_filter.py` | フィルター（μ̂, ν, サプライズ）と ExpProfit・ΔExpProfit の計算 | 3章、4.4節、手順6・7 |
| `monthly_panel.py` | 月次パネル（公表済みの最新値を引き継ぐ） | 4.4節、手順8 |
| `run_pipeline.py` | 一連の処理の実行と、要約統計量（Table I）・パラメータ分布（Figure 1）の出力 | ― |
| `validate_sample.py` | サンプルデータでの確認 | ― |

## 論文から変えた点・論文にない点

| 項目 | 本実装 | 理由 |
|---|---|---|
| 業種分類 | 東証33業種 | 日本データ（論文は FF30） |
| 業種中央値の最低企業数 | 3社（満たない月は前月の値を引き継ぐ） | 論文に記載なし |
| 初回推定の最低四半期数 | 20四半期 | 論文に記載なし |
| SMM の重み行列 | diag(1, 1, 1/(σ_X σ_s)²) | 論文に記載なし。共分散の単位をそろえるため |
| フィルターの式 | 理論式（式(7)(8)、Appendix B） | 4.4節の実装式と λ の位置が食い違うため |
| フィルターの数値計算 | 半陰的な差分（`NUMERICAL_SCHEME = "implicit"`） | 論文どおりの陽的 Euler 法は dt = 0.25 で発散する企業がある |
| 推定・フィルターの頻度 | 四半期ごとに計算し、月次では引き継ぐ | 4.4節（月次）と Appendix A（四半期）が食い違うため |
| 事後分散の初期値 | ν0 = σ_μ²/(2κ) | 論文に記載なし |
| 古い決算の除外 | 決算期末から6か月超は月次パネルで使わない | 論文に記載なし（HXZ の慣行） |
