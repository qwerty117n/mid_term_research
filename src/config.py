"""
Hou and Sinagl (2025) "Learning about Expected Profitability" 日本版実装の設定値。

論文に記載がある値は論文に合わせ、記載がない値（本実装で決めたもの）には【独自】と付けている。
"""

from pathlib import Path

# ===== パス =====
ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "sample_data" / "sample_data.csv"   # ROE_Industry_query.sql の出力
OUTPUT_DIR = ROOT / "output" / "hou_sinagl"

# ===== 時間の単位 =====
DT = 0.25            # 1四半期 = 0.25年（論文 4.4節：dt = 1/4）
HORIZON = 1.0        # 期待収益性の予測期間 τ − t = 1年（論文 5.2節）

# ===== ROE の前処理 =====
WINSOR_LOWER = 0.01  # 四半期ごとに 1%・99% で winsorize（論文 Appendix A 手順2）
WINSOR_UPPER = 0.99

# ===== 業種シグナル =====
# 業種内の「プラスのROE」の中央値（論文 Appendix A 手順3）
MIN_INDUSTRY_FIRMS = 3   # 【独自】中央値の計算に必要な最低企業数。満たない月は前月の値を引き継ぐ
                         #   東証33業種には企業数の少ない業種（鉱業、空運業など）があるため3社とした

# ===== パラメータ推定（SMM） =====
MIN_WINDOW_QUARTERS = 20  # 【独自】初回推定に必要な最低四半期数（論文に記載なし）
REESTIMATE_EVERY = 8      # 8四半期（2年）ごとに再推定（論文 4.3節）
PARAM_LOWER = 1e-3        # λ, κ の探索範囲（論文：0未満・5超を除外）
PARAM_UPPER = 5.0
N_SIM_PATHS = 200         # 【独自】シミュレーションの経路数
SIM_BURN_IN = 20          # 【独自】シミュレーションの助走期間（四半期）
SMM_MAXITER = 400         # 【独自】Nelder-Mead の最大反復回数
RANDOM_SEED = 12345

# ===== フィルター =====
# 業種サプライズの定義
#   "paper" : dW̃s = (Δs − μ̂·dt) / σs   論文の式(10)・4.4節どおり（論文 Table I の dW̃s 平均が負なのと整合）
#   "level" : dW̃s = (s − μ̂) / σs        【独自】式(6) s = μ + ε を水準の観測とみなす代替定式化
INDUSTRY_INNOVATION = "paper"

# フィルターの数値計算の方法
#   "implicit" : 【独自】半陰的な差分。σ_s が小さい企業でも発散しない（既定）
#   "explicit" : 論文 4.4節どおりの陽的 Euler 法。dt = 0.25 では発散する企業がある
NUMERICAL_SCHEME = "implicit"

# ===== 月次パネル =====
MAX_STALE_MONTHS = 6  # 【独自】決算期末から6か月を超えた古いROEは使わない（HXZ 2015 の慣行）

# ===== 並列計算 =====
N_JOBS = None  # None なら CPU コア数
