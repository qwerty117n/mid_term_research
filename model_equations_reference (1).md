# Hou & Sinagl (2025) モデル式リファレンス

実装のために必要な数式を、連続時間モデル → 離散時間実装式の順にまとめる。

---

## 0. 記法

| 記号 | 意味 |
|---|---|
| $i$ | 企業 |
| $t$ | 時点(四半期または月) |
| $X_{it}$ | 企業 $i$ の実現収益性(ROE) |
| $s_{it}$ | 業界内ROE中央値(観測可能な業界シグナル) |
| $\mu_{it}$ | 企業 $i$ の「本当の実力」(潜在的な収益性ドライバー、観測不可) |
| $\hat\mu_{it}$ | $\mu_{it}$ についての投資家の信念(事後推定値) |
| $\nu_{it}$ | 信念の不確実性(事後分散) |
| $\lambda_i$ | $X_{it}$ が $\mu_{it}$ に近づく速さ(平均回帰速度) |
| $\kappa_i$ | $\mu_{it}$ が長期水準 $\bar\mu_i$ に近づく速さ(平均回帰速度) |
| $\bar\mu_i$ | 企業 $i$ の長期的な収益性の水準(アンカー) |
| $\sigma_{iX},\ \sigma_{is},\ \sigma_{\mu i}$ | それぞれ $X,\ s,\ \mu$ のランダムな揺れの大きさ(ボラティリティ) |
| $dt$ | ごく短い時間の刻み幅(実装では四半期の $1/4$ = 1ヶ月相当) |
| $dW$ | その時間の間に起きる、予測不可能なランダムショック |

---

## 1. モデルの骨格(連続時間・SDE)

### (a) 潜在的な収益性ドライバー $\mu_{it}$ の動き方

$$
d\mu_{it} = \kappa_i(\bar\mu_i - \mu_{it})\,dt + \sigma_{\mu i}\,dW_{it}^{\mu}
$$

$\mu_{it}$ は長期水準 $\bar\mu_i$ に向かって速度 $\kappa_i$ で引き戻される(オルンシュタイン・ウーレンベック過程)。

### (b) 実現収益性 $X_{it}$ の動き方(第1の観測)

$$
dX_{it} = \lambda_i(\mu_{it} - X_{it})\,dt + \sigma_{iX}\,dW_{it}^{X}
$$

観測される会計上のROEは、本当の実力 $\mu_{it}$ に向かって速度 $\lambda_i$ で追いつく。

### (c) 業界シグナル $s_{it}$(第2の観測)

$$
s_{it} = \mu_{it} + \epsilon_{it}^{s}, \qquad \epsilon_{it}^{s} \sim N(0, \sigma_{is}^2)
$$

業界中央値は、$\mu_{it}$ をノイズ付きで直接観測したものとみなす(こちらは動的方程式ではなく代数的な関係)。

---

## 2. フィルター更新式(Proposition 3.1)— カルマン・ブーシーフィルター

### 信念の更新

$$
d\hat\mu_{it} = \kappa_i(\bar\mu_i - \hat\mu_{it})\,dt + \frac{\lambda_i \nu_{it}}{\sigma_{iX}}\,d\tilde W_{it}^{X} + \frac{\nu_{it}}{\sigma_{is}}\,d\tilde W_{it}^{s}
$$

- 第1項:予測部分(長期水準への回帰)
- 第2・3項:ゲイン × サプライズ(観測が2つあるので2項)

### 不確実性の更新(リカッチ方程式)

$$
d\nu_{it} = \left(\sigma_{\mu i}^2 - 2\kappa_i \nu_{it} - \frac{\lambda_i^2}{\sigma_{iX}^2}\nu_{it}^2 - \frac{1}{\sigma_{is}^2}\nu_{it}^2\right) dt
$$

- $\sigma_{\mu i}^2$:$\mu$自体が動くことで不確実性が増える項
- $-2\kappa_i \nu_{it}$:平均回帰によって不確実性の増加が抑えられる項
- 残り2項:観測によって不確実性が減る項

### サプライズ(イノベーション)の定義

$$
d\tilde W_{it}^{X} = \frac{dX_{it} - \lambda_i(\hat\mu_{it} - X_{it})\,dt}{\sigma_{iX}}, \qquad
d\tilde W_{it}^{s} = \frac{ds_{it} - \hat\mu_{it}\,dt}{\sigma_{is}}
$$

「実際に起きた変化」−「モデルが予測していた変化」を、標準偏差で割って標準化したもの。

---

## 3. 期待収益性の分解(Proposition 3.2)

$\tau > t$ 時点での期待収益性:

$$
E_t[X_{i,\tau}] = a_i X_{it} + b_i \hat\mu_{it} + c_i \bar\mu_i
$$

重み($\lambda_i, \kappa_i$ から計算、$a_i + b_i + c_i = 1$):

$$
a_i = e^{-\lambda_i(\tau-t)}, \qquad
b_i = \frac{\lambda_i}{\lambda_i - \kappa_i}\left(e^{-\kappa_i(\tau-t)} - e^{-\lambda_i(\tau-t)}\right), \qquad
c_i = 1 - a_i - b_i
$$

### 実装上のターゲット変数

$$
\text{ExpProf}_{it} = E_t[X_{i,t+1}] \quad \text{(期待収益性の水準)}
$$

$$
\Delta\text{ExpProf}_{it} = E_t[X_{i,t+1}] - X_{it} \quad \text{(期待収益性の"変化"、本命の投資シグナル)}
$$

---

## 4. 離散時間・実装版(4.4節)— 実際にコードで回す式

$dt = 1/4$(四半期を1とする単位で、月次ステップに相当)として、月末ごとに以下を計算する。

### サプライズの計算(実データから)

$$
\Delta X_{it} \equiv X_{it} - X_{i,t-1}, \qquad \Delta s_{it} \equiv s_{it} - s_{i,t-1}
$$

$$
\text{surprise}_X = \frac{\Delta X_{it} - \lambda_i(\hat\mu_{i,t-1} - X_{i,t-1})\,dt}{\sigma_{x,i}}, \qquad
\text{surprise}_s = \frac{\Delta s_{it} - \hat\mu_{i,t-1}\,dt}{\sigma_{s,i}}
$$

### 信念と不確実性の更新

$$
\hat\mu_{it} = \hat\mu_{i,t-1} + \kappa_i(\bar\mu_i - \hat\mu_{i,t-1})\,dt
+ \frac{\lambda_i \nu_{i,t-1}}{\sigma_{x,i}}\,\text{surprise}_X
+ \frac{\lambda_i \nu_{i,t-1}}{\sigma_{s,i}}\,\text{surprise}_s
$$

$$
\nu_{it} = \nu_{i,t-1} + \left(\sigma_{\mu,i}^2 - 2\kappa_i \nu_{i,t-1} - \frac{\lambda_i \nu_{i,t-1}^2}{\sigma_{x,i}^2} - \frac{\nu_{i,t-1}^2}{\sigma_{s,i}^2}\right) dt
$$

$$
\text{if } \nu_{it} < 0: \quad \nu_{it} = \nu_{i,t-1} \quad \text{(数値安定化のための安全策)}
$$

---

## 5. パラメータ推定(SMM、4.3節)

各企業 $i$、各推定時点 $t$ について、パラメータベクトル $\theta_{it} = (\lambda_{it}, \kappa_{it}, \sigma_{\mu,it})$ を、拡大窓 $\mathcal{W}_{it}$(2年ごとに再推定)内のデータを使って以下を最小化することで求める:

$$
\theta_{it} \in \arg\min_{\theta \in \Theta} \left(m_{it}^{\text{sim}}(\theta) - m_{it}^{\text{data}}\right)' W_{it} \left(m_{it}^{\text{sim}}(\theta) - m_{it}^{\text{data}}\right)
$$

ターゲットとする3つのモーメント($m_{it}^{\text{data}}$):

1. 企業ROEの自己相関
2. 業界ROEの自己相関
3. 企業ROEと業界ROEの共分散

初期化に使う補助的な統計量(窓内の実データから直接計算):

$$
\bar\mu_{it} = \text{窓内の } X_{i\tau} \text{ の標本平均}, \qquad
\sigma_{X,it} = \text{窓内の } X_{i\tau} \text{ の標本標準偏差}, \qquad
\sigma_{s,it} = \text{窓内の } s_{i\tau} \text{ の標本標準偏差}
$$

$(\lambda, \kappa)$ の初期値は、企業・業界ROEそれぞれの AR(1) 回帰から推定する。

---

## 6. 実装フロー(まとめ)

1. **データ準備**:各企業・各月について $X_{it}$(直近開示ROEの繰り越し)と $s_{it}$(リアルタイム業界中央値)を作る
2. **パラメータ推定(2年ごと)**:SMMで $\theta_{it} = (\lambda_i, \kappa_i, \sigma_{\mu,i})$ を推定 → 再推定日まで値を引き継ぐ
3. **毎月の信念更新ループ**:$\text{surprise}_X, \text{surprise}_s$ を計算 → $\hat\mu_{it}, \nu_{it}$ を更新(第4節の式)
4. **期待収益性の計算**:$a_i, b_i, c_i$ を $\lambda_i, \kappa_i$ から計算 → $E_t[X_{i,t+1}] = a_i X_{it} + b_i \hat\mu_{it} + c_i \bar\mu_i$ → $\Delta\text{ExpProf}_{it} = E_t[X_{i,t+1}] - X_{it}$
5. **シグナルの利用**:$\Delta\text{ExpProf}_{it}$ を月次でクロスセクションに10分位ソート → ロング・ショートポートフォリオ、または Fama-MacBeth 回帰の説明変数として利用
