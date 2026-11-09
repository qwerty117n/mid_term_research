# %% [markdown]
# 期待収益性 Ŷ による分位ポートフォリオ・RMW_Ŷ ファクター・スパニングテスト
# 前提（ノートブック上で定義・実行済み）:
#   - データ: ch, rt, df_mkt_cap
#   - table7_japan.py セルA の関数: add_expected, PROF
# 各 "# %%" ごとにノートブックのセルへ貼り付けて実行する

# %% セルA：関数定義
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

FACTOR_SETS = {
    'ff3': ['MKT', 'SMB', 'HML'],
    'ff5': ['MKT', 'SMB', 'HML', 'RMW', 'CMA'],
}


def show(title, obj=None, note=None):
    """表の前に見出しと説明を表示"""
    print('\n' + '=' * 80)
    print('■ ' + title)
    if note:
        for line in note.strip().split('\n'):
            print('  ' + line.strip())
    print('=' * 80)
    if obj is not None:
        display(obj)


# ---------- 共通の部品 ----------
def make_rtw(rt, df_mkt_cap):
    """月次リターンに、前月末時価総額（加重ウェイト）を付与"""
    w = df_mkt_cap.rename(columns={'ticker': 'security_code', 'market_cap': 'w'}).copy()
    w['month'] = pd.to_datetime(w['date'].astype(str), format='%Y%m').dt.to_period('M') + 1
    rtw = rt.merge(w[['security_code', 'month', 'w']], on=['security_code', 'month'], how='inner')
    return rtw[(rtw['w'] > 0) & rtw['ret'].notna()]


def assign_bins(sig, var, q, bp_mask=None):
    """形成年ごとに分位点で 0,1,2,... を割り当て。bp_mask でブレークポイント算出用の銘柄を限定"""
    out = pd.Series(np.nan, index=sig.index)
    for _, g in sig.groupby('form_year'):
        g = g.dropna(subset=[var])
        ref = g if bp_mask is None else g[bp_mask(g)]
        if len(ref) < 10:
            continue
        edges = ref[var].quantile(q).to_numpy()
        out.loc[g.index] = np.searchsorted(edges, g[var].to_numpy(), side='right')
    return out


def vw_ret(panel, keys):
    """月 × keys ごとの加重平均リターン（ウェイトは列 w）"""
    p = panel.assign(rw=panel['ret'] * panel['w'])
    g = p.groupby(['month'] + keys)[['rw', 'w']].sum()
    return (g['rw'] / g['w']).unstack(keys)


def ts_reg(y, X=None, nw_lags=12):
    """時系列回帰（Newey-West）。X=None なら平均の検定。返り値: 係数表（coef, t, p）、Adj-R2、月数"""
    d = y.rename('y').to_frame() if X is None else pd.concat([y.rename('y'), X], axis=1)
    d = d.dropna()
    names = ['alpha'] + ([] if X is None else list(X.columns))
    Xc = np.column_stack([np.ones(len(d))] + ([] if X is None else [d[c].to_numpy(float) for c in X.columns]))
    res = sm.OLS(d['y'].to_numpy(float), Xc).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
    tab = pd.DataFrame({'coef': res.params, 't': res.tvalues, 'p': res.pvalues}, index=names)
    return tab, res.rsquared_adj, len(d)


def two_by_three(sig, rtw, var, bp_mask=None):
    """FF 型 2×3 ソート（サイズ中央値 × var の 30/70%）。(高 − 低, 小 − 大) を返す"""
    s = sig.dropna(subset=[var, 'ME_jun']).copy()
    s['sz'] = assign_bins(s, 'ME_jun', [0.5], bp_mask)
    s['vb'] = assign_bins(s, var, [0.3, 0.7], bp_mask)
    s = s.dropna(subset=['sz', 'vb'])
    s[['sz', 'vb']] = s[['sz', 'vb']].astype(int)
    p = rtw.merge(s[['security_code', 'form_year', 'sz', 'vb']], on=['security_code', 'form_year'])
    R = vw_ret(p, ['sz', 'vb'])
    high_low = 0.5 * (R[(0, 2)] + R[(1, 2)]) - 0.5 * (R[(0, 0)] + R[(1, 0)])
    small_big = R.xs(0, level='sz', axis=1).mean(axis=1) - R.xs(1, level='sz', axis=1).mean(axis=1)
    return high_low, small_big


# ---------- シグナル ----------
def build_signal_table(ch, y, inv, **kw):
    """Y_t（実績）と Ŷ（期待）が両方そろう銘柄の表。kw は add_expected に渡す（例：winsor_dep=True）"""
    try:
        yhat, _ = add_expected(ch, y, inv, **kw)
    except TypeError:
        print(f'注意：add_expected が引数 {kw} に対応していないため、既定の設定で計算します')
        yhat, _ = add_expected(ch, y, inv)
    sig = ch.merge(yhat, on=['security_code', 'form_year'], how='inner')
    return sig.dropna(subset=[y, 'Yhat', 'ME_jun']).copy()


# ---------- 1. 分位ポートフォリオ ----------
def quantile_returns(sig, rtw, var='Yhat', n_q=5, weight='vw', bp_mask=None):
    """
    毎年6月に var で n_q 分位に分け、7月〜翌6月の月次リターンを計算
    返り値：R（列 Q1..Qn と H-L = Qn − Q1）、N（各分位の平均銘柄数）
    """
    s = sig.dropna(subset=[var]).copy()
    q = list(np.linspace(0, 1, n_q + 1)[1:-1])
    s['q'] = assign_bins(s, var, q, bp_mask)
    s = s.dropna(subset=['q'])
    s['q'] = s['q'].astype(int)
    p = rtw.merge(s[['security_code', 'form_year', 'q']], on=['security_code', 'form_year'])
    if weight == 'ew':
        p = p.assign(w=1.0)
    R = vw_ret(p, ['q']).reindex(columns=range(n_q))
    R.columns = [f'Q{c + 1}' for c in R.columns]
    R['H-L'] = R[f'Q{n_q}'] - R['Q1']
    N = p.groupby(['month', 'q']).size().unstack().mean().reindex(range(n_q))
    N.index = [f'Q{c + 1}' for c in N.index]
    return R, N


def quantile_summary(R, N=None, nw_lags=12):
    """各分位と H-L の平均・t値・p値・年率シャープレシオ・平均銘柄数"""
    rows = {}
    for c in R.columns:
        r = R[c].dropna()
        est, _, n = ts_reg(r, None, nw_lags)
        rows[c] = {'平均 (%/月)': round(est.loc['alpha', 'coef'] * 100, 3),
                   't値 (NW)': round(est.loc['alpha', 't'], 2),
                   'p値': round(est.loc['alpha', 'p'], 3),
                   '年率Sharpe': round(r.mean() / r.std() * np.sqrt(12), 2),
                   '月数': n,
                   '平均銘柄数': round(N[c], 0) if (N is not None and c in N.index) else np.nan}
    return pd.DataFrame(rows).T


def spread_verdict(R, nw_lags=12, alpha=0.05):
    """最上位と最下位のスプレッドの検定結果を文章で返す"""
    est, _, n = ts_reg(R['H-L'], None, nw_lags)
    m, t, p = est.loc['alpha', 'coef'] * 100, est.loc['alpha', 't'], est.loc['alpha', 'p']
    sig_txt = '有意である' if p < alpha else '有意ではない'
    return (f'H-L（最上位 − 最下位）の平均は {m:.3f}%/月、t値 {t:.2f}、p値 {p:.3f}（{n}か月）。'
            f'{int(alpha * 100)}%水準でスプレッドは{sig_txt}。')


def plot_quantiles(R, n_q, title):
    """左：分位ごとの累積リターン（重ね描き）、中：分位ごとの平均（95%区間）、右：H-L の累積リターン"""
    Q = [f'Q{i + 1}' for i in range(n_q)]
    colors = plt.get_cmap('viridis')(np.linspace(0, 0.9, n_q))
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    for i, c in enumerate(Q):
        cum = (1 + R[c].fillna(0)).cumprod()
        axes[0].plot(cum.index.to_timestamp(), cum.to_numpy(), color=colors[i], label=c)
    axes[0].set_yscale('log')
    axes[0].set_title('Cumulative return by quantile')
    axes[0].legend()

    m = R[Q].mean() * 100
    se = R[Q].std() / np.sqrt(R[Q].count()) * 100
    axes[1].bar(Q, m.to_numpy(), yerr=1.96 * se.to_numpy(), color=colors, capsize=3)
    axes[1].axhline(0, color='gray', lw=0.8)
    axes[1].set_ylabel('Mean monthly return (%)')
    axes[1].set_title('Average return by quantile (bars: 95% CI)')

    cum = (1 + R['H-L'].fillna(0)).cumprod()
    axes[2].plot(cum.index.to_timestamp(), cum.to_numpy(), color='black')
    axes[2].axhline(1, color='gray', lw=0.8)
    axes[2].set_yscale('log')
    axes[2].set_title(f'Cumulative return of H-L (Q{n_q} - Q1)')

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


# ---------- 2. RMW ファクター ----------
def build_rmw_pair(sig, rtw, y, bp_mask=None):
    """同じ銘柄で、期待収益性版（Ŷ）と実績収益性版（Y_t）の RMW を作る"""
    rmw_exp, _ = two_by_three(sig, rtw, 'Yhat', bp_mask)
    rmw_act, _ = two_by_three(sig, rtw, y, bp_mask)
    return rmw_exp, rmw_act


def factor_stats(df, nw_lags=12):
    """ファクターの基本統計"""
    rows = {}
    for c in df.columns:
        x = df[c].dropna()
        est, _, n = ts_reg(x, None, nw_lags)
        cum = (1 + x).cumprod()
        rows[c] = {'平均 (%/月)': round(x.mean() * 100, 3),
                   't値 (NW)': round(est.loc['alpha', 't'], 2),
                   '標準偏差 (%/月)': round(x.std() * 100, 3),
                   '年率Sharpe': round(x.mean() / x.std() * np.sqrt(12), 2),
                   '最大ドローダウン (%)': round((cum / cum.cummax() - 1).min() * 100, 1),
                   '開始': str(x.index.min()), '終了': str(x.index.max()), '月数': n}
    return pd.DataFrame(rows).T


def plot_factor_pairs(own, pairs, ref=None, window=60):
    """
    own: 自前ファクターの DataFrame。pairs: [(期待版の列名, 実績版の列名, 図のタイトル), ...]
    ref: 参考として重ねる外部ファクター（例：French の RMW）。window: 移動平均の月数
    上段：累積リターン、下段：移動平均リターン（年率）
    """
    fig, axes = plt.subplots(2, len(pairs), figsize=(7 * len(pairs), 8), squeeze=False)
    for j, (c_exp, c_act, ttl) in enumerate(pairs):
        lines = [(c_exp, own[c_exp], 'tab:blue'), (c_act, own[c_act], 'tab:orange')]
        if ref is not None:
            lines.append((ref.name, ref, 'tab:gray'))
        for label, x, col in lines:
            x = x.dropna()
            cum = (1 + x).cumprod()
            axes[0, j].plot(cum.index.to_timestamp(), cum.to_numpy(), color=col, label=label)
            roll = x.rolling(window).mean() * 12 * 100
            axes[1, j].plot(roll.index.to_timestamp(), roll.to_numpy(), color=col, label=label)
        axes[0, j].set_yscale('log')
        axes[0, j].axhline(1, color='gray', lw=0.8)
        axes[0, j].set_title(f'{ttl}: cumulative return')
        axes[0, j].legend()
        axes[1, j].axhline(0, color='gray', lw=0.8)
        axes[1, j].set_title(f'{ttl}: {window}-month rolling mean (% p.a.)')
        axes[1, j].legend()
    fig.tight_layout()
    plt.show()


# ---------- 3. スパニングテスト ----------
def load_french_factors(path):
    """
    French Data Library 形式の月次ファクター（Date, Mkt-RF, SMB, HML, RMW, CMA, RF；単位 %）を読み込み、小数に直す
    返り値: index = Period[M]、列 = MKT, SMB, HML, RMW, CMA, RF
    """
    ff = pd.read_csv(path, skipinitialspace=True)
    ff.columns = [c.strip() for c in ff.columns]
    ff = ff.rename(columns={'Date': 'date', 'Mkt-RF': 'MKT'})
    ff['month'] = pd.to_datetime(ff['date'].astype(str).str.strip(), format='%Y%m').dt.to_period('M')
    ff = ff.drop(columns='date').set_index('month').astype(float) / 100   # % → 小数
    return ff


def _fmt_reg(est, r2, n):
    """回帰結果を「係数 (t値)」の形に整形"""
    out = {}
    for k in est.index:
        scale = 100 if k == 'alpha' else 1
        label = 'α (%/月)' if k == 'alpha' else f'β {k}'
        out[label] = f"{est.loc[k, 'coef'] * scale:.3f} ({est.loc[k, 't']:.2f})"
    out['Adj-R2'] = f'{r2:.2f}'
    out['月数'] = n
    return out


def spanning_tests(own, FF, pairs, model='ff5', nw_lags=12):
    """
    各ペア（期待版・実績版）について、外部ファクターモデルに対するアルファと、相互スパニングを検定
    model: 'ff5'（既定：MKT, SMB, HML, RMW, CMA）／ 'ff3'（MKT, SMB, HML）
    比較を公平にするため、各ペアの中では同じ月で推定する
    """
    base = FACTOR_SETS[model]
    tables = {}
    for c_exp, c_act, ttl in pairs:
        G = pd.concat([FF[base], own[[c_exp, c_act]]], axis=1).dropna()
        specs = {
            f'{c_exp} ~ {model.upper()}':            (c_exp, base),
            f'{c_act} ~ {model.upper()}':            (c_act, base),
            f'{c_exp} ~ {model.upper()} + {c_act}':  (c_exp, base + [c_act]),
            f'{c_act} ~ {model.upper()} + {c_exp}':  (c_act, base + [c_exp]),
        }
        cols = {}
        for name, (dep, xs) in specs.items():
            est, r2, n = ts_reg(G[dep], G[xs], nw_lags)
            cols[name] = _fmt_reg(est, r2, n)
        tables[ttl] = pd.DataFrame(cols).fillna('–')
    return tables


# %% セルB：設定と準備
N_Q = 5                              # 分位数（3, 5, 10 など）
WEIGHT = 'vw'                        # 'vw'：時価総額加重 ／ 'ew'：等ウェイト
MODEL = 'ff5'                        # スパニングテストのモデル：'ff5'（既定）／ 'ff3'
FACTOR_PATH = '../data/factor_rtn.csv'
SIGNAL_KW = {'CBOP': {}, 'CP': {'winsor_dep': True}}   # add_expected に渡す引数（CP は外れ値対策）

rtw = make_rtw(rt, df_mkt_cap)
sigs = {y: build_signal_table(ch, y, inv, **SIGNAL_KW[y]) for y, inv in PROF.items()}


# %% セルC：Ŷ による分位ポートフォリオ（分位ごとの重ね描き・スプレッドの検定）
quant = {}
for y in PROF:
    R, N = quantile_returns(sigs[y], rtw, 'Yhat', N_Q, WEIGHT)
    quant[y] = R
    show(f'[分位ポートフォリオ] 期待{y}（Ŷ）で {N_Q} 分位・{WEIGHT.upper()}',
         quantile_summary(R, N),
         note='毎年6月に Ŷ で分位を作り、7月〜翌6月の月次リターンを計算（%/月、t値は Newey-West）。\n'
              'H-L = 最上位 − 最下位（ゼロコストのロングショート）。\n' + spread_verdict(R))
    plot_quantiles(R, N_Q, f'Expected {y} (Yhat) | {N_Q} quantiles | {WEIGHT.upper()}')


# %% セルD：RMW_Ŷ（期待）と RMW_Y（実績）の作成・基本統計・描画
own = {}
for y in PROF:
    rmw_exp, rmw_act = build_rmw_pair(sigs[y], rtw, y)
    own[f'RMW_exp{y}'] = rmw_exp
    own[f'RMW_act{y}'] = rmw_act
own = pd.DataFrame(own)

FF = load_french_factors(FACTOR_PATH)

show('[RMW] 自前ファクターの基本統計', factor_stats(own),
     note='2×3 ソート（規模の中央値 × 収益性の 30/70%）、時価総額加重、毎年6月に組み替え。\n'
          'RMW_exp：期待収益性（Ŷ）で作成 ／ RMW_act：実績収益性（Y_t）で作成（同じ銘柄の範囲）。')

show('[RMW] 相関行列（自前ファクターと French ファクター）',
     pd.concat([own, FF[['MKT', 'SMB', 'HML', 'RMW', 'CMA']]], axis=1).dropna().corr().round(2),
     note='French ファクターと重なる期間で計算。')

PAIRS = [('RMW_expCBOP', 'RMW_actCBOP', 'CBOP: expected vs actual'),
         ('RMW_expCP', 'RMW_actCP', 'CP: expected vs actual')]
plot_factor_pairs(own, PAIRS, ref=FF['RMW'].rename('French RMW'))


# %% セルE：スパニングテスト（既定：FF5。MODEL='ff3' で3ファクター）
for ttl, tab in spanning_tests(own, FF, PAIRS, model=MODEL).items():
    show(f'[スパニング] {ttl}（{MODEL.upper()}）', tab,
         note='各列が1本の時系列回帰。α は %/月、括弧内は Newey-West t値。\n'
              '「~ モデル」：外部ファクターで説明できないアルファが残るか。\n'
              '「~ モデル + 他方」：もう一方の RMW を加えてもアルファが残るか（残れば、その RMW は他方にない情報を持つ）。')

# 3ファクターでの検証も並べる場合
for ttl, tab in spanning_tests(own, FF, PAIRS, model='ff3').items():
    show(f'[スパニング] {ttl}（FF3）', tab)
