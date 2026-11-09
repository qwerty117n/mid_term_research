# %% [markdown]
# 期待収益性 Ŷ による分位ポートフォリオ・RMW_Ŷ ファクター・スパニングテスト
# 前提（ノートブック上で定義・実行済み）:
#   - データ: ch, rt, df_mkt_cap
#   - table7_japan.py セルA の関数: add_expected, PROF
# 外部ファクターは French Data Library 形式の CSV（FACTOR_PATH）から読み込む。
# French のデータは 1990年7月以降のみなので、ファクターの検証・描画はその期間にそろえる。
# 各 "# %%" ごとにノートブックのセルへ貼り付けて実行する

# %% セルA：関数定義
import os
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


# ---------- 期間の扱い ----------
def to_period(x):
    """'1999-12'、'199912'、199912、pd.Period のいずれも月次の Period に変換（None はそのまま）"""
    if x is None or isinstance(x, pd.Period):
        return x
    s = str(x).replace('-', '').replace('/', '').strip()
    return pd.Period(f'{s[:4]}-{s[4:6]}', freq='M')


def slice_period(df, start=None, end=None):
    """index（Period[M]）を開始・終了月で絞る。None は端まで"""
    return df.loc[to_period(start):to_period(end)]


def align_to_ff(df, FF):
    """
    French ファクターに値がある月だけを残し、さらに df のすべての列に値がある月に絞る
    （French より前の月・後の月、どちらかが欠けている月を除き、両者の長さをそろえる）
    """
    idx = df.index.intersection(FF.dropna().index)
    return df.loc[idx].dropna()


def period_info(df):
    """データの期間と月数を文字列で返す"""
    return f'{df.index.min()}〜{df.index.max()}（{len(df)}か月）'


# ---------- 共通の部品 ----------
def load_french_factors(path):
    """
    French Data Library 形式の月次ファクター（Date, Mkt-RF, SMB, HML, RMW, CMA, RF；単位 %）を読み込み、小数に直す
    返り値: index = Period[M]、列 = MKT, SMB, HML, RMW, CMA, RF
    """
    ff = pd.read_csv(path, skipinitialspace=True)
    ff.columns = [c.strip() for c in ff.columns]
    ff = ff.rename(columns={'Date': 'date', 'Mkt-RF': 'MKT'})
    ff['month'] = pd.to_datetime(ff['date'].astype(str).str.strip(), format='%Y%m').dt.to_period('M')
    ff = ff.drop(columns='date').set_index('month').sort_index().astype(float) / 100   # % → 小数
    return ff


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
def _expected(ch, y, inv, **kw):
    """add_expected を呼ぶ（古い版で引数に未対応なら、xvars 以外を外して再計算）"""
    try:
        return add_expected(ch, y, inv, **kw)[0]
    except TypeError:
        kw2 = {k: v for k, v in kw.items() if k == 'xvars'}
        print(f'注意：add_expected が引数 {kw} に対応していないため、{kw2} で計算します')
        return add_expected(ch, y, inv, **kw2)[0]


def build_signal_table(ch, y, inv, **kw):
    """
    実績 Y_t、期待 Ŷ（フル：Y, ΔY, B/M, 規模, 投資）、期待 Ŷ_prof（収益性のみ：Y, ΔY）が
    すべてそろう銘柄の表。kw は add_expected に渡す（例：winsor_dep=True）
    """
    keys = ['security_code', 'form_year']
    yhat = _expected(ch, y, inv, **kw)
    yprof = _expected(ch, y, inv, xvars='prof', **kw).rename(columns={'Yhat': 'Yhat_prof'})
    sig = ch.merge(yhat, on=keys, how='inner').merge(yprof, on=keys, how='inner')
    return sig.dropna(subset=[y, 'Yhat', 'Yhat_prof', 'ME_jun']).copy()


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


def quantile_summary(R, N=None, FF=None, model='ff5', nw_lags=12):
    """各分位と H-L の平均・t値・p値・年率シャープレシオ・French モデルに対するアルファ"""
    rows = {}
    for c in R.columns:
        r = R[c].dropna()
        est, _, n = ts_reg(r, None, nw_lags)
        row = {'平均 (%/月)': round(est.loc['alpha', 'coef'] * 100, 3),
               't値 (NW)': round(est.loc['alpha', 't'], 2),
               'p値': round(est.loc['alpha', 'p'], 3),
               '年率Sharpe': round(r.mean() / r.std() * np.sqrt(12), 2)}
        if FF is not None:
            ex = r if c == 'H-L' else (r - FF['RF'].reindex(r.index)).dropna()   # H-L はゼロコスト
            a, _, _ = ts_reg(ex, FF[FACTOR_SETS[model]], nw_lags)
            row[f'{model.upper()} α (%/月)'] = round(a.loc['alpha', 'coef'] * 100, 3)
            row[f'{model.upper()} α t値'] = round(a.loc['alpha', 't'], 2)
        row['月数'] = n
        row['平均銘柄数'] = round(N[c], 0) if (N is not None and c in N.index) else np.nan
        rows[c] = row
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

    fig.suptitle(f'{title} | {R.index.min()} - {R.index.max()}')
    fig.tight_layout()
    plt.show()


# ---------- 2. RMW ファクター ----------
def build_rmw_set(sig, rtw, y, bp_mask=None):
    """同じ銘柄で、期待（フル）・期待（収益性のみ）・実績 の3種類の RMW を作る"""
    rmw_exp, _ = two_by_three(sig, rtw, 'Yhat', bp_mask)
    rmw_prof, _ = two_by_three(sig, rtw, 'Yhat_prof', bp_mask)
    rmw_act, _ = two_by_three(sig, rtw, y, bp_mask)
    return {f'RMW_exp{y}': rmw_exp, f'RMW_expprof{y}': rmw_prof, f'RMW_act{y}': rmw_act}


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


def plot_factor_groups(own, groups, ref=None, window=60):
    """
    own: 自前ファクターの DataFrame。groups: [([列名, ...], 図のタイトル), ...]
    ref: 参考として重ねる外部ファクター（例：French の RMW）。window: 移動平均の月数
    上段：累積リターン、下段：移動平均リターン（年率）
    各図では、描く系列（ref を含む）すべてに値がある共通の月だけを使い、同じ月を起点（=1）に累積する
    """
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple']
    fig, axes = plt.subplots(2, len(groups), figsize=(7 * len(groups), 8), squeeze=False)
    for j, (cols, ttl) in enumerate(groups):
        parts = [own[cols]] + ([ref] if ref is not None else [])
        D = pd.concat(parts, axis=1).dropna()           # 共通の月にそろえる
        palette = {c: colors[k % len(colors)] for k, c in enumerate(cols)}
        if ref is not None:
            palette[ref.name] = 'tab:gray'
        for label in D.columns:
            x = D[label]
            cum = (1 + x).cumprod()
            axes[0, j].plot(cum.index.to_timestamp(), cum.to_numpy(), color=palette[label], label=label)
            roll = x.rolling(window).mean() * 12 * 100
            axes[1, j].plot(roll.index.to_timestamp(), roll.to_numpy(), color=palette[label], label=label)
        axes[0, j].set_yscale('log')
        axes[0, j].axhline(1, color='gray', lw=0.8)
        axes[0, j].set_title(f'{ttl}: cumulative return\n({D.index.min()} - {D.index.max()}, {len(D)} months)')
        axes[0, j].legend()
        axes[1, j].axhline(0, color='gray', lw=0.8)
        axes[1, j].set_title(f'{ttl}: {window}-month rolling mean (% p.a.)')
        axes[1, j].legend()
    fig.tight_layout()
    plt.show()


# ---------- 3. スパニングテスト ----------
def _fmt_reg(est, r2, n, idx):
    """回帰結果を「係数 (t値)」の形に整形"""
    out = {}
    for k in est.index:
        scale = 100 if k == 'alpha' else 1
        label = 'α (%/月)' if k == 'alpha' else f'β {k}'
        out[label] = f"{est.loc[k, 'coef'] * scale:.3f} ({est.loc[k, 't']:.2f})"
    out['Adj-R2'] = f'{r2:.2f}'
    out['期間'] = f'{idx.min()}〜{idx.max()}'
    out['月数'] = n
    return out


def spanning_tests(own, FF, pairs, model='ff5', start=None, end=None, nw_lags=12):
    """
    各ペア（検証対象・比較対象）について、外部ファクターモデルに対するアルファと、相互スパニングを検定
    model: 'ff5'（既定：MKT, SMB, HML, RMW, CMA）／ 'ff3'（MKT, SMB, HML）
    start, end: 推定期間（例：'2000-01'、'201912'）。None なら French ファクターと重なる全期間
    比較を公平にするため、各ペアの中では同じ月で推定する
    """
    base = FACTOR_SETS[model]
    M = model.upper()
    tables = {}
    for c_a, c_b, ttl in pairs:
        G = slice_period(pd.concat([FF[base], own[[c_a, c_b]]], axis=1).dropna(), start, end)
        specs = {
            f'{c_a} ~ {M}':          (c_a, base),
            f'{c_b} ~ {M}':          (c_b, base),
            f'{c_a} ~ {M} + {c_b}':  (c_a, base + [c_b]),
            f'{c_b} ~ {M} + {c_a}':  (c_b, base + [c_a]),
        }
        cols = {}
        for name, (dep, xs) in specs.items():
            est, r2, n = ts_reg(G[dep], G[xs], nw_lags)
            cols[name] = _fmt_reg(est, r2, n, G.index)
        tables[ttl] = pd.DataFrame(cols).fillna('–')
    return tables


def alpha_by_period(own, FF, specs, periods, model='ff5', nw_lags=12):
    """
    期間ごとのアルファ（%/月、括弧内 t値）を一覧にする
    specs: {列名: (被説明変数, [追加の説明変数, ...])}。外部ファクター（model）は常に含める
    periods: {期間名: (開始, 終了)}
    """
    base = FACTOR_SETS[model]
    used = sorted({dep for dep, _ in specs.values()} | {c for _, extra in specs.values() for c in extra})
    G_all = pd.concat([FF[base], own[used]], axis=1).dropna()   # 全指定で共通の月にそろえる
    rows = {}
    for pname, (st, en) in periods.items():
        G = slice_period(G_all, st, en)
        row = {}
        for name, (dep, extra) in specs.items():
            est, _, n = ts_reg(G[dep], G[base + extra], nw_lags)
            row[name] = f"{est.loc['alpha', 'coef'] * 100:.3f} ({est.loc['alpha', 't']:.2f})"
        row['期間'] = f'{G.index.min()}〜{G.index.max()}'
        row['月数'] = len(G)
        rows[pname] = row
    return pd.DataFrame(rows).T


# %% セルB：設定と準備
FACTOR_PATH = '../data/factor_rtn.csv'   # French Data Library 形式のファクターの CSV
MODEL = 'ff5'                            # スパニングテストのモデル：'ff5'（既定）／ 'ff3'
START, END = None, None                  # スパニングテストの推定期間（例：'2000-01', '2019-12'）。None は端まで
N_Q = 5                                  # 分位数（3, 5, 10 など）
WEIGHT = 'vw'                            # 'vw'：時価総額加重 ／ 'ew'：等ウェイト
ALIGN_TO_FF = True                       # True：分位ポートフォリオも French ファクターの期間にそろえる
SIGNAL_KW = {'CBOP': {}, 'CP': {'winsor_dep': True}}   # add_expected に渡す引数（CP は外れ値対策）

print('作業フォルダ:', os.getcwd(), '| ファイルの有無:', os.path.exists(FACTOR_PATH))
FF = load_french_factors(FACTOR_PATH)
show('[確認] French ファクター（小数表記）', FF.describe().T.round(4),
     note=f'期間：{FF.index.min()} 〜 {FF.index.max()}（以降のファクター検証・描画はこの期間にそろえる）')

rtw = make_rtw(rt, df_mkt_cap)
sigs = {y: build_signal_table(ch, y, inv, **SIGNAL_KW[y]) for y, inv in PROF.items()}


# %% セルC：Ŷ による分位ポートフォリオ（分位ごとの重ね描き・スプレッドの検定）
quant = {}
for y in PROF:
    for var, label in [('Yhat', '期待（フル）'), ('Yhat_prof', '期待（収益性のみ）')]:
        R, N = quantile_returns(sigs[y], rtw, var, N_Q, WEIGHT)
        if ALIGN_TO_FF:
            R = align_to_ff(R, FF)   # French に値がある月、かつ全分位に値がある月にそろえる
        quant[(y, var)] = R
        show(f'[分位ポートフォリオ] {y}：{label} で {N_Q} 分位・{WEIGHT.upper()}',
             quantile_summary(R, N, FF, MODEL),
             note=f'期間：{R.index.min()}〜{R.index.max()}。毎年6月に分位を作り、7月〜翌6月の月次リターン（%/月、t値は Newey-West）。\n'
                  'H-L = 最上位 − 最下位（ゼロコストのロングショート）。各分位のアルファは French の RF を引いた超過リターンで推定。\n'
                  + spread_verdict(R))
        plot_quantiles(R, N_Q, f'{y} | {var} | {N_Q} quantiles | {WEIGHT.upper()}')


# %% セルD：RMW（期待・期待（収益性のみ）・実績）の作成・基本統計・描画
own_all = {}
for y in PROF:
    own_all.update(build_rmw_set(sigs[y], rtw, y))
own_all = pd.DataFrame(own_all)
own = align_to_ff(own_all, FF)   # French に値があり、かつ自前の全ファクターに値がある月だけ

show('[確認] 期間のそろえ方',
     pd.DataFrame({'期間': [period_info(own_all.dropna()), period_info(FF.dropna()), period_info(own)]},
                  index=['自前ファクター（そろえる前）', 'French ファクター', '共通期間（以降の計算・描画に使用）']))

show('[RMW] 自前ファクターの基本統計（共通期間）', factor_stats(own),
     note='2×3 ソート（規模の中央値 × 収益性の 30/70%）、時価総額加重、毎年6月に組み替え。すべて同じ銘柄の範囲。\n'
          'RMW_exp：期待（フル：Y, ΔY, B/M, 規模, 投資）／ RMW_expprof：期待（収益性のみ：Y, ΔY）／ RMW_act：実績 Y_t。\n'
          '全ファクターを French と同じ共通期間で計算しているので、開始・終了・月数はすべて一致する。')

show('[RMW] 相関行列（自前ファクターと French ファクター）',
     pd.concat([own, FF[['MKT', 'SMB', 'HML', 'RMW', 'CMA']]], axis=1).dropna().corr().round(2))

GROUPS = [([f'RMW_exp{y}', f'RMW_expprof{y}', f'RMW_act{y}'], f'{y}: expected vs actual') for y in PROF]
plot_factor_groups(own, GROUPS, ref=FF['RMW'].rename('French RMW'))


# %% セルE：スパニングテスト（期待（フル）vs 実績、期待（収益性のみ）vs 実績）
PAIRS = [(f'RMW_exp{y}', f'RMW_act{y}', f'{y}：期待（フル） vs 実績') for y in PROF]
PAIRS_PROF = [(f'RMW_expprof{y}', f'RMW_act{y}', f'{y}：期待（収益性のみ） vs 実績') for y in PROF]

for ttl, tab in spanning_tests(own, FF, PAIRS + PAIRS_PROF, model=MODEL, start=START, end=END).items():
    show(f'[スパニング] {ttl}（{MODEL.upper()}）', tab,
         note='各列が1本の時系列回帰。α は %/月、括弧内は Newey-West t値。\n'
              '「~ モデル」：外部ファクターで説明できないアルファが残るか。\n'
              '「~ モデル + 他方」：もう一方の RMW を加えてもアルファが残るか（残れば、他方にない情報を持つ）。\n'
              '期待（収益性のみ）でアルファが残れば、改善の源は規模・投資ではなく収益性の平均回帰。')


# %% セルF：期間を変えたスパニングテスト（時期によらないことの確認）
PERIODS = {
    'French の全期間': (None, None),
    '1990年代':        (None, '1999-12'),
    '2000年代':        ('2000-01', '2009-12'),
    '2010年以降':      ('2010-01', None),
    '前半':            (None, '2007-06'),
    '後半':            ('2007-07', None),
}
for y in PROF:
    specs = {
        f'exp ~ {MODEL.upper()}':              (f'RMW_exp{y}', []),
        f'expprof ~ {MODEL.upper()}':          (f'RMW_expprof{y}', []),
        f'act ~ {MODEL.upper()}':              (f'RMW_act{y}', []),
        f'exp ~ {MODEL.upper()} + act':        (f'RMW_exp{y}', [f'RMW_act{y}']),
        f'expprof ~ {MODEL.upper()} + act':    (f'RMW_expprof{y}', [f'RMW_act{y}']),
        f'act ~ {MODEL.upper()} + exp':        (f'RMW_act{y}', [f'RMW_exp{y}']),
    }
    show(f'[期間別アルファ] {y}（{MODEL.upper()}）', alpha_by_period(own, FF, specs, PERIODS, MODEL),
         note='各セルは α（%/月）と括弧内の Newey-West t値。期間を短くすると t値は下がるので、\n'
              '各期間で有意かどうかより、期待版のアルファの符号と大きさが安定しているかを見る。\n'
              '期間の区切りは PERIODS で自由に変更できる（「前半・後半」は French の期間をおおよそ二分）。')
