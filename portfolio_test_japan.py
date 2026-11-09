# %% [markdown]
# 分位ポートフォリオによる検証（Y_t / Ŷ / リビジョン）
# 前提: ノートブック上で ch, rt, df_mkt_cap と、table7_japan.py セルA の関数（add_expected, PROF）が定義済み
# 各 "# %%" ごとにノートブックのセルへ貼り付けて実行する

# %% セルA：関数定義
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


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


# ---------- 共通の部品（h3_factor_japan.py と同じ内容） ----------
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
    """時系列回帰（Newey-West）。X=None なら平均の検定"""
    d = y.rename('y').to_frame() if X is None else pd.concat([y.rename('y'), X], axis=1)
    d = d.dropna()
    names = ['alpha'] + ([] if X is None else list(X.columns))
    Xc = np.column_stack([np.ones(len(d))] + ([] if X is None else [d[c].to_numpy(float) for c in X.columns]))
    res = sm.OLS(d['y'].to_numpy(float), Xc).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
    return pd.DataFrame({'coef': res.params, 't': res.tvalues}, index=names), res.rsquared_adj


def two_by_three(sig, rtw, var, bp_mask=None):
    """FF 型 2×3 ソート（サイズ中央値 × var の 30/70%）。(高 - 低, 小 - 大) を返す"""
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


def build_factors(ch, rtw, bp_mask=None, rf=None):
    """自前の日本版ファクター（円建て）: MKT, SMB, HML, CMA"""
    mkt = vw_ret(rtw.assign(_all=0), ['_all'])[0]
    if rf is not None:
        mkt = mkt - rf.reindex(mkt.index)
    hml, smb = two_by_three(ch.dropna(subset=['lnBM']), rtw, 'lnBM', bp_mask)
    aggr_cons, _ = two_by_three(ch, rtw, 'dA', bp_mask)
    return pd.DataFrame({'MKT': mkt, 'SMB': smb, 'HML': hml, 'CMA': -aggr_cons}).dropna()


# ---------- シグナルの準備 ----------
def prepare_signals(ch, y, inv, size=None, **kw):
    """
    Y_t、Ŷ、リビジョン（Ŷ − Y_t）を同じ銘柄で揃えた表を作る
    size: None（全銘柄）/ 'big'（各年の時価総額中央値以上）/ 'small'（中央値未満）
    kw: add_expected に渡す引数（xvars, timing, avg_years, winsor_dep）
    """
    yhat, _ = add_expected(ch, y, inv, **kw)
    sig = ch.merge(yhat, on=['security_code', 'form_year'], how='inner')
    sig = sig.dropna(subset=[y, 'Yhat', 'ME_jun']).copy()
    sig['rev'] = sig['Yhat'] - sig[y]
    if size is not None:
        med = sig.groupby('form_year')['ME_jun'].transform('median')
        sig = sig[sig['ME_jun'] >= med] if size == 'big' else sig[sig['ME_jun'] < med]
    return sig


# ---------- 分位ポートフォリオ ----------
def quantile_portfolios(sig, rtw, var, n_q=5, weight='vw', bp_mask=None):
    """
    毎年6月に var で n_q 分位に分け、7月〜翌6月の月次リターンを計算
    weight='vw'：前月末時価総額で加重 ／ 'ew'：等ウェイト
    返り値：R（月次リターン。列 Q1..Qn と H-L = Qn − Q1）、N（各分位の平均銘柄数）
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


def summarize_portfolios(R, N=None, F=None, rf=None, nw_lags=12):
    """各分位と H-L の平均・t値・年率シャープレシオ・FF3/FF4 アルファ"""
    rows = {}
    for c in R.columns:
        r = R[c].dropna()
        ex = r if (rf is None or c == 'H-L') else (r - rf.reindex(r.index)).dropna()
        est, _ = ts_reg(r, None, nw_lags)
        row = {'平均 (%/月)': est.loc['alpha', 'coef'] * 100,
               't値': est.loc['alpha', 't'],
               '年率Sharpe': ex.mean() / ex.std() * np.sqrt(12)}
        if F is not None:
            for name, cols in [('FF3 α', ['MKT', 'SMB', 'HML']), ('FF4 α', ['MKT', 'SMB', 'HML', 'CMA'])]:
                e, _ = ts_reg(ex, F[cols], nw_lags)
                row[name] = e.loc['alpha', 'coef'] * 100
                row[f'{name} t値'] = e.loc['alpha', 't']
        if N is not None and c in N.index:
            row['平均銘柄数'] = N[c]
        rows[c] = row
    return pd.DataFrame(rows).T.round(3)


def monotonicity(R):
    """分位の平均リターンが単調に増えるかの目安"""
    means = R.drop(columns='H-L').mean().to_numpy()
    steps = np.diff(means)
    rank_corr = pd.Series(means).corr(pd.Series(np.arange(len(means))), method='spearman')
    return pd.Series({'上昇した段の数': int((steps > 0).sum()),
                      '段の総数': len(steps),
                      '順位相関（分位番号 vs 平均）': round(rank_corr, 3)})


def diff_test(R_a, R_b, nw_lags=12):
    """2つのシグナルの H-L の差（a − b）の平均と t 値"""
    d = (R_a['H-L'] - R_b['H-L']).dropna()
    est, _ = ts_reg(d, None, nw_lags)
    return pd.Series({'差の平均 (%/月)': round(est.loc['alpha', 'coef'] * 100, 3),
                      't値': round(est.loc['alpha', 't'], 2),
                      '月数': len(d)})


def quantile_characteristics(sig, var, n_q=5, cols=None, bp_mask=None):
    """各分位の平均的な特性（形成年ごとの平均を、さらに年で平均）"""
    s = sig.dropna(subset=[var]).copy()
    q = list(np.linspace(0, 1, n_q + 1)[1:-1])
    s['q'] = assign_bins(s, var, q, bp_mask)
    s = s.dropna(subset=['q'])
    s['q'] = s['q'].astype(int) + 1
    s['ME_jun_億円'] = s['ME_jun'] / 1e8
    cols = cols or [c for c in [var, 'ME_jun_億円', 'lnBM', 'dA', 'rev'] if c in s.columns]
    out = s.groupby(['form_year', 'q'])[cols].mean().groupby('q').mean()
    out.index = [f'Q{i}' for i in out.index]
    return out.round(4)


def plot_quantiles(results, n_q, title):
    """
    results: {ラベル: R}。左：分位ごとの平均リターン（95% 区間）、右：H-L の累積リターン
    ※ matplotlib の既定フォントは日本語非対応のため、図中のラベルは英語
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(n_q)
    width = 0.8 / len(results)
    for k, (label, R) in enumerate(results.items()):
        Q = R[[f'Q{i + 1}' for i in range(n_q)]]
        m = Q.mean() * 100
        se = Q.std() / np.sqrt(Q.count()) * 100
        axes[0].bar(x + k * width, m.to_numpy(), width, yerr=1.96 * se.to_numpy(),
                    capsize=3, label=label)
        cum = (1 + R['H-L'].fillna(0)).cumprod()
        axes[1].plot(cum.index.to_timestamp(), cum.to_numpy(), label=label)
    axes[0].set_xticks(x + width * (len(results) - 1) / 2)
    axes[0].set_xticklabels([f'Q{i + 1}' for i in range(n_q)])
    axes[0].axhline(0, color='gray', lw=0.8)
    axes[0].set_ylabel('Mean monthly return (%)')
    axes[0].set_title('Average return by quantile (bars: 95% CI)')
    axes[0].legend()
    axes[1].axhline(1, color='gray', lw=0.8)
    axes[1].set_yscale('log')
    axes[1].set_title('Cumulative return of H-L (Q%d - Q1)' % n_q)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def run_quantile_analysis(ch, rtw, F, y, inv, n_q=5, weight='vw', signals=('Y_t', 'Yhat'),
                          size=None, bp_mask=None, rf=None, plot=True, nw_lags=12, **kw):
    """
    指定した分位数で、各シグナルのポートフォリオを作り、表・検定・図を出す
    signals: 'Y_t'（現在の収益性）、'Yhat'（期待収益性）、'rev'（リビジョン = Ŷ − Y_t）から選択
    size: None / 'big' / 'small'
    kw: add_expected に渡す引数（例：winsor_dep=True, xvars='prof'）
    """
    sig = prepare_signals(ch, y, inv, size=size, **kw)
    col_of = {'Y_t': y, 'Yhat': 'Yhat', 'rev': 'rev'}
    tag = f'{y}・{n_q}分位・{"時価総額加重" if weight == "vw" else "等ウェイト"}' + \
          (f'・{size}' if size else '・全銘柄')

    results = {}
    for name in signals:
        var = col_of[name]
        R, N = quantile_portfolios(sig, rtw, var, n_q, weight, bp_mask)
        results[name] = R
        show(f'[分位ポートフォリオ] {tag}：{name} でソート',
             summarize_portfolios(R, N, F, rf, nw_lags),
             note='毎年6月に分位を作り、7月〜翌6月の月次リターンを計算。平均・アルファは %/月、t値は Newey-West。\n'
                  'H-L = 最上位 − 最下位（ゼロコストのロングショート）。H-L の t値が約2以上なら、スプレッドは有意。')
        show(f'[単調性] {tag}：{name}', monotonicity(R).to_frame('値'),
             note=f'分位の平均リターンが {n_q - 1} 段のうち何段で上昇したか。順位相関が 1 に近いほど単調。')
        show(f'[分位の特性] {tag}：{name}', quantile_characteristics(sig, var, n_q, bp_mask=bp_mask),
             note='各分位の平均的な特性。時価総額（億円）や B/M に大きな偏りがないか確認する。')

    if 'Yhat' in results and 'Y_t' in results:
        show(f'[比較] {tag}：H-L の差（Ŷ − Y_t）', diff_test(results['Yhat'], results['Y_t'], nw_lags).to_frame('値'),
             note='Ŷ でソートしたスプレッドが、Y_t でソートしたスプレッドより有意に大きいか（同じ銘柄・同じ月で比較）。')

    if plot:
        plot_quantiles(results, n_q, f'{y} | {n_q} quantiles | {weight.upper()} | {size or "all"}')
    return results, sig


# %% セルB：準備（ウェイト付きリターン・自前ファクター）
rtw = make_rtw(rt, df_mkt_cap)
RF = None   # 無リスク金利（月次・小数、index=Period[M]）があれば設定。H-L のアルファには影響しない
F = build_factors(ch, rtw, rf=RF)


# %% セルC：メイン（5分位・時価総額加重・全銘柄）
N_Q = 5          # ← 分位数はここで指定（例：3, 5, 10）
WEIGHT = 'vw'    # 'vw'：時価総額加重 ／ 'ew'：等ウェイト

res_cbop, sig_cbop = run_quantile_analysis(ch, rtw, F, 'CBOP', 'dA', n_q=N_Q, weight=WEIGHT)
res_cp, sig_cp = run_quantile_analysis(ch, rtw, F, 'CP', 'dB', n_q=N_Q, weight=WEIGHT, winsor_dep=True)


# %% セルD：頑健性（分位数・加重・規模別）
# 10分位
run_quantile_analysis(ch, rtw, F, 'CBOP', 'dA', n_q=10, weight='vw')

# 等ウェイト
run_quantile_analysis(ch, rtw, F, 'CBOP', 'dA', n_q=N_Q, weight='ew')

# 規模別（各年の時価総額中央値で分割）
run_quantile_analysis(ch, rtw, F, 'CBOP', 'dA', n_q=N_Q, weight='vw', size='big')
run_quantile_analysis(ch, rtw, F, 'CBOP', 'dA', n_q=N_Q, weight='vw', size='small')

# 収益性だけで作った Ŷ
run_quantile_analysis(ch, rtw, F, 'CBOP', 'dA', n_q=N_Q, weight='vw', xvars='prof')

# リビジョン（Ŷ − Y_t）でソート：収益性の水準と強い負の相関があるので、結果の解釈に注意
run_quantile_analysis(ch, rtw, F, 'CBOP', 'dA', n_q=N_Q, weight='vw', signals=('Y_t', 'Yhat', 'rev'))
