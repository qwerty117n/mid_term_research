# %% [markdown]
# H3: 期待収益性（Ŷ）で作ったポートフォリオ / RMW は、現在収益性（Y_t）より大きなプレミアム・アルファを持つか
# 論文対応: Appendix Table A1（p.49, 本文 p.21）、RMW の議論（§4.3 末尾 p.16-17）
# 前提: Table 3 ノートブックの ch, rt と、table7_japan.py セルA の関数（add_expected, nw_mean, PROF）が定義済み
# 各 "# %%" ごとにノートブックのセルへ貼り付けて実行する

# %% セルA：関数定義
import numpy as np
import pandas as pd
import statsmodels.api as sm


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
    """月 × keys ごとの時価総額加重リターン"""
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


def build_signals(ch, y, inv, **kw):
    """Y_t と Ŷ が両方そろう銘柄（比較のため同一ユニバース）"""
    yhat, _ = add_expected(ch, y, inv, **kw)
    sig = ch.merge(yhat, on=['security_code', 'form_year'], how='inner')
    return sig.dropna(subset=[y, 'Yhat', 'ME_jun'])


def quintile_table(sig, rtw, var, F, bp_mask=None, rf=None, nw_lags=12):
    """var の 5 分位・時価総額加重ポートフォリオ（Table A1 相当）"""
    s = sig.copy()
    s['q'] = assign_bins(s, var, [0.2, 0.4, 0.6, 0.8], bp_mask)
    s = s.dropna(subset=['q'])
    s['q'] = s['q'].astype(int)
    p = rtw.merge(s[['security_code', 'form_year', 'q']], on=['security_code', 'form_year'])
    R = vw_ret(p, ['q'])
    R.columns = [f'Q{c + 1}' for c in R.columns]
    R['H-L'] = R['Q5'] - R['Q1']

    rows = {}
    for c in R.columns:
        r = R[c] if (rf is None or c == 'H-L') else R[c] - rf.reindex(R.index)
        est, _ = ts_reg(R[c], None, nw_lags)
        rows[c] = {'Mean': f"{est.loc['alpha', 'coef'] * 100:.2f} ({abs(est.loc['alpha', 't']):.2f})"}
        for name, cols in [('FF3 α', ['MKT', 'SMB', 'HML']), ('FF4 α', ['MKT', 'SMB', 'HML', 'CMA'])]:
            est, _ = ts_reg(r, F[cols], nw_lags)
            rows[c][name] = f"{est.loc['alpha', 'coef'] * 100:.2f} ({abs(est.loc['alpha', 't']):.2f})"
    return pd.DataFrame(rows), R


def rmw_compare(sig, rtw, y, F, bp_mask=None, nw_lags=12):
    """RMW を Y_t と Ŷ で構築して比較（平均・アルファ・相互スパニング）"""
    rmw_y, _ = two_by_three(sig, rtw, y, bp_mask)
    rmw_h, _ = two_by_three(sig, rtw, 'Yhat', bp_mask)
    G = pd.concat([F, rmw_y.rename('RMW_Y'), rmw_h.rename('RMW_Yhat')], axis=1).dropna()
    base = ['MKT', 'SMB', 'HML', 'CMA']
    specs = {
        'RMW_Y    : 平均':              ('RMW_Y', None),
        'RMW_Yhat : 平均':              ('RMW_Yhat', None),
        'RMW_Y    ~ FF4':               ('RMW_Y', base),
        'RMW_Yhat ~ FF4':               ('RMW_Yhat', base),
        'RMW_Yhat ~ FF4 + RMW_Y':       ('RMW_Yhat', base + ['RMW_Y']),
        'RMW_Y    ~ FF4 + RMW_Yhat':    ('RMW_Y', base + ['RMW_Yhat']),
    }
    rows = {}
    for name, (dep, xs) in specs.items():
        est, r2 = ts_reg(G[dep], None if xs is None else G[xs], nw_lags)
        row = {'α or 平均 (%/月)': f"{est.loc['alpha', 'coef'] * 100:.2f} ({abs(est.loc['alpha', 't']):.2f})"}
        if xs is None:
            row['年率Sharpe'] = f"{G[dep].mean() / G[dep].std() * np.sqrt(12):.2f}"
        else:
            row['Adj-R2'] = f'{r2:.2f}'
            other = [c for c in xs if c.startswith('RMW')]
            if other:
                row[f'β {other[0]}'] = f"{est.loc[other[0], 'coef']:.2f} ({abs(est.loc[other[0], 't']):.2f})"
        rows[name] = row
    print('corr(RMW_Y, RMW_Yhat) =', round(G['RMW_Y'].corr(G['RMW_Yhat']), 3), '| 月数 =', len(G))
    return pd.DataFrame(rows).T.fillna('–'), G


# %% セルB：準備（ウェイト付きリターン・自前ファクター）
rtw = make_rtw(rt, df_mkt_cap)

RF = None   # 無リスク金利（月次・小数、index=Period[M] の Series）があれば設定。H-L / RMW には影響しない
F = build_factors(ch, rtw, rf=RF)

# ファクターの妥当性チェック（日本では HML が強く正、SMB は弱いのが通常）
display(pd.DataFrame({c: ts_reg(F[c])[0].loc['alpha'] for c in F.columns}).T
          .assign(coef=lambda d: d['coef'] * 100).round(2))


# %% セルC：5 分位ポートフォリオ（Table A1 相当）
for y, inv in PROF.items():
    sig = build_signals(ch, y, inv)   # CP で 1 段目が不安定なら build_signals(ch, y, inv, winsor_dep=True)
    for var in [y, 'Yhat']:
        tab, R = quintile_table(sig, rtw, var, F, rf=RF)
        print(f'===== {y}: sorted on {var} =====')
        display(tab)


# %% セルD：RMW の比較（Y_t 版 vs Ŷ 版）
for y, inv in PROF.items():
    sig = build_signals(ch, y, inv)
    print(f'===== {y} =====')
    tab, G = rmw_compare(sig, rtw, y, F)
    display(tab)
