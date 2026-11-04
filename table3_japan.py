"""
Wahal (2025) "Expected Profitability" Table 3 の日本市場版

前提とする入力（sample_code.py / get_pl_data で取得したもの）:
    df_rtn_features : DATE, TICKER, RTN, month(Period[M]), ret, R2_12, R1_1
    df_bs           : stock_id, security_code, fiscal_year_end_yyyymm, accounting_type,
                      settlement_flg, fiscal_period_months, total_assets, book_equity,
                      receivables, inventories, prepaid_expenses, accounts_payable,
                      accrued_expenses_proxy, deferred_revenue_current, deferred_revenue_noncurrent
    df_pl           : stock_id, security_code, fiscal_year_end_yyyymm, accounting_type,
                      settlement_flg, fiscal_period_months, industry,
                      operating_income, interest_expense, research_and_development_expense
    df_mkt_cap      : date(YYYYMM), market_cap, ticker

使い方（notebook）:
    from table3_japan import run_checks, build_characteristics, table3
    run_checks(df_rtn_features, df_bs, df_pl, df_mkt_cap)
    ch, rt = build_characteristics(df_rtn_features, df_bs, df_pl, df_mkt_cap,
                                   exclude_ind=[...], rule='ff')
    panel_a = table3(ch, rt)
    panel_b = table3(ch, rt, universe_mask=lambda s: pd.Series(True, index=s.index))  # 暫定
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

KEY = ['stock_id', 'accounting_type', 'fiscal_year_end_yyyymm']
CTRL = ['lnBM', 'lnME', 'R1_1', 'R2_12']


# ============================================================
# 0. 事前チェック
# ============================================================
def run_checks(df_rtn_features, df_bs, df_pl, df_mkt_cap):
    print('rtn  dup (month, TICKER):', df_rtn_features.duplicated(['month', 'TICKER']).sum())
    print('mcap dup (date, ticker) :', df_mkt_cap.duplicated(['date', 'ticker']).sum())
    print('bs   dup (KEY)          :', df_bs.duplicated(KEY).sum())
    print('pl   dup (KEY)          :', df_pl.duplicated(KEY).sum())
    print('rtn  期間:', df_rtn_features['month'].min(), '-', df_rtn_features['month'].max())
    print('mcap 期間:', df_mkt_cap['date'].min(), '-', df_mkt_cap['date'].max())
    print('\n会計基準:\n', df_bs['accounting_type'].value_counts())
    print('\n決算月数:\n', df_bs['fiscal_period_months'].value_counts().head())
    n_std = df_bs.groupby('stock_id')['accounting_type'].nunique()
    print('\n会計基準変更のあった企業数:', (n_std > 1).sum())
    print('ID一致率 (security_code in TICKER):',
          round(df_bs['security_code'].isin(set(df_rtn_features['TICKER'])).mean(), 4))
    yr = df_pl['fiscal_year_end_yyyymm'] // 100
    zero = (df_pl.assign(yr=yr)
                 .groupby('yr')[['interest_expense', 'research_and_development_expense']]
                 .apply(lambda g: (g == 0).mean()))
    print('\n支払利息・研究開発費 = 0 の割合（5年おき）:\n', zero.iloc[::5].round(3))
    print('\n年別企業数（5年おき）:\n',
          df_bs.groupby(df_bs['fiscal_year_end_yyyymm'] // 100)['stock_id'].nunique().iloc[::5])


# ============================================================
# 1. 会計パネル（企業 × 決算期）
# ============================================================
def _add_lag(df, cols, k):
    """同一企業・同一会計基準で、決算期がちょうど k 年前の値を結合"""
    lag = df[KEY + cols].copy()
    lag['fiscal_year_end_yyyymm'] += 100 * k
    return df.merge(lag.rename(columns={c: f'{c}_lag{k}' for c in cols}), on=KEY, how='left')


def build_accounting_panel(df_bs, df_pl):
    acc = df_bs.merge(
        df_pl[KEY + ['industry', 'operating_income', 'interest_expense',
                     'research_and_development_expense']],
        on=KEY, how='inner')
    assert not acc.duplicated(KEY).any(), 'KEY が一意ではありません'

    acc = acc[acc['fiscal_period_months'] == 12].copy()  # 変則決算を除外

    # 正味運転資本（アクルーアル対象）: RECT + INVT + XPP - (DRC + DRLT) - AP - XACC
    acc['nwc'] = (acc['receivables'] + acc['inventories'] + acc['prepaid_expenses']
                  - acc['deferred_revenue_current'] - acc['deferred_revenue_noncurrent']
                  - acc['accounts_payable'] - acc['accrued_expenses_proxy'])

    acc = _add_lag(acc, ['total_assets', 'book_equity', 'nwc'], 1)
    d_accr = acc['nwc'] - acc['nwc_lag1']
    ta0 = acc['total_assets_lag1'].where(acc['total_assets_lag1'] > 0)
    be0 = acc['book_equity_lag1'].where(acc['book_equity_lag1'] > 0)

    # CBOP = (営業利益 + R&D - Δアクルーアル) / 前期総資産
    # CP   = (営業利益 - 支払利息 - Δアクルーアル) / 前期自己資本
    acc['CBOP'] = (acc['operating_income'] + acc['research_and_development_expense'] - d_accr) / ta0
    acc['CP'] = (acc['operating_income'] - acc['interest_expense'] - d_accr) / be0
    acc['dA'] = acc['total_assets'] / ta0 - 1
    acc['dB'] = acc['book_equity'] / be0 - 1

    acc = _add_lag(acc, ['CBOP', 'CP'], 1)
    acc = _add_lag(acc, ['CBOP', 'CP'], 2)
    for y in ['CBOP', 'CP']:
        acc[f'd{y}'] = acc[y] - acc[f'{y}_lag1']

    # 会計基準変更年の重複: 必要変数が揃う行 → 新基準（最終決算期の基準）の行を優先
    req = ['CBOP', 'CBOP_lag1', 'CBOP_lag2', 'CP', 'CP_lag1', 'CP_lag2']
    acc['_n_ok'] = acc[req].notna().sum(axis=1)
    last_std = (acc.sort_values('fiscal_year_end_yyyymm')
                   .groupby('stock_id')['accounting_type'].last())
    acc['_is_new'] = acc['accounting_type'].eq(acc['stock_id'].map(last_std)).astype(int)
    acc = (acc.sort_values(['stock_id', 'fiscal_year_end_yyyymm', '_n_ok', '_is_new'],
                           ascending=[True, True, False, False])
              .drop_duplicates(['stock_id', 'fiscal_year_end_yyyymm'])
              .drop(columns=['_n_ok', '_is_new']))
    return acc


# ============================================================
# 2-3. 形成年への割り当て + 市場変数
# ============================================================
def build_characteristics(df_rtn_features, df_bs, df_pl, df_mkt_cap,
                          exclude_ind=(), rule='ff'):
    """
    rule='ff': t 年 6 月に、暦年 t-1 に終了した決算期を使う（論文どおり）
    rule='jp': t 年 6 月に、t 年 3 月までに終了した決算期を使う
    """
    acc = build_accounting_panel(df_bs, df_pl)

    fy = acc['fiscal_year_end_yyyymm'] // 100
    fm = acc['fiscal_year_end_yyyymm'] % 100
    if rule == 'ff':
        acc['form_year'] = fy + 1
    elif rule == 'jp':
        acc['form_year'] = np.where(fm <= 3, fy, fy + 1)
    else:
        raise ValueError(rule)
    acc = (acc.dropna(subset=['security_code'])
              .sort_values('fiscal_year_end_yyyymm')
              .drop_duplicates(['security_code', 'form_year'], keep='last'))

    mcap = df_mkt_cap.rename(columns={'ticker': 'security_code'})
    me_jun = (mcap[mcap['date'] % 100 == 6]
              .assign(form_year=lambda d: d['date'] // 100)
              .rename(columns={'market_cap': 'ME_jun'})[['security_code', 'form_year', 'ME_jun']])
    me_fye = (mcap.rename(columns={'date': 'fiscal_year_end_yyyymm', 'market_cap': 'ME_fye'})
                  [['security_code', 'fiscal_year_end_yyyymm', 'ME_fye']])

    ch = acc.merge(me_jun, on=['security_code', 'form_year'], how='inner')
    ch = ch.merge(me_fye, on=['security_code', 'fiscal_year_end_yyyymm'], how='left')
    ch['lnME'] = np.log(ch['ME_jun'].where(ch['ME_jun'] > 0))
    ch['lnBM'] = np.log(ch['book_equity'].where(ch['book_equity'] > 0) / ch['ME_fye'])

    # 金融・公益を除外（業種欠損も除外）
    ch = ch[ch['industry'].notna() & ~ch['industry'].isin(list(exclude_ind))].copy()

    rt = df_rtn_features.rename(columns={'TICKER': 'security_code'})
    rt = rt[['security_code', 'month', 'ret', 'R1_1', 'R2_12']].copy()
    rt['form_year'] = np.where(rt['month'].dt.month >= 7,
                               rt['month'].dt.year, rt['month'].dt.year - 1)
    return ch, rt


# ============================================================
# 4. サンプル構築（winsorize + 月次リターン結合）
# ============================================================
def _winsorize_by(df, cols, by='form_year', p=0.01):
    for c in cols:
        g = df.groupby(by)[c]
        lo = g.transform(lambda s: s.quantile(p))
        hi = g.transform(lambda s: s.quantile(1 - p))
        df[c] = df[c].clip(lower=lo, upper=hi)
    return df


def build_sample(ch, rt, y, inv, universe_mask=None):
    """y: 'CBOP' or 'CP', inv: 'dA' or 'dB'。列(1)-(3)で共通のサンプルを返す"""
    cols = [y, f'{y}_lag1', f'{y}_lag2', f'd{y}', inv, 'lnBM', 'lnME']
    s = ch.dropna(subset=cols).copy()  # 2 期前まで必須
    if universe_mask is not None:      # マイクロキャップ除外（Panel B）
        bp = (s[universe_mask(s)].groupby('form_year')['ME_jun'].quantile(0.2)
              .rename('bp').reset_index())
        s = s.merge(bp, on='form_year')
        s = s[s['ME_jun'] >= s['bp']].copy()
    s = _winsorize_by(s, [y, f'{y}_lag1', f'{y}_lag2', f'd{y}', inv])
    s = s.merge(rt, on=['security_code', 'form_year'], how='inner')
    return s.dropna(subset=['ret', 'R1_1', 'R2_12'])


# ============================================================
# 5. Fama-MacBeth
# ============================================================
def fama_macbeth(df, xs, yvar='ret', nw_lags=12):
    rows = []
    for m, g in df.groupby('month'):
        X = np.column_stack([np.ones(len(g)), g[xs].to_numpy(dtype=float)])
        yv = g[yvar].to_numpy(dtype=float)
        n, k = X.shape
        if n < k + 10:
            continue
        b, *_ = np.linalg.lstsq(X, yv, rcond=None)
        e = yv - X @ b
        adj_r2 = 1 - (e @ e / (n - k)) / yv.var(ddof=1)
        rows.append({'month': m, **dict(zip(xs, b[1:])), 'adj_r2': adj_r2, 'n': n})
    C = pd.DataFrame(rows).set_index('month')

    out = {}
    for c in xs:
        res = sm.OLS(C[c].to_numpy(), np.ones(len(C))).fit(
            cov_type='HAC', cov_kwds={'maxlags': nw_lags})
        out[c] = (res.params[0] * 100, res.tvalues[0])  # 係数は ×100
    return out, C


def table3(ch, rt, universe_mask=None, nw_lags=12):
    """Table 3 の 1 パネル分（CBOP / CP × 列(1)-(3)）を返す"""
    row_order = ['Y_t', 'Y_t-1', 'dY_t', 'lnBM', 'lnME', 'inv', 'R1_1', 'R2_12',
                 'Avg. Adj-R2', 'Avg. N', 'Months']
    res, coefs = {}, {}
    for y, inv in [('CBOP', 'dA'), ('CP', 'dB')]:
        s = build_sample(ch, rt, y, inv, universe_mask)
        rename = {y: 'Y_t', f'{y}_lag1': 'Y_t-1', f'd{y}': 'dY_t', inv: 'inv'}
        for col, xs in [('(1)', [y]), ('(2)', [y, f'{y}_lag1']), ('(3)', [y, f'd{y}'])]:
            out, C = fama_macbeth(s, xs + CTRL + [inv], nw_lags=nw_lags)
            cell = {rename.get(k, k): f'{v[0]:.2f} ({abs(v[1]):.2f})' for k, v in out.items()}
            cell['Avg. Adj-R2'] = f"{C['adj_r2'].mean():.3f}"
            cell['Avg. N'] = f"{C['n'].mean():,.0f}"
            cell['Months'] = f'{len(C)}'
            res[(y, col)] = cell
            coefs[(y, col)] = C
    table = pd.DataFrame(res).reindex(row_order).fillna('–')
    return table, coefs


def check_reparam(coefs, y):
    """列(2)と列(3)の整合性: (3)Y = (2)Y + (2)Y_lag1, (3)dY = -(2)Y_lag1"""
    c2, c3 = coefs[(y, '(2)')].mean() * 100, coefs[(y, '(3)')].mean() * 100
    print(f'[{y}] (3)Y_t={c3[y]:.3f} vs (2)Y_t+(2)Y_t-1={c2[y] + c2[f"{y}_lag1"]:.3f} | '
          f'(3)dY={c3[f"d{y}"]:.3f} vs -(2)Y_t-1={-c2[f"{y}_lag1"]:.3f}')
