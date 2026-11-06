# %% [markdown]
# Wahal (2025) "Expected Profitability" Table 7 の日本市場版
# 前提: Table 3 のノートブックで ch, rt を作成済みであること
# 各 "# %%" ごとにノートブックのセルへ貼り付けて実行する

# %% セルA：関数定義
import numpy as np
import pandas as pd
import statsmodels.api as sm

CTRL = ['lnBM', 'lnME', 'R1_1', 'R2_12']
PROF = {'CBOP': 'dA', 'CP': 'dB'}


def winsorize_by(df, cols, by='form_year', p=0.01):
    for c in cols:
        g = df.groupby(by)[c]
        lo = g.transform(lambda s: s.quantile(p))
        hi = g.transform(lambda s: s.quantile(1 - p))
        df[c] = df[c].clip(lower=lo, upper=hi)
    return df


def nw_mean(x, nw_lags=12):
    """時系列平均と Newey-West t 値"""
    x = pd.Series(x).dropna().to_numpy(float)
    res = sm.OLS(x, np.ones(len(x))).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
    return res.params[0], res.tvalues[0]


# ---------- 1段目：将来収益性予測 ----------
def first_stage(ch, y, inv, xvars='full', winsor_dep=False):
    """
    各年: Y_τ を X_{τ-1} に回帰。返り値 B の index は est_year = τ
    （Y_τ が判明する形成年。τ 年 6 月には推定可能）
    xvars='full': Y, ΔY, lnBM, lnME, 投資（論文の式(5)）
    xvars='prof': Y, ΔY のみ（規模・投資・B/M の情報を除いた確認用）
    winsor_dep: 被説明変数 Y_τ も年ごとに winsorize（論文からの変更。CP の外れ値対策）
    """
    X_cols = [y, f'd{y}', 'lnBM', 'lnME', inv] if xvars == 'full' else [y, f'd{y}']
    base = ch[['stock_id', 'fiscal_year_end_yyyymm'] + X_cols].dropna().copy()
    base['_fy'] = base['fiscal_year_end_yyyymm'] // 100
    base = winsorize_by(base, X_cols, by='_fy')  # 説明変数を年ごとに winsorize

    # 被説明変数: 同一企業のちょうど 1 年後の決算期の Y
    lead = ch[['stock_id', 'fiscal_year_end_yyyymm', 'form_year', y]].dropna().copy()
    lead = lead.rename(columns={y: 'y_lead', 'form_year': 'est_year'})
    lead['fiscal_year_end_yyyymm'] -= 100
    pair = base.merge(lead, on=['stock_id', 'fiscal_year_end_yyyymm'], how='inner')
    if winsor_dep:
        pair = winsorize_by(pair, ['y_lead'], by='est_year')

    rows = []
    for t, g in pair.groupby('est_year'):
        X = np.column_stack([np.ones(len(g)), g[X_cols].to_numpy(float)])
        yv = g['y_lead'].to_numpy(float)
        n, k = X.shape
        if n < k + 10:
            continue
        b, *_ = np.linalg.lstsq(X, yv, rcond=None)
        e = yv - X @ b
        adj_r2 = 1 - (e @ e / (n - k)) / yv.var(ddof=1)
        rows.append({'est_year': t, 'const': b[0], **dict(zip(X_cols, b[1:])),
                     'adj_r2': adj_r2, 'n': n})
    return pd.DataFrame(rows).set_index('est_year').sort_index(), X_cols


def first_stage_summary(B, nw_lags=2):
    """1段目係数の平均（Table 1 の列(3)に相当）"""
    out = {}
    for c in [c for c in B.columns if c not in ('adj_r2', 'n')]:
        m, t = nw_mean(B[c], nw_lags)
        out[c] = f'{m:.3f} ({abs(t):.2f})'
    out['Avg. Adj-R2'] = f"{B['adj_r2'].mean():.3f}"
    out['Avg. N'] = f"{B['n'].mean():,.0f}"
    out['Years'] = len(B)
    return pd.Series(out)


def add_expected(ch, y, inv, xvars='full', timing='safe', avg_years=1, winsor_dep=False):
    """
    期待収益性 Yhat を作成
    timing='safe' : b_τ（τ 年 6 月に既知）× X_τ → τ 年 7 月からのリターンに使用
    timing='paper': b_{τ+1} × X_τ → τ+1 年 7 月からのリターンに使用（論文 p.19 の記述）
    avg_years: 直近何年分の係数を平均して使うか（1 = 最新年のみ）
    """
    B, X_cols = first_stage(ch, y, inv, xvars, winsor_dep)
    Bs = (B[['const'] + X_cols].rolling(avg_years, min_periods=1).mean()
          .add_prefix('b_'))

    base = ch[['security_code', 'fiscal_year_end_yyyymm', 'form_year'] + X_cols].dropna().copy()
    base['_fy'] = base['fiscal_year_end_yyyymm'] // 100
    base = winsorize_by(base, X_cols, by='_fy')  # 1段目と同じ処理
    shift = 0 if timing == 'safe' else 1
    base['coef_year'] = base['form_year'] + shift
    base = base.merge(Bs, left_on='coef_year', right_index=True, how='inner')
    base['Yhat'] = base['b_const'] + sum(base[f'b_{c}'] * base[c] for c in X_cols)

    out = base[['security_code', 'coef_year', 'Yhat']].rename(columns={'coef_year': 'form_year'})
    return out, B


# ---------- 2段目：Fama-MacBeth（加重対応） ----------
def fm_reg(df, xs, yvar='ret', w=None, nw_lags=12):
    rows = []
    for m, g in df.groupby('month'):
        X = np.column_stack([np.ones(len(g)), g[xs].to_numpy(float)])
        yv = g[yvar].to_numpy(float)
        n, k = X.shape
        if n < k + 10:
            continue
        wt = np.ones(n) if w is None else g[w].to_numpy(float)
        sw = np.sqrt(wt)
        b, *_ = np.linalg.lstsq(X * sw[:, None], yv * sw, rcond=None)
        e = yv - X @ b
        ybar = np.average(yv, weights=wt)
        adj_r2 = 1 - (np.sum(wt * e**2) / (n - k)) / (np.sum(wt * (yv - ybar)**2) / (n - 1))
        rows.append({'month': m, **dict(zip(xs, b[1:])), 'adj_r2': adj_r2, 'n': n})
    C = pd.DataFrame(rows).set_index('month')
    out = {c: nw_mean(C[c], nw_lags) for c in xs}
    return out, C


def build_sample7(ch, rt, y, inv, yhat, universe_mask=None):
    s = ch.merge(yhat, on=['security_code', 'form_year'], how='inner')
    s = s.dropna(subset=[y, 'Yhat', inv, 'lnBM', 'lnME']).copy()
    if universe_mask is not None:  # マイクロキャップ除外
        bp = (s[universe_mask(s)].groupby('form_year')['ME_jun'].quantile(0.2)
              .rename('bp').reset_index())
        s = s.merge(bp, on='form_year')
        s = s[s['ME_jun'] >= s['bp']].copy()
    s = winsorize_by(s, [y, inv])
    s = s.merge(rt, on=['security_code', 'form_year'], how='inner')
    return s.dropna(subset=['ret', 'R1_1', 'R2_12'])


def iqr_return(s, C, var):
    """各月の係数 × その月の四分位範囲（%/月）"""
    q = s.groupby('month')[var].quantile([0.25, 0.75]).unstack()
    iqr = (q[0.75] - q[0.25]).reindex(C.index)
    return C[var] * iqr * 100


def table7(ch, rt, universe_mask=None, weight=None, xvars='full',
           timing='safe', avg_years=1, winsor_dep=False, nw_lags=12):
    row_order = ['Y_t', 'Yhat', 'lnBM', 'lnME', 'inv', 'R1_1', 'R2_12',
                 'Avg. Adj-R2', 'Avg. N', 'Months', 'IQR ret', 'IQR diff (3)-(1)']
    res, store = {}, {}
    for y, inv in PROF.items():
        yhat, B = add_expected(ch, y, inv, xvars, timing, avg_years, winsor_dep)
        s = build_sample7(ch, rt, y, inv, yhat, universe_mask)
        rename = {y: 'Y_t', inv: 'inv'}
        Cs = {}
        for col, xs in [('(1)', [y]), ('(2)', [y, 'Yhat']), ('(3)', ['Yhat'])]:
            out, C = fm_reg(s, xs + CTRL + [inv], w=weight, nw_lags=nw_lags)
            Cs[col] = C
            cell = {rename.get(k, k): f'{m*100:.2f} ({abs(t):.2f})' for k, (m, t) in out.items()}
            cell['Avg. Adj-R2'] = f"{C['adj_r2'].mean():.3f}"
            cell['Avg. N'] = f"{C['n'].mean():,.0f}"
            cell['Months'] = f'{len(C)}'
            res[(y, col)] = cell

        iqr_y = iqr_return(s, Cs['(1)'], y)
        iqr_h = iqr_return(s, Cs['(3)'], 'Yhat')
        d_m, d_t = nw_mean((iqr_h - iqr_y).dropna(), nw_lags)
        res[(y, '(1)')]['IQR ret'] = f'{iqr_y.mean():.2f}'
        res[(y, '(3)')]['IQR ret'] = f'{iqr_h.mean():.2f}'
        res[(y, '(3)')]['IQR diff (3)-(1)'] = f'{d_m:.2f} ({abs(d_t):.2f})'
        store[y] = {'sample': s, 'first_stage': B, 'coefs': Cs}

    table = pd.DataFrame(res).reindex(row_order).fillna('–')
    return table, store


# %% セルB：1段目の確認（Table 1 の列(3)相当）
for y, inv in PROF.items():
    B, _ = first_stage(ch, y, inv, xvars='full')
    print(f'===== {y} =====')
    display(first_stage_summary(B).to_frame('avg (t)'))

# CP の外れ値チェック：年別係数と分布
B_cp, _ = first_stage(ch, 'CP', 'dB')
display(B_cp.round(3))
print(ch['CP'].describe(percentiles=[.001, .01, .5, .99, .999]))

# 被説明変数も winsorize した場合の 1段目（CP が安定するか）
B_cp_w, _ = first_stage(ch, 'CP', 'dB', winsor_dep=True)
display(first_stage_summary(B_cp_w).to_frame('CP, winsor_dep=True'))


# %% セルC：Table 7 メイン（全銘柄・等加重・timing='safe'）
panel7_a, store7_a = table7(ch, rt)
display(panel7_a)

# Ŷ と Y_t の分布・相関
for y in PROF:
    s = store7_a[y]['sample'].drop_duplicates(['security_code', 'form_year'])
    print(y, 'corr(Y, Yhat) =', round(s[[y, 'Yhat']].corr().iloc[0, 1], 3))
    display(s[[y, 'Yhat']].describe().T)


# %% セルD：頑健性
# Panel B：マイクロキャップ除外（暫定基準。市場区分データ取得後に差し替え）
panel7_b, _ = table7(ch, rt, universe_mask=lambda s: pd.Series(True, index=s.index))
display(panel7_b)

# 時価総額加重（Table 8 相当）
panel7_vw, _ = table7(ch, rt, weight='ME_jun')
display(panel7_vw)

# 収益性のみで作った Ŷ（規模・投資・B/M の情報を除いた確認用）
panel7_prof, _ = table7(ch, rt, xvars='prof')
display(panel7_prof)

# 論文どおりのタイミング
panel7_paper, _ = table7(ch, rt, timing='paper')
display(panel7_paper)

# 1段目の被説明変数も winsorize（CP の外れ値対策）
panel7_wd, _ = table7(ch, rt, winsor_dep=True)
display(panel7_wd)

# 1段目係数を直近3年平均で安定化
panel7_avg3, _ = table7(ch, rt, avg_years=3)
display(panel7_avg3)
