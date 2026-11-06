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
    show(f'[1段目] 将来収益性の予測回帰：{y}（Table 1 列(3)相当）',
         first_stage_summary(B).to_frame('平均係数 (t値)'),
         note='Y_{t+1} を Y_t, ΔY_t, ln(B/M), ln(ME), 投資 に年次クロスセクション回帰した係数の平均。\n'
              'ΔY_t が負なら「収益性は平均回帰する」。')

B_cp, _ = first_stage(ch, 'CP', 'dB')
show('[1段目] CP の年別係数（外れ値の年がないかの確認）', B_cp.round(3),
     note='特定の年だけ係数が極端に大きい・小さい場合は、被説明変数の外れ値が疑われる。')

show('[1段目] CP の分布（裾の長さの確認）',
     ch['CP'].describe(percentiles=[.001, .01, .5, .99, .999]).to_frame('CP'),
     note='0.1% 点・99.9% 点が中央値から極端に離れていれば、外れ値の影響が大きい。')

B_cp_w, _ = first_stage(ch, 'CP', 'dB', winsor_dep=True)
show('[1段目] CP：被説明変数も winsorize した場合（winsor_dep=True）',
     first_stage_summary(B_cp_w).to_frame('平均係数 (t値)'),
     note='上の CP の結果より t 値が上がり安定すれば、CP の不安定さは外れ値が原因。')


# %% セルC：Table 7 メイン（全銘柄・等加重・timing='safe'）
panel7_a, store7_a = table7(ch, rt)
show('[Table 7 メイン] 全銘柄・等ウェイト', panel7_a,
     note='月次リターンの Fama-MacBeth 回帰。係数×100、括弧内は Newey-West t 値（絶対値）。\n'
          '列(1)：現在の収益性 Y_t のみ ／ 列(2)：Y_t と期待収益性 Ŷ の両方 ／ 列(3)：Ŷ のみ。\n'
          '列(3)の Ŷ が列(1)の Y_t より大きく有意なら H1 を支持。列(2)で Ŷ が残れば H2 を支持。')

for y in PROF:
    s = store7_a[y]['sample'].drop_duplicates(['security_code', 'form_year'])
    show(f'[Table 7 補足] {y}：Y_t と Ŷ の分布', s[[y, 'Yhat']].describe().T,
         note=f'corr(Y_t, Ŷ) = {s[[y, "Yhat"]].corr().iloc[0, 1]:.3f}\n'
              'Ŷ は予測値なので Y_t より散らばりが小さいのが普通。')


# %% セルD：頑健性
panel7_b, _ = table7(ch, rt, universe_mask=lambda s: pd.Series(True, index=s.index))
show('[頑健性] Panel B：マイクロキャップ除外・等ウェイト', panel7_b,
     note='毎年6月の時価総額が下位20%の銘柄を除外（暫定基準：全銘柄の20パーセンタイル）。\n'
          '結果が残れば、極小型株だけの現象ではない。')

panel7_vw, _ = table7(ch, rt, weight='ME_jun')
show('[頑健性] 時価総額加重（Table 8 相当）', panel7_vw,
     note='6月末時価総額をウェイトにした回帰。大型株の影響が大きくなる。')

panel7_prof, _ = table7(ch, rt, xvars='prof')
show('[頑健性] Ŷ を Y_t と ΔY_t だけで作った場合（xvars="prof"）', panel7_prof,
     note='規模・投資・B/M の情報を除いた Ŷ。結果が残れば、改善の源は収益性の平均回帰。')

panel7_paper, _ = table7(ch, rt, timing='paper')
show('[頑健性] 論文どおりのタイミング（timing="paper"）', panel7_paper,
     note='翌年の係数を使い、1年後のリターンに適用（論文 p.19 の記述）。メインは先読みのない "safe"。')

panel7_wd, _ = table7(ch, rt, winsor_dep=True)
show('[頑健性] 1段目の被説明変数も winsorize（winsor_dep=True）', panel7_wd,
     note='CP の外れ値対策。論文からの変更点。')

panel7_avg3, _ = table7(ch, rt, avg_years=3)
show('[頑健性] 1段目の係数を直近3年平均（avg_years=3）', panel7_avg3,
     note='予測式の係数のぶれをならした場合。')
